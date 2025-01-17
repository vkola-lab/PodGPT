# coding=utf-8
#
# GNU Affero General Public License v3.0 License
#
# PodGPT: An Audio-augmented Large Language Model for Research and Education
# Copyright (C) 2024 Kolachalama Laboratory at Boston University

import gc
import yaml
import contextlib

import torch
from sqlalchemy import select
from transformers import AutoModelForCausalLM, AutoTokenizer

from lib.database import get_session, PMCMetadata


def rag_formatting(original_text, documents, abstract=None):
    """
    Formats query and documents for Retrieval-Augmented Generation (RAG), removing duplicates.

    :param original_text: str, the user query or input text.
    :param documents: list, a list of retrieved documents.
    :return: str, a formatted string combining the query and document texts.
    """
    if documents:
        # Remove duplicates based on the 'text' field
        unique_documents = []
        seen_texts = set()
        for doc in documents:
            if doc['text'] not in seen_texts:
                unique_documents.append(doc)
                seen_texts.add(doc['text'])

        # Sort unique documents by 'score' in descending order
        unique_documents = sorted(unique_documents, key=lambda x: x['score'], reverse=True)

        # document_texts = [doc['text'] for doc in unique_documents]
        # Use only the top 3 documents if there are more than 3
        document_texts = [doc['text'] for doc in unique_documents[:3]]

        documents_text = "\n".join(document_texts)
        if abstract:
            modified_query = f"Documents:\n{abstract}\n{documents_text}\n\nQuestion:\n{original_text}"
        else:
            modified_query = f"Documents:\n{documents_text}\n\nQuestion:\n{original_text}"
    else:
        if abstract:
            modified_query = f"Abstract: {abstract}\nQuestion: {original_text}"
        else:
            modified_query = f"{original_text}"

    return modified_query


def format_documents(docs):
    """
    Formats the retrieved and ranked documents into a JSON structure.
    Now includes full article text from PMCFullArticles table.

    :param docs: list, a list of documents with their metadata and scores.
    :return: list, a list of JSON objects containing document details.
    """
    documents = []
    with get_session() as session:
        for doc in docs:
            # Retrieve metadata for each document using its PMC ID
            metadata = session.scalars(select(PMCMetadata).filter_by(accessionid=doc['pmc_id'])).first()

            if metadata:
                # Add formatted document details to the output, now including full article text
                documents.append({
                    "pmid": metadata.pmid,
                    "text": doc['text'],
                    "mla_citation": metadata.citation_mla,
                    "score": doc['score']
                })

    return documents


def download_pretrained_model(config):
    """
    Initialize model
    :param config: the YAML configuration file
    """
    model_name = config.get("model_name")
    hf_read_token = config.get("hf_read_token")
    save_dir = config.get("save_dir")

    # Load the base model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        use_auth_token=hf_read_token,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        return_tensors="pt",
        use_fast=False,
        use_auth_token=hf_read_token,
        device_map="cpu",
    )

    # Since there are PAD tokens in the latest LLMs, such as LLaMA 3.3
    # We are now using "if not tokenizer.pad_token_id" instead of
    # directly using "tokenizer.pad_token = tokenizer.eos_token"
    # https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct/blob/main/tokenizer_config.json#L2062
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    # if "llama" in model_name.lower() or "mistralai" in model_name.lower():
    #     tokenizer.pad_token = tokenizer.eos_token

    # Print the number of parameters
    print_parameters(model=model)

    # Save the tokenizer
    tokenizer.save_pretrained(save_dir)
    print('Successfully save the tokenizer!')

    # Save the pre-trained model
    model.save_pretrained(save_dir)
    print('Successfully save the model!\n\n')

    # Although we are loading the model and tokenizer to "cpu"
    # We still add these CUDA cache cleaning codes
    # Feel free to remove these if you don't need
    del model
    del tokenizer
    with contextlib.suppress(AssertionError):
        torch.distributed.destroy_process_group()
    gc.collect()
    torch.cuda.empty_cache()


def stop_token_list():
    """
    The stop token list for vLLM engine
    Note: You can add more stop tokens
    if you are using other LLMs that have stop tokens
    """
    stop_tokens = [
        "Question:",
    ]

    return stop_tokens


def load_config(file_name):
    """
    Load parameters and path from the YAML file
    :param file_name: the name of the YAML file
    :return config: The configuration info
    """
    fopen = open(file_name)
    config = yaml.load(fopen, Loader=yaml.FullLoader)
    fopen.close()

    return config


def print_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    # Retrieve a list of all named parameters in the model
    model_parameters = list(model.named_parameters())

    # Calculate the total number of parameters using a generator expression
    all_param = sum(p.numel() for _, p in model_parameters)

    # Calculate the total number of trainable parameters using a generator expression
    # that filters parameters which require gradients
    trainable_params = sum(p.numel() for _, p in model_parameters if p.requires_grad)

    # Print out the number of trainable parameters, total parameters,
    # and the percentage of parameters that are trainable
    # The percentage is formatted to two decimal places
    print(
        f"Trainable params: {trainable_params:,} | "
        f"All params: {all_param:,} | "
        f"Trainable%: {100 * trainable_params / all_param:.2f}%"
    )


class CustomStream:
    """
    Save all the running logs
    """

    def __init__(self, filename, console_stream):
        self.filename = filename
        self.console_stream = console_stream

    def write(self, text):
        with open(self.filename, 'a') as file:
            file.write(text)
        self.console_stream.write(text)

    def flush(self):
        pass


def prompt_template(tokenizer=None, input=None):
    """
    Prompt Template
    Gemma: https://ai.google.dev/gemma/docs/formatting
    LLaMA 3: https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3/
    Mistrial: https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2
    :param tokenizer: the tokenizer
    :param input: User input question and content
    :return prompt: Prompt Template with query
    """
    messages = [
        {
            "role": "user", "content": input
        }
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        # We add the generation prompt ("<start_of_turn>model") in the Prompt during training
        # to be consistent with Model Inference
        add_generation_prompt=True
    )

    return prompt


def str2bool(v):
    """
    Convert a string or boolean input into a boolean value.

    This function is useful when parsing command-line arguments where
    boolean flags might be passed as strings (e.g., "true", "false").

    :param v: The input value to be converted. Can be a string or boolean.
    :return: A boolean value corresponding to the input.
    :raises argparse.ArgumentTypeError: If the input string does not match any valid boolean representations.
    """
    # Check if the input is already a boolean, and return it as-is
    if isinstance(v, bool):
        return v

    # Convert string input to lowercase and check for "truthy" values
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    # Check for "falsy" values
    elif v.lower() in ("no", "false", "f", "0"):
        return False
    # Raise an error for invalid boolean representations
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")
