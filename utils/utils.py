# coding=utf-8
#
# GNU Affero General Public License v3.0 License
#
# PodGPT: An Audio-augmented Large Language Model for Research and Education
# Copyright (C) 2024 Kolachalama Laboratory at Boston University

import gc
import yaml
import contextlib
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


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

    # Clean the cache
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


def prompt_template_medAlpaca(input=None):
    """
    Prompt Template for the medAlpaca model
    Evaluation codes: https://github.com/kbressem/medAlpaca/blob/main/eval/eval_usmle.py#L157-L162
    Prompt example: https://github.com/kbressem/medAlpaca/blob/main/medalpaca/handler.py#L170-L191
    :param input: User input question and content
    :return prompt: Prompt Template with query
    """
    prompt = (f"Below is an instruction that describes a task, paired with an input that provides further context. "
              f"Write a response that appropriately completes the request.\n\n"
              f"### Instruction:\nAnswer this multiple choice question.\n\n"
              f"### Input:\n{input}\n\n"
              f"### Response:\nThe Answer to the question is:")

    return prompt


def prompt_template_MMedLM(input=None, language="English"):
    """
    Prompt Template for the MMed-Llama-3-8B-EnIns model
    Evaluation codes: https://github.com/MAGIC-AI4Med/MMedLM/blob/main/inference/inference.py#L86-L91
    Prompt example: https://github.com/MAGIC-AI4Med/MMedLM/blob/main/inference/inference.py#L10-L21
    :param input: User input question and content
    :param language: The language of the benchmark, as used in the instruction,
                     "You're a {language} doctor, kindly address ..."
    :return prompt: Prompt Template with query
    """
    # Pre-defined prompts references
    # Leaderboard: https://huggingface.co/spaces/openlifescienceai/open_medical_llm_leaderboard
    # Language Model Evaluation Harness: https://github.com/EleutherAI/lm-evaluation-harness
    # Special Notice: We use `"Directly answer the best option:"` instead of `Answer:`
    # to better guide LLMs to generate the best option and to easier extract the best option from the responses
    english_prompt = "Directly answer the best option:"
    english_prompt_pubmedqa = "Directly answer yes/no/maybe:"
    hindi_prompt = "सीधे सबसे अच्छे विकल्प के साथ जवाब दें:"
    french_prompt = "Répondez directement avec la meilleure option:"
    spanish_prompt = "Responde directamente con la mejor opción:"
    chinese_prompt = "直接回答最优选项:"

    # https://github.com/MAGIC-AI4Med/MMedLM/blob/main/inference/inference.py#L87
    instruction = (f"You're a {language} doctor, kindly address the medical queries according to the patient's "
                   f"account. Answer with the best option directly.")

    # https://github.com/MAGIC-AI4Med/MMedLM/blob/main/inference/inference.py#L88
    if english_prompt_pubmedqa in input:
        # For the PubMedQA Benchmark
        question = input
        question = question.replace(english_prompt_pubmedqa, "")
        # Special Notice: We don't use "A. yes B. no C. maybe"
        # to align with our answer extraction codes
        options = "yes, no, or maybe"
    else:
        question = input.split("\nA.")[0]
        options = "\nA." + input.split("\nA.")[1]
        options = options.replace(english_prompt, "")
        options = options.replace(hindi_prompt, "")
        options = options.replace(spanish_prompt, "")
        options = options.replace(french_prompt, "")
        options = options.replace(chinese_prompt, "")

    input = (f"###Question: {question} Which of the following is the best treatment for this patient? "
             f"###Options: {options}")

    # https://github.com/MAGIC-AI4Med/MMedLM/blob/main/inference/inference.py#L11-L14
    prompt = (f"Below is an instruction that describes a task, paired with an input that provides further context. "
              f"Write a response that appropriately completes the request.\n\n"
              f"### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:")

    return prompt


def response_template(model_name, output):
    """
    Response Template
    Gemma: https://ai.google.dev/gemma/docs/formatting
    LLaMA 3: https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3/
    Mistrial: https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2
    :param output: Answer response
    :return: Response Template with response
    """
    if "gemma" in model_name:
        response = f"{output}<end_of_turn><eos>"
    elif "mistralai" in model_name:
        response = f"{output}</s>"
    elif "llama" in model_name:
        response = f"{output}<|eot_id|><|end_of_text|>"
    else:
        response = f"{output}"

    return response


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
