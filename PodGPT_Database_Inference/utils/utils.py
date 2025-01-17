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
