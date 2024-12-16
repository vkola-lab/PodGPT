# coding=utf-8
#
# GNU Affero General Public License v3.0 License
#
# PodGPT: An Audio-augmented Large Language Model for Research and Education
# Copyright (C) 2024 Kolachalama Laboratory at Boston University

import re
import math
import json

import jsonlines
import pandas as pd
from vllm.distributed.parallel_state import (
    destroy_model_parallel,
    destroy_distributed_environment,
)

from lib.pipeline import Pipeline
from utils.vllm_utils import performance_eval
from utils.utils import *
from utils.benchmark_utils import *


# Pre-defined prompts references
# Leaderboard: https://huggingface.co/spaces/openlifescienceai/open_medical_llm_leaderboard
# Language Model Evaluation Harness: https://github.com/EleutherAI/lm-evaluation-harness
# Notice 1: We use `"Directly answer the best option:"` instead of `Answer:`
# to better guide LLMs to generate the best option and to easier extract the best option from the responses
prompt_w_doc = "Directly answer the best option with explanations:"
prompt_wo_doc = "Directly answer the best option:"
prompt_pubmedqa_w_doc = "\nInstruction:\nDirectly answer yes/no/maybe with explanations:"
prompt_pubmedqa_wo_doc = "\nInstruction:\nDirectly answer yes/no/maybe:"


def medqa_eval(config, mode, tokenizer, file_path, data_path, batch_size=32):
    """
    Performance Evaluation on the MedQA Benchmark with Batch Processing
    :param config: the configuration file
    :param mode: one of "small" or "large"
        "small": Evaluate the smaller models
        "large": Evaluate the larger models (with the LoRA Adapter)
    :param tokenizer: the tokenizer
    :param file_path: save file path and file name
    :param data_path: testing dataset path
    :param batch_size: number of queries to process in each batch
    """
    prompts = []
    answers = []
    documents = []

    # Initialize the pipeline
    rag_pipeline = Pipeline()

    # Read all items from the dataset
    with open(data_path, "r+", encoding="utf-8") as f:
        items = list(jsonlines.Reader(f))  # Load all data into a list

    # Split the data into batches
    num_batches = math.ceil(len(items) / batch_size)

    for batch_idx in range(num_batches):
        batch_items = items[batch_idx * batch_size:(batch_idx + 1) * batch_size]
        batch_queries = [medqa_format(item) for item in batch_items]  # Extract queries

        # Batch retrieve documents
        batch_documents = rag_pipeline.retrieve_documents(
            original_text=[
                re.sub(r'A\..*', '', query, flags=re.DOTALL).strip() for query in batch_queries
            ]
        )

        for query, document, item in zip(batch_queries, batch_documents, batch_items):
            rag_query = rag_formatting(original_text=query, documents=document)
            if document:
                temp_ins = prompt_template(tokenizer=tokenizer, input=rag_query + prompt_w_doc)
            else:
                temp_ins = prompt_template(tokenizer=tokenizer, input=rag_query + prompt_wo_doc)

            prompts.append(temp_ins)  # Collect prompts
            answers.append(item["answer_idx"])  # Collect answers
            documents.append(document)  # Collect documents

    # Release the GPU cache
    del rag_pipeline
    destroy_model_parallel()
    destroy_distributed_environment()
    with contextlib.suppress(AssertionError):
        torch.distributed.destroy_process_group()
    gc.collect()
    torch.cuda.empty_cache()

    # Performance evaluation
    performance_eval(config, mode, prompts, answers, documents, file_path)


def pubmedqa_eval(config, mode, tokenizer, file_path, data_path, batch_size=32):
    """
    Performance Evaluation on the PubMedQA Benchmark
    :param config: the configuration file
    :param mode: one of "small", "large", or "chatgpt"
        "small": Evaluate the smaller models
        "large": Evaluate the larger models (with the LoRA Adapter)
    :param tokenizer: the tokenizer
    :param file_path: save file path and file name
    :param data_path: testing dataset path
    """
    prompts = []
    answers = []
    documents = []

    # Initialize the pipeline
    rag_pipeline = Pipeline()

    items = []
    with open(data_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

        # Iterate through each item
        for key, item in data.items():
            items.append(item)

    # Split the data into batches
    num_batches = math.ceil(len(items) / batch_size)

    for batch_idx in range(num_batches):
        batch_items = items[batch_idx * batch_size:(batch_idx + 1) * batch_size]
        batch_abstracts = [pubmedqa_format(item)[0] for item in batch_items]  # Extract abstracts
        batch_queries = [pubmedqa_format(item)[1] for item in batch_items]  # Extract queries

        # Batch retrieve documents
        batch_documents = rag_pipeline.retrieve_documents(
            original_text=[query.strip() for query in batch_queries]
        )

        for query, abstract, document, item in zip(batch_queries, batch_abstracts, batch_documents, batch_items):
            rag_query = rag_formatting(original_text=query, documents=document, abstract=abstract)
            if document:
                temp_ins = prompt_template(tokenizer=tokenizer, input=rag_query + prompt_pubmedqa_w_doc)
            else:
                temp_ins = prompt_template(tokenizer=tokenizer, input=rag_query + prompt_pubmedqa_wo_doc)

            prompts.append(temp_ins)  # Collect prompts
            temp_ans = item["final_decision"]
            if temp_ans == "yes":
                temp_ans = 'A'
            elif temp_ans == "no":
                temp_ans = 'B'
            else:
                temp_ans = 'C'
            answers.append(temp_ans)  # Collect answers
            documents.append(document)  # Collect documents

    # Release the GPU cache
    del rag_pipeline
    destroy_model_parallel()
    destroy_distributed_environment()
    with contextlib.suppress(AssertionError):
        torch.distributed.destroy_process_group()
    gc.collect()
    torch.cuda.empty_cache()

    # Performance evaluation
    performance_eval(config, mode, prompts, answers, documents, file_path)


def medmcqa_eval(config, mode, tokenizer, file_path, data_path, batch_size=32):
    """
    Performance Evaluation on the MedMCQA Benchmark with Batch Processing
    :param config: the configuration file
    :param mode: one of "small" or "large"
        "small": Evaluate the smaller models
        "large": Evaluate the larger models (with the LoRA Adapter)
    :param tokenizer: the tokenizer
    :param file_path: save file path and file name
    :param data_path: testing dataset path
    :param batch_size: number of queries to process in each batch
    """
    prompts = []
    answers = []
    documents = []

    # Initialize the pipeline
    rag_pipeline = Pipeline()

    items = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))

    # Split the data into batches
    num_batches = math.ceil(len(items) / batch_size)

    for batch_idx in range(num_batches):
        batch_items = items[batch_idx * batch_size:(batch_idx + 1) * batch_size]
        batch_queries = [medmcqa_format(item) for item in batch_items]  # Extract queries

        # Batch retrieve documents
        batch_documents = rag_pipeline.retrieve_documents(
            original_text=[
                re.sub(r'A\..*', '', query, flags=re.DOTALL).strip() for query in batch_queries
            ]
        )

        for query, document, item in zip(batch_queries, batch_documents, batch_items):
            rag_query = rag_formatting(original_text=query, documents=document)
            if document:
                temp_ins = prompt_template(tokenizer=tokenizer, input=rag_query + prompt_w_doc)
            else:
                temp_ins = prompt_template(tokenizer=tokenizer, input=rag_query + prompt_wo_doc)

            prompts.append(temp_ins)  # Collect prompts
            temp_ans = item["cop"]
            if temp_ans == 1:
                temp_ans = 'A'
            elif temp_ans == 2:
                temp_ans = 'B'
            elif temp_ans == 3:
                temp_ans = 'C'
            else:
                temp_ans = 'D'
            answers.append(temp_ans)  # Collect answers
            documents.append(document)  # Collect documents

    # Release the GPU cache
    del rag_pipeline
    destroy_model_parallel()
    destroy_distributed_environment()
    with contextlib.suppress(AssertionError):
        torch.distributed.destroy_process_group()
    gc.collect()
    torch.cuda.empty_cache()

    # Performance evaluation
    performance_eval(config, mode, prompts, answers, documents, file_path)


def medexpqa_eval(config, mode, tokenizer, file_path, data_path, batch_size=32):
    """
    Performance Evaluation on the MedExpQA Benchmark
    :param config: the configuration file
    :param mode: one of "small" or "large"
        "small": Evaluate the smaller models
        "large": Evaluate the larger models (with the LoRA Adapter)
    :param tokenizer: the tokenizer
    :param file_path: save file path and file name
    :param data_path: testing dataset path
    :param batch_size: number of queries to process in each batch
    """
    prompts = []
    answers = []
    documents = []

    # Initialize the pipeline
    rag_pipeline = Pipeline()

    # Read all items from the dataset
    with open(data_path, "r+", encoding="utf-8") as f:
        items = list(jsonlines.Reader(f))  # Load all data into a list

    # Split the data into batches
    num_batches = math.ceil(len(items) / batch_size)

    for batch_idx in range(num_batches):
        batch_items = items[batch_idx * batch_size:(batch_idx + 1) * batch_size]
        batch_queries = [medexpqa_format(item) for item in batch_items]  # Extract queries

        # Batch retrieve documents
        batch_documents = rag_pipeline.retrieve_documents(
            original_text=[
                re.sub(r'A\..*', '', query, flags=re.DOTALL).strip() for query in batch_queries
            ]
        )

        for query, document, item in zip(batch_queries, batch_documents, batch_items):
            rag_query = rag_formatting(original_text=query, documents=document)
            if document:
                temp_ins = prompt_template(tokenizer=tokenizer, input=rag_query + prompt_w_doc)
            else:
                temp_ins = prompt_template(tokenizer=tokenizer, input=rag_query + prompt_wo_doc)

            prompts.append(temp_ins)  # Collect prompts
            temp_ans = item["correct_option"]
            if temp_ans == 1:
                temp_ans = 'A'
            elif temp_ans == 2:
                temp_ans = 'B'
            elif temp_ans == 3:
                temp_ans = 'C'
            elif temp_ans == 4:
                temp_ans = 'D'
            else:
                temp_ans = 'E'
            answers.append(temp_ans)  # Collect answers
            documents.append(document)  # Collect documents

    # Release the GPU cache
    del rag_pipeline
    destroy_model_parallel()
    destroy_distributed_environment()
    with contextlib.suppress(AssertionError):
        torch.distributed.destroy_process_group()
    gc.collect()
    torch.cuda.empty_cache()

    # Performance evaluation
    performance_eval(config, mode, prompts, answers, documents, file_path)


def mmlu_eval(config, mode, tokenizer, file_path, data_path, batch_size=32):
    """
    Performance Evaluation on the MMLU Benchmark
    :param config: the configuration file
    :param mode: one of "small" or "large"
        "small": Evaluate the smaller models
        "large": Evaluate the larger models (with the LoRA Adapter)
    :param tokenizer: the tokenizer
    :param file_path: save file path and file name
    :param data_path: testing dataset path
    :param batch_size: number of queries to process in each batch
    """
    prompts = []
    answers = []
    documents = []

    # Initialize the pipeline
    rag_pipeline = Pipeline()

    # Read all items from the dataset
    items = []
    csv = pd.read_csv(data_path, header=None)
    for index, item in csv.iterrows():
        items.append(item)

    # Split the data into batches
    num_batches = math.ceil(len(items) / batch_size)

    for batch_idx in range(num_batches):
        batch_items = items[batch_idx * batch_size:(batch_idx + 1) * batch_size]
        batch_queries = [mmlu_format(item) for item in batch_items]  # Extract queries

        # Batch retrieve documents
        batch_documents = rag_pipeline.retrieve_documents(
            original_text=[
                re.sub(r'A\..*', '', query, flags=re.DOTALL).strip() for query in batch_queries
            ]
        )

        for query, document, item in zip(batch_queries, batch_documents, batch_items):
            rag_query = rag_formatting(original_text=query, documents=document)
            if document:
                temp_ins = prompt_template(tokenizer=tokenizer, input=rag_query + prompt_w_doc)
            else:
                temp_ins = prompt_template(tokenizer=tokenizer, input=rag_query + prompt_wo_doc)

            prompts.append(temp_ins)  # Collect prompts
            answers.append(item[5])  # Collect answers
            documents.append(document)  # Collect documents

    # Release the GPU cache
    del rag_pipeline
    destroy_model_parallel()
    destroy_distributed_environment()
    with contextlib.suppress(AssertionError):
        torch.distributed.destroy_process_group()
    gc.collect()
    torch.cuda.empty_cache()

    # Performance evaluation
    performance_eval(config, mode, prompts, answers, documents, file_path)
