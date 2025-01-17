# coding=utf-8
#
# GNU Affero General Public License v3.0 License
#
# PodGPT: An Audio-augmented Large Language Model for Research and Education
# Copyright (C) 2024 Kolachalama Laboratory at Boston University

import re
import gc
import math
import contextlib

import torch
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


def eval(config, mode, rag, tokenizer, file_path, data_path, batch_size=8):
    """
    Performance Evaluation on your database
    :param config: the configuration file
    :param mode: either "podgpt" or "chatgpt"
    :param rag: whether to use the RAG database and pipeline
    :param tokenizer: the tokenizer
    :param file_path: save file path and file name
    :param data_path: testing dataset path
    :param batch_size: number of queries to process in each batch
    """
    prompts = []
    answers = []
    documents = []

    # Read all items from the dataset
    items = []
    csv = pd.read_csv(data_path, header=None)
    if rag:
        # Initialize the pipeline
        rag_pipeline = Pipeline()

        # Split the data into batches
        for index, item in csv.iterrows():
            items.append(item)
        num_batches = math.ceil(len(items) / batch_size)

        for batch_idx in range(num_batches):
            batch_items = items[batch_idx * batch_size:(batch_idx + 1) * batch_size]
            batch_queries = [database_format(item) for item in batch_items]  # Extract queries

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

    else:
        for index, item in csv.iterrows():
            # Get the question
            question = database_format(item=item)
            temp_ins = prompt_template(tokenizer=tokenizer, input=question + prompt_wo_doc)
            prompts.append(temp_ins)

            # Get the label answer
            temp_ans = item[5]
            answers.append(temp_ans)
            documents.append(None)

    # Performance evaluation
    performance_eval(config, mode, prompts, answers, documents, file_path)
