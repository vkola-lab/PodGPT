# coding=utf-8
#
# GNU Affero General Public License v3.0 License
#
# PodGPT: An Audio-augmented Large Language Model for Research and Education
# Copyright (C) 2024 Kolachalama Laboratory at Boston University

import os

from transformers import AutoTokenizer

from utils.eval_utils import *


def evaluation(config, mode="podgpt", rag=True):
    """
    Conduct model inference, get the model's responses, and evaluate performance
    :param config: the configuration file
    :param mode: one of "podgpt" or "chatgpt"
        "podgpt": Evaluate PodGPT
        "chatgpt": Evaluate the OpenAI ChatGPT
    :param rag: whether to use the RAG database and pipeline
    """
    # Retrieve the pathes of needed hyperparameters
    model_name = config.get("model_name")
    result_dir = config.get("result_dir")

    # Load tokenizer
    if mode == "podgpt":
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    else:
        tokenizer = None
        print("The evaluation mode should be either `podgpt` or `chatgpt`!")

    # Get your database
    database = config.get("database")

    # Start the performance evaluation
    print("Start performance evaluation on your own database!")
    dataset = "database"
    test_path = database
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/"
    eval(config=config, mode=mode, rag=rag, tokenizer=tokenizer, file_path=file_path, data_path=test_path)
