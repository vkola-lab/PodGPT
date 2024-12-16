# coding=utf-8
#
# GNU Affero General Public License v3.0 License
#
# PodGPT: An Audio-augmented Large Language Model for Research and Education
# Copyright (C) 2024 Kolachalama Laboratory at Boston University
#
# BENCHMARKS AND CREDITS
#
# MedQA-USMLE
# We conducted performance evaluation on the MedQA-USMLE Benchmark
# Notice: We use the 4-option version of the dataset
# Paper: What Disease does this Patient Have? A Large-scale Open Domain Question Answering Dataset from Medical Exams
# Link: https://arxiv.org/abs/2009.13081
# Database and Codes: https://github.com/jind11/MedQA
#
# MedMCQA
# We conducted performance evaluation on the MedMCQA Benchmark
# Paper: MedMCQA : A Large-scale Multi-Subject Multi-Choice Dataset for Medical domain Question Answering
# Link: https://arxiv.org/abs/2203.14371
# Homepage: https://medmcqa.github.io/
# Database and Codes: https://github.com/MedMCQA/MedMCQA
#
# MedExpQA English
# We conducted performance evaluation on the MedExpQA (English) Benchmark
# Paper: MedExpQA: Multilingual Benchmarking of Large Language Models for Medical Question Answering
# Link: https://arxiv.org/abs/2404.05590
# Database and Codes: https://huggingface.co/datasets/HiTZ/MedExpQA
#
# MMLU Medicine Subset
# We conducted performance evaluation on medicine-related 6 databases
# anatomy_test, clinical_knowledge_test, college_biology_test
# college_medicine_test, medical_genetics_test, professional_medicine_test
# Paper: Measuring Massive Multitask Language Understanding
# Link: https://arxiv.org/abs/2009.03300
# Database and Codes: https://www.kaggle.com/datasets/lizhecheng/mmlu-dataset

import os

from utils.eval_utils import *
from utils.utils import download_pretrained_model


def evaluation(config, mode="small"):
    """
    Conduct model inference, get the model's responses, and evaluate performance
    :param config: the configuration file
    :param mode: one of "small", "large", or "chatgpt"
        "small": Evaluate the smaller models
        "large": Evaluate the larger models (with the LoRA Adapter)
        "chatgpt": Evaluate the OpenAI ChatGPT
    :param eval_pretrain: is this evaluating the pre-trained model?
    :param checkpoint_id: a list of checkpoint IDs
    """
    # Retrieve the pathes of needed hyperparameters
    model_name = config.get("model_name")
    result_dir = config.get("result_dir")
    eval_pretrain = config.get("eval_pretrain")

    # Evaluating the larger models
    if mode == "large":
        download_pretrained_model(config=config)

        # Load the tokenizer
        # Since there is no prompt template for Mixtral MoE, we will use `Mistral-7B-Instruct-v0.3` prompt template
        if "Mixtral-8x" in model_name:
            tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Evaluating the smaller models
    elif mode == "small":
        if eval_pretrain:
            print("Start to download the original pre-trained model and tokenizer!")
            download_pretrained_model(config=config)

        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Print the only three evaluation modes.
    else:
        tokenizer = None
        print("The evaluation mode should be one of small, large, or chatgpt.")

    # Benchmarks
    english_medicine_medqa = config.get("english_medicine_medqa")
    english_medicine_pubmedqa = config.get("english_medicine_pubmedqa")
    english_medicine_medmcqa = config.get("english_medicine_medmcqa")
    english_medicine_medexpqa = config.get("english_medicine_medexpqa")
    english_mmlu_medicine_anatomy = config.get("english_mmlu_medicine_anatomy")
    english_mmlu_medicine_clinical_knowledge = config.get("english_mmlu_medicine_clinical_knowledge")
    english_mmlu_medicine_college_biology = config.get("english_mmlu_medicine_college_biology")
    english_mmlu_medicine_college_medicine = config.get("english_mmlu_medicine_college_medicine")
    english_mmlu_medicine_medical_genetics = config.get("english_mmlu_medicine_medical_genetics")
    english_mmlu_medicine_professional_medicine = config.get("english_mmlu_medicine_professional_medicine")

    # Start the performance evaluation
    print("English Benchmark - Start performance evaluation on the MMLU Anatomy Benchmark")
    dataset = "rag_english_mmlu_medicine_anatomy"
    test_path = english_mmlu_medicine_anatomy
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/"
    mmlu_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("English Benchmark - Start performance evaluation on the MMLU Clinical Knowledge Benchmark")
    dataset = "rag_english_mmlu_medicine_clinical_knowledge"
    test_path = english_mmlu_medicine_clinical_knowledge
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/"
    mmlu_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("English Benchmark - Start performance evaluation on the MMLU College Biology Benchmark")
    dataset = "rag_english_mmlu_medicine_college_biology"
    test_path = english_mmlu_medicine_college_biology
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/"
    mmlu_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("English Benchmark - Start performance evaluation on the MMLU College Medicine Benchmark")
    dataset = "rag_english_mmlu_medicine_college_medicine"
    test_path = english_mmlu_medicine_college_medicine
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/"
    mmlu_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("English Benchmark - Start performance evaluation on the MMLU Medical Genetics Benchmark")
    dataset = "rag_english_mmlu_medicine_medical_genetics"
    test_path = english_mmlu_medicine_medical_genetics
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/"
    mmlu_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("English Benchmark - Start performance evaluation on the MMLU Professional Medicine Benchmark")
    dataset = "rag_english_mmlu_medicine_professional_medicine"
    test_path = english_mmlu_medicine_professional_medicine
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/"
    mmlu_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("English Benchmark - Start performance evaluation on the MedQA Benchmark")
    dataset = "rag_english_medicine_medqa"
    test_path = english_medicine_medqa
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/"
    medqa_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("English Benchmark - Start performance evaluation on the PubMedQA Benchmark")
    dataset = "rag_english_medicine_pubmedqa"
    test_path = english_medicine_pubmedqa
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/"
    pubmedqa_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("English Benchmark - Start performance evaluation on the MedMCQA Benchmark")
    dataset = "rag_english_medicine_medmcqa"
    test_path = english_medicine_medmcqa
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/"
    medmcqa_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("English Benchmark - Start performance evaluation on the MedExpQA English Benchmark")
    dataset = "rag_english_medicine_medexpqa"
    test_path = english_medicine_medexpqa
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/"
    medexpqa_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)
