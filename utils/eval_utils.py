# coding=utf-8
#
# GNU Affero General Public License v3.0 License
#
# PodGPT: An Audio-augmented Large Language Model for Research and Education
# Copyright (C) 2024 Kolachalama Laboratory at Boston University

import json

import jsonlines
import pandas as pd

from utils.utils import *
from utils.benchmark_utils import *
from utils.vllm_utils import performance_eval


def medqa_eval(config, mode, tokenizer, file_path, data_path):
    """
    Performance Evaluation on the MedQA Benchmark
    :param config: the configuration file
    :param mode: one of "small", "large", or "chatgpt"
        "small": Evaluate the smaller models
        "large": Evaluate the larger models (with the LoRA Adapter)
        "chatgpt": Evaluate the OpenAI ChatGPT
    :param tokenizer: the tokenizer
    :param file_path: save file path and file name
    :param data_path: testing dataset path
    """
    model_name = config.get("model_name")

    prompts = []
    answers = []

    with open(data_path, "r+", encoding="utf-8") as f:
        for idx, item in enumerate(jsonlines.Reader(f)):
            question = medqa_format(item)
            if "medalpaca" in model_name:
                temp_ins = prompt_template_medAlpaca(input=question)
            elif "MMed-Llama" in model_name:
                temp_ins = prompt_template_MMedLM(input=question, language="English")
            else:
                temp_ins = prompt_template(tokenizer=tokenizer, input=question)
            prompts.append(temp_ins)

            # Get the label answer
            temp_ans = item["answer_idx"]
            answers.append(temp_ans)

    # Performance evaluation
    performance_eval(config=config, mode=mode, prompts=prompts, answers=answers, file_path=file_path)


def pubmedqa_eval(config, mode, tokenizer, file_path, data_path):
    """
    Performance Evaluation on the PubMedQA Benchmark
    :param config: the configuration file
    :param mode: one of "small", "large", or "chatgpt"
        "small": Evaluate the smaller models
        "large": Evaluate the larger models (with the LoRA Adapter)
        "chatgpt": Evaluate the OpenAI ChatGPT
    :param tokenizer: the tokenizer
    :param file_path: save file path and file name
    :param data_path: testing dataset path
    """
    model_name = config.get("model_name")

    prompts = []
    answers = []

    with open(data_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

        # Iterate through each item and print the final decision
        for key, item in data.items():
            question = pubmedqa_format(item)
            if "medalpaca" in model_name:
                temp_ins = prompt_template_medAlpaca(input=question)
            elif "MMed-Llama" in model_name:
                temp_ins = prompt_template_MMedLM(input=question, language="English")
            else:
                temp_ins = prompt_template(tokenizer=tokenizer, input=question)
            prompts.append(temp_ins)

            # Get the label answer
            temp_ans = item["final_decision"]
            if temp_ans == "yes":
                temp_ans = 'A'
            elif temp_ans == "no":
                temp_ans = 'B'
            else:
                temp_ans = 'C'
            answers.append(temp_ans)

    # Performance evaluation
    performance_eval(config=config, mode=mode, prompts=prompts, answers=answers, file_path=file_path)


def medmcqa_eval(config, mode, tokenizer, file_path, data_path):
    """
    Performance Evaluation on the MedMCQA Benchmark
    :param config: the configuration file
    :param mode: one of "small", "large", or "chatgpt"
        "small": Evaluate the smaller models
        "large": Evaluate the larger models (with the LoRA Adapter)
        "chatgpt": Evaluate the OpenAI ChatGPT
    :param tokenizer: the tokenizer
    :param file_path: save file path and file name
    :param data_path: testing dataset path
    """
    model_name = config.get("model_name")

    prompts = []
    answers = []

    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            question = medmcqa_format(item)
            if "medalpaca" in model_name:
                temp_ins = prompt_template_medAlpaca(input=question)
            elif "MMed-Llama" in model_name:
                temp_ins = prompt_template_MMedLM(input=question, language="English")
            else:
                temp_ins = prompt_template(tokenizer=tokenizer, input=question)
            prompts.append(temp_ins)

            # Get the label answer
            temp_ans = item["cop"]
            if temp_ans == 1:
                temp_ans = 'A'
            elif temp_ans == 2:
                temp_ans = 'B'
            elif temp_ans == 3:
                temp_ans = 'C'
            else:
                temp_ans = 'D'
            answers.append(temp_ans)

    # Performance evaluation
    performance_eval(config=config, mode=mode, prompts=prompts, answers=answers, file_path=file_path)


def usmle_eval(config, mode, tokenizer, file_path, data_path):
    """
    Performance Evaluation on the Internal USMLE QA (Lindsey, Divya, and Meagan)
    :param config: the configuration file
    :param mode: one of "small", "large", or "chatgpt"
        "small": Evaluate the smaller models
        "large": Evaluate the larger models (with the LoRA Adapter)
        "chatgpt": Evaluate the OpenAI ChatGPT
    :param tokenizer: the tokenizer
    :param file_path: save file path and file name
    :param data_path: testing dataset path
    """
    model_name = config.get("model_name")

    prompts = []
    answers = []

    with open(data_path, 'r', encoding="utf-8") as f:
        contents = json.loads(f.read())
        for item in contents:
            question = usmle_format(item)
            if "medalpaca" in model_name:
                temp_ins = prompt_template_medAlpaca(input=question)
            elif "MMed-Llama" in model_name:
                temp_ins = prompt_template_MMedLM(input=question, language="English")
            else:
                temp_ins = prompt_template(tokenizer=tokenizer, input=question)
            prompts.append(temp_ins)

            # Get the label answer
            temp_ans = item["answer_id"]
            answers.append(temp_ans)

    # Performance evaluation
    performance_eval(config=config, mode=mode, prompts=prompts, answers=answers, file_path=file_path)


def medexpqa_eval(config, mode, tokenizer, file_path, data_path):
    """
    Performance Evaluation on the MedExpQA Benchmark
    :param config: the configuration file
    :param mode: one of "small", "large", or "chatgpt"
        "small": Evaluate the smaller models
        "large": Evaluate the larger models (with the LoRA Adapter)
        "chatgpt": Evaluate the OpenAI ChatGPT
    :param tokenizer: the tokenizer
    :param file_path: save file path and file name
    :param data_path: testing dataset path
    """
    model_name = config.get("model_name")

    prompts = []
    answers = []

    with open(data_path, "r+", encoding="utf-8") as f:
        for idx, item in enumerate(jsonlines.Reader(f)):
            if "spanish" in file_path.lower():
                question = medexpqa_format(item=item, lang="Spanish")
            elif "french" in file_path.lower():
                question = medexpqa_format(item=item, lang="French")
            else:
                question = medexpqa_format(item=item, lang="English")

            if "medalpaca" in model_name:
                temp_ins = prompt_template_medAlpaca(input=question)
            elif "MMed-Llama" in model_name:
                if "spanish" in file_path.lower():
                    temp_ins = prompt_template_MMedLM(input=question, language="Spanish")
                elif "french" in file_path.lower():
                    temp_ins = prompt_template_MMedLM(input=question, language="French")
                else:
                    temp_ins = prompt_template_MMedLM(input=question, language="English")
            else:
                temp_ins = prompt_template(tokenizer=tokenizer, input=question)
            prompts.append(temp_ins)

            # Get the label answer
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
            answers.append(temp_ans)

    # Performance evaluation
    performance_eval(config=config, mode=mode, prompts=prompts, answers=answers, file_path=file_path)


def mmlu_eval(config, mode, tokenizer, file_path, data_path):
    """
    Performance Evaluation on the MMLU Benchmark
    :param config: the configuration file
    :param mode: one of "small", "large", or "chatgpt"
        "small": Evaluate the smaller models
        "large": Evaluate the larger models (with the LoRA Adapter)
        "chatgpt": Evaluate the OpenAI ChatGPT
    :param tokenizer: the tokenizer
    :param file_path: save file path and file name
    :param data_path: testing dataset path
    """
    model_name = config.get("model_name")

    prompts = []
    answers = []

    csv = pd.read_csv(data_path, header=None)
    for index, item in csv.iterrows():
        if "hindi" in file_path.lower():
            question = mmlu_format(item=item, lang="Hindi")
        elif "spanish" in file_path.lower():
            question = mmlu_format(item=item, lang="Spanish")
        elif "french" in file_path.lower():
            question = mmlu_format(item=item, lang="French")
        else:
            question = mmlu_format(item=item, lang="English")

        if "medalpaca" in model_name:
            temp_ins = prompt_template_medAlpaca(input=question)
        elif "MMed-Llama" in model_name:
            if "hindi" in file_path.lower():
                temp_ins = prompt_template_MMedLM(input=question, language="Hindi")
            elif "spanish" in file_path.lower():
                temp_ins = prompt_template_MMedLM(input=question, language="Spanish")
            elif "french" in file_path.lower():
                temp_ins = prompt_template_MMedLM(input=question, language="French")
            else:
                temp_ins = prompt_template_MMedLM(input=question, language="English")
        else:
            temp_ins = prompt_template(tokenizer=tokenizer, input=question)
        prompts.append(temp_ins)

        # Get the label answer
        temp_ans = item[5]
        answers.append(temp_ans)

    # Performance evaluation
    performance_eval(config=config, mode=mode, prompts=prompts, answers=answers, file_path=file_path)


def mcmle_eval(config, mode, tokenizer, file_path, data_path):
    """
    Performance Evaluation on the MedQA-MCMLE Benchmark
    :param config: the configuration file
    :param mode: one of "small", "large", or "chatgpt"
        "small": Evaluate the smaller models
        "large": Evaluate the larger models (with the LoRA Adapter)
        "chatgpt": Evaluate the OpenAI ChatGPT
    :param tokenizer: the tokenizer
    :param file_path: save file path and file name
    :param data_path: testing dataset path
    """
    model_name = config.get("model_name")

    prompts = []
    answers = []

    with open(data_path, "r+", encoding="utf-8") as f:
        for idx, item in enumerate(jsonlines.Reader(f)):
            question = mcmle_format(item)
            if "medalpaca" in model_name:
                temp_ins = prompt_template_medAlpaca(input=question)
            elif "MMed-Llama" in model_name:
                temp_ins = prompt_template_MMedLM(input=question, language="Chinese")
            else:
                temp_ins = prompt_template(tokenizer=tokenizer, input=question)
            prompts.append(temp_ins)

            # Get the label answer
            temp_ans = item["answer_idx"]
            answers.append(temp_ans)

    # Performance evaluation
    performance_eval(config=config, mode=mode, prompts=prompts, answers=answers, file_path=file_path)


def cmmlu_eval(config, mode, tokenizer, file_path, data_path):
    """
    Performance Evaluation on the CMMLU Benchmark
    :param config: the configuration file
    :param mode: one of "small", "large", or "chatgpt"
        "small": Evaluate the smaller models
        "large": Evaluate the larger models (with the LoRA Adapter)
        "chatgpt": Evaluate the OpenAI ChatGPT
    :param tokenizer: the tokenizer
    :param file_path: save file path and file name
    :param data_path: testing dataset path
    """
    model_name = config.get("model_name")

    prompts = []
    answers = []

    csv = pd.read_csv(data_path, header=0)
    for index, item in csv.iterrows():
        question = cmmlu_format(item)
        if "medalpaca" in model_name:
            temp_ins = prompt_template_medAlpaca(input=question)
        elif "MMed-Llama" in model_name:
            temp_ins = prompt_template_MMedLM(input=question, language="Chinese")
        else:
            temp_ins = prompt_template(tokenizer=tokenizer, input=question)
        prompts.append(temp_ins)

        # Get the label answer
        temp_ans = item[6]
        answers.append(temp_ans)

    # Performance evaluation
    performance_eval(config=config, mode=mode, prompts=prompts, answers=answers, file_path=file_path)


def frenchmedmcqa_eval(config, mode, tokenizer, file_path, data_path):
    """
    Performance Evaluation on the FrenchMedMCQA Benchmark
    :param config: the configuration file
    :param mode: one of "small", "large", or "chatgpt"
        "small": Evaluate the smaller models
        "large": Evaluate the larger models (with the LoRA Adapter)
        "chatgpt": Evaluate the OpenAI ChatGPT
    :param tokenizer: the tokenizer
    :param file_path: save file path and file name
    :param data_path: testing dataset path
    """
    model_name = config.get("model_name")

    prompts = []
    answers = []

    with open(data_path, 'r', encoding="utf-8") as f:
        contents = json.loads(f.read())
        for item in contents:
            # We only use the single-answer subset
            if item["nbr_correct_answers"] == 1:
                question = frenchmedmcqa_format(item)
                if "medalpaca" in model_name:
                    temp_ins = prompt_template_medAlpaca(input=question)
                elif "MMed-Llama" in model_name:
                    temp_ins = prompt_template_MMedLM(input=question, language="French")
                else:
                    temp_ins = prompt_template(tokenizer=tokenizer, input=question)
                prompts.append(temp_ins)

                # Get the label answer
                temp_ans = item["correct_answers"][0]
                answers.append(temp_ans.upper())

    # Performance evaluation
    performance_eval(config=config, mode=mode, prompts=prompts, answers=answers, file_path=file_path)


def headqa_eval(config, mode, tokenizer, file_path, data_path):
    """
    Performance Evaluation on the HEAD-QA Benchmark
    :param config: the configuration file
    :param mode: one of "small", "large", or "chatgpt"
        "small": Evaluate the smaller models
        "large": Evaluate the larger models (with the LoRA Adapter)
        "chatgpt": Evaluate the OpenAI ChatGPT
    :param tokenizer: the tokenizer
    :param file_path: save file path and file name
    :param data_path: testing dataset path
    """
    model_name = config.get("model_name")

    prompts = []
    answers = []

    with open(data_path, 'r', encoding="utf-8") as f:
        contents = json.loads(f.read())
        for exams in contents["exams"]:
            for item in contents["exams"][exams]["data"]:
                question = headqa_format(item)
                if "medalpaca" in model_name:
                    temp_ins = prompt_template_medAlpaca(input=question)
                elif "MMed-Llama" in model_name:
                    temp_ins = prompt_template_MMedLM(input=question, language="Spanish")
                else:
                    temp_ins = prompt_template(tokenizer=tokenizer, input=question)
                prompts.append(temp_ins)

                # Get the label answer
                temp_ans = item["ra"]
                if temp_ans == "1":
                    temp_ans = 'A'
                elif temp_ans == "2":
                    temp_ans = 'B'
                elif temp_ans == "3":
                    temp_ans = 'C'
                else:
                    temp_ans = 'D'
                answers.append(temp_ans)

    # Performance evaluation
    performance_eval(config=config, mode=mode, prompts=prompts, answers=answers, file_path=file_path)
