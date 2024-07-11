# coding=utf-8
#
# LLMs INFERENCE ACKNOWLEDGEMENT
# vLLM - Easy, fast, and cheap LLM serving for everyone
# https://github.com/vllm-project/vllm
#
# LICENSE OF THE INFERENCE ENGINE
# Apache 2.0 License
# https://github.com/vllm-project/vllm/blob/main/LICENSE

import time
import json

import jsonlines
import pandas as pd
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from utils.utils import stop_token_list, prompt_template
from utils.benchmark_utils import *
from utils.answer_utils import *


def performance_eval(config, llm, prompts, answers, file_path):
    """
    Generate responses by vLLM and conduct performance evaluation
    :param config: the configuration file
    :param llm: the vLLM engine
    :param prompts: prompted questions
    :param answers: ground truth answers
    :param file_path: save file path and file name
    """
    # Load some configurations
    model_name = config.get("model_name")
    max_new_tokens = config.get("max_new_tokens")
    eval_pretrain = config.get("eval_pretrain")
    lora_path = config.get("lora_path")

    start_t = time.time()
    responses = []

    if "QUANT" in model_name or "GPTQ" in model_name:
        sampling_params = SamplingParams(
            temperature=0,
            top_p=1,
            max_tokens=max_new_tokens,
            # https://huggingface.co/astronomer/Llama-3-8B-Instruct-GPTQ-8-Bit
            # For loading this model onto vLLM, make sure all requests have
            # "stop_token_ids":[128001, 128009] to temporarily address the non-stop generation issue.
            # vLLM does not yet respect generation_config.json.
            # vLLM team is working on a fix for this https://github.com/vllm-project/vllm/issues/4180
            stop_token_ids=[128001, 128009],
        )
    else:
        stop_tokens = stop_token_list()
        # https://github.com/vllm-project/vllm/blob/main/vllm/sampling_params.py#L38-L66
        sampling_params = SamplingParams(
            temperature=0,
            top_p=1,
            max_tokens=max_new_tokens,
            stop=stop_tokens
        )

    if eval_pretrain:
        # For the evaluation of the pre-trained model
        completions = llm.generate(
            prompts,
            sampling_params,
        )
    else:
        # For the evaluation of the LoRA model
        completions = llm.generate(
            prompts,
            sampling_params,
            lora_request=LoRARequest("adapter", 1, lora_path)
        )

    for i, output in enumerate(completions):
        temp_gen = output.outputs[0].text
        responses.append(temp_gen)
    print('Successfully finished generating', len(prompts), 'samples!')

    acc = []
    all_out = []  # Save all the model's responses
    invalid_out = []  # Only save the invalid responses
    for idx, (question, response, answer) in enumerate(zip(prompts, responses, answers)):
        # question: the medical question
        # response: the model's response
        # answer: the ground truth answer

        # Special Notice:
        # option_range="a-dA-D" means options A, B, C, D
        # option_range="a-eA-E" means options A, B, C, D, E
        # option_range="a-jA-J" means options A, B, C, D, E, F, G, H, I, J
        # English Benchmarks - 11 Benchmarks
        if "english_pubmedqa" in file_path.lower():
            # We separately write codes to evaluate our models' performance
            # on the PubMedQA benchmark (yes/no/maybe)
            prediction = extract_answer_for_pubmedqa(completion=response)
        elif ("english_mmlu" in file_path.lower()
              or "english_medmcqa" in file_path.lower()
              or "english_medqa" in file_path.lower()):
            prediction = extract_answer(completion=response, option_range="a-dA-D")  # 4 Options
            if prediction is None:
                response = response_with_option(prompt=question, response=response)
                prediction = extract_answer(completion=response, option_range="a-dA-D")  # 4 Options
        elif "english_medexpqa" in file_path.lower():
            prediction = extract_answer(completion=response, option_range="a-eA-E")  # 5 Options
            if prediction is None:
                response = response_with_option(prompt=question, response=response)
                prediction = extract_answer(completion=response, option_range="a-eA-E")  # 5 Options
        elif "english_usmle" in file_path.lower():
            # Most of the questions are with 5 options (A - E)
            # But there are a few minorities
            # 146: (F), 30: (G), 15: (H), 3: (I), 1: (J)
            prediction = extract_answer(completion=response, option_range="a-jA-J")
            if prediction is None:
                response = response_with_option(prompt=question, response=response)
                prediction = extract_answer(completion=response, option_range="a-jA-J")

        # Chinese Benchmarks - 8 Benchmarks
        elif "chinese" in file_path.lower():
            # CMMLU and MCMLE are all with 4 options
            prediction = extract_answer_for_chinese(completion=response, option_range="a-dA-D")
            if prediction is None:
                response = response_with_option(prompt=question, response=response)
                prediction = extract_answer_for_chinese(completion=response, option_range="a-dA-D")

        # French Benchmarks - 8 Benchmarks
        elif "french_mmlu" in file_path.lower():
            prediction = extract_answer_for_french(completion=response, option_range="a-dA-D")
            if prediction is None:
                response = response_with_option(prompt=question, response=response)
                prediction = extract_answer_for_french(completion=response, option_range="a-dA-D")
        elif "french_medexpqa" in file_path.lower() or "french_medmcqa" in file_path.lower():
            prediction = extract_answer_for_french(completion=response, option_range="a-eA-E")
            if prediction is None:
                response = response_with_option(prompt=question, response=response)
                prediction = extract_answer_for_french(completion=response, option_range="a-eA-E")

        # Spanish Benchmarks - 8 Benchmarks
        elif "spanish_mmlu" in file_path.lower() or "spanish_headqa" in file_path.lower():
            prediction = extract_answer_for_spanish(completion=response, option_range="a-dA-D")
            if prediction is None:
                response = response_with_option(prompt=question, response=response)
                prediction = extract_answer_for_spanish(completion=response, option_range="a-dA-D")
        elif "spanish_medexpqa" in file_path.lower():
            prediction = extract_answer_for_spanish(completion=response, option_range="a-eA-E")
            if prediction is None:
                response = response_with_option(prompt=question, response=response)
                prediction = extract_answer_for_spanish(completion=response, option_range="a-eA-E")

        # Hindi Benchmarks - 6 Benchmarks
        elif "hindi_mmlu":
            prediction = extract_answer_for_hindi(completion=response, option_range="a-dA-D")
            if prediction is None:
                response = response_with_option(prompt=question, response=response)
                prediction = extract_answer_for_hindi(completion=response, option_range="a-dA-D")

        if prediction is not None:
            acc.append(prediction == answer)
            temp = {
                'question': question,
                'output': response,
                'extracted answer': prediction,
                'answer': answer
            }
            all_out.append(temp)

        else:
            acc.append(False)
            temp = {
                'question': question,
                'output': response,
                'extracted answer': prediction,
                'answer': answer
            }
            all_out.append(temp)
            invalid_out.append(temp)

    accuracy = sum(acc) / len(acc)
    end_t = time.time()
    elapsed_t = end_t - start_t
    print(f"Finished performance evaluation in {elapsed_t:.2f} seconds")

    # Print the length of the invalid output and the accuracy
    print('Invalid output length:', len(invalid_out), ', Testing length:', len(acc), ', Accuracy:', accuracy)

    # Save the invalid output in a JSON file
    with open(file_path.replace('.json', '_invalid_responses.json'), 'w') as file:
        json.dump(invalid_out, file, ensure_ascii=False, indent=4)
    print('Successfully save the invalid output.')

    # Save all the responses in a JSON file
    with open(file_path.replace('.json', '_all_responses.json'), 'w') as file:
        json.dump(all_out, file, ensure_ascii=False, indent=4)
    print('Successfully save all the output.')


def medqa_eval(config, llm, tokenizer, file_path, data_path):
    """
    Performance Evaluation on the MedQA Benchmark
    :param config: the configuration file
    :param llm: the vLLM engine
    :param tokenizer: the tokenizer
    :param file_path: save file path and file name
    :param data_path: testing dataset path
    """
    prompts = []
    answers = []

    with open(data_path, "r+", encoding="utf8") as f:
        for idx, item in enumerate(jsonlines.Reader(f)):
            question = medqa_format(item)
            temp_ins = prompt_template(tokenizer=tokenizer, input=question)
            prompts.append(temp_ins)

            # Get the label answer
            temp_ans = item["answer_idx"]
            answers.append(temp_ans)

    performance_eval(config=config, llm=llm, prompts=prompts, answers=answers, file_path=file_path)


def pubmedqa_eval(config, llm, tokenizer, file_path, data_path):
    """
    Performance Evaluation on the PubMedQA Benchmark
    :param config: the configuration file
    :param llm: the vLLM engine
    :param tokenizer: the tokenizer
    :param file_path: save file path and file name
    :param data_path: testing dataset path
    """
    prompts = []
    answers = []

    with open(data_path, 'r', encoding='utf8') as file:
        data = json.load(file)

        # Iterate through each item and print the final decision
        for key, item in data.items():
            question = pubmedqa_format(item)
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

    performance_eval(config=config, llm=llm, prompts=prompts, answers=answers, file_path=file_path)


def medmcqa_eval(config, llm, tokenizer, file_path, data_path):
    """
    Performance Evaluation on the MedMCQA Benchmark
    :param config: the configuration file
    :param llm: the vLLM engine
    :param tokenizer: the tokenizer
    :param file_path: save file path and file name
    :param data_path: testing dataset path
    """
    prompts = []
    answers = []

    with open(data_path, "r", encoding="utf8") as f:
        for line in f:
            item = json.loads(line)
            question = medmcqa_format(item)
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

    performance_eval(config=config, llm=llm, prompts=prompts, answers=answers, file_path=file_path)


def usmle_eval(config, llm, tokenizer, file_path, data_path):
    """
    Performance Evaluation on the Internal USMLE QA (Lindsey, Divya, and Meagan)
    :param config: the configuration file
    :param llm: the vLLM engine
    :param tokenizer: the tokenizer
    :param file_path: save file path and file name
    :param data_path: testing dataset path
    """
    prompts = []
    answers = []

    with open(data_path, 'r') as f:
        contents = json.loads(f.read())
        for item in contents:
            question = usmle_format(item)
            temp_ins = prompt_template(tokenizer=tokenizer, input=question)
            prompts.append(temp_ins)

            # Get the label answer
            temp_ans = item["answer_id"]
            answers.append(temp_ans)

    performance_eval(config=config, llm=llm, prompts=prompts, answers=answers, file_path=file_path)


def medexpqa_eval(config, llm, tokenizer, file_path, data_path):
    """
    Performance Evaluation on the MedExpQA Benchmark
    :param config: the configuration file
    :param llm: the vLLM engine
    :param tokenizer: the tokenizer
    :param file_path: save file path and file name
    :param data_path: testing dataset path
    """
    prompts = []
    answers = []

    with open(data_path, "r+", encoding="utf8") as f:
        for idx, item in enumerate(jsonlines.Reader(f)):
            if "spanish" in file_path.lower():
                question = medexpqa_format(item=item, lang="Spanish")
            elif "french" in file_path.lower():
                question = medexpqa_format(item=item, lang="French")
            else:
                question = medexpqa_format(item=item, lang="English")

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

    performance_eval(config=config, llm=llm, prompts=prompts, answers=answers, file_path=file_path)


def mmlu_eval(config, llm, tokenizer, file_path, data_path):
    """
    Performance Evaluation on the MMLU Benchmark
    :param config: the configuration file
    :param llm: the vLLM engine
    :param tokenizer: the tokenizer
    :param file_path: save file path and file name
    :param data_path: testing dataset path
    """
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

        temp_ins = prompt_template(tokenizer=tokenizer, input=question)
        prompts.append(temp_ins)

        # Get the label answer
        temp_ans = item[5]
        answers.append(temp_ans)

    performance_eval(config=config, llm=llm, prompts=prompts, answers=answers, file_path=file_path)


def mcmle_eval(config, llm, tokenizer, file_path, data_path):
    """
    Performance Evaluation on the MedQA-MCMLE Benchmark
    :param config: the configuration file
    :param llm: the vLLM engine
    :param tokenizer: the tokenizer
    :param file_path: save file path and file name
    :param data_path: testing dataset path
    """
    prompts = []
    answers = []

    with open(data_path, "r+", encoding="utf8") as f:
        for idx, item in enumerate(jsonlines.Reader(f)):
            question = mcmle_format(item)
            temp_ins = prompt_template(tokenizer=tokenizer, input=question)
            prompts.append(temp_ins)

            # Get the label answer
            temp_ans = item["answer_idx"]
            answers.append(temp_ans)

    performance_eval(config=config, llm=llm, prompts=prompts, answers=answers, file_path=file_path)


def cmmlu_eval(config, llm, tokenizer, file_path, data_path):
    """
    Performance Evaluation on the CMMLU Benchmark
    :param config: the configuration file
    :param llm: the vLLM engine
    :param tokenizer: the tokenizer
    :param file_path: save file path and file name
    :param data_path: testing dataset path
    """
    prompts = []
    answers = []

    csv = pd.read_csv(data_path, header=0)
    for index, item in csv.iterrows():
        question = cmmlu_format(item)
        temp_ins = prompt_template(tokenizer=tokenizer, input=question)
        prompts.append(temp_ins)

        # Get the label answer
        temp_ans = item[6]
        answers.append(temp_ans)

    performance_eval(config=config, llm=llm, prompts=prompts, answers=answers, file_path=file_path)


def frenchmedmcqa_eval(config, llm, tokenizer, file_path, data_path):
    """
    Performance Evaluation on the FrenchMedMCQA Benchmark
    :param config: the configuration file
    :param llm: the vLLM engine
    :param tokenizer: the tokenizer
    :param file_path: save file path and file name
    :param data_path: testing dataset path
    """
    prompts = []
    answers = []

    with open(data_path, 'r') as f:
        contents = json.loads(f.read())
        for item in contents:
            # We only use the single-answer subset
            if item["nbr_correct_answers"] == 1:
                question = frenchmedmcqa_format(item)
                temp_ins = prompt_template(tokenizer=tokenizer, input=question)
                prompts.append(temp_ins)

                # Get the label answer
                temp_ans = item["correct_answers"][0]
                answers.append(temp_ans.upper())

    performance_eval(config=config, llm=llm, prompts=prompts, answers=answers, file_path=file_path)


def headqa_eval(config, llm, tokenizer, file_path, data_path):
    """
    Performance Evaluation on the HEAD-QA Benchmark
    :param config: the configuration file
    :param llm: the vLLM engine
    :param tokenizer: the tokenizer
    :param file_path: save file path and file name
    :param data_path: testing dataset path
    """
    prompts = []
    answers = []

    with open(data_path, 'r') as f:
        contents = json.loads(f.read())
        for exams in contents["exams"]:
            for item in contents["exams"][exams]["data"]:
                question = headqa_format(item)
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

    performance_eval(config=config, llm=llm, prompts=prompts, answers=answers, file_path=file_path)
