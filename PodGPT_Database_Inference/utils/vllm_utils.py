# coding=utf-8
#
# GNU Affero General Public License v3.0 License
#
# PodGPT: An Audio-augmented Large Language Model for Research and Education
# Copyright (C) 2024 Kolachalama Laboratory at Boston University
#
# LLMs INFERENCE ACKNOWLEDGEMENT
# vLLM - Easy, fast, and cheap LLM serving for everyone
# https://github.com/vllm-project/vllm
#
# LICENSE OF THE INFERENCE ENGINE
# Apache 2.0 License
# https://github.com/vllm-project/vllm/blob/main/LICENSE

import gc
import time
import json
import contextlib

import openai
import torch
from huggingface_hub import snapshot_download
import ray
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from vllm.distributed.parallel_state import (
    destroy_model_parallel,
    destroy_distributed_environment,
)

from utils.answer_utils import *


def performance_eval(config, mode, prompts, answers, documents, file_path):
    """
    Generate responses by vLLM and conduct performance evaluation
    :param config: the configuration file
    :param mode: one of "podgpt" or "chatgpt"
        "podgpt": Evaluate PodGPT
        "chatgpt": Evaluate the OpenAI ChatGPT
    :param prompts: prompted questions
    :param answers: ground truth answers
    :param documents: the related documents to the query
    :param file_path: save file path and file name
    """
    # Load some configurations
    model_name = config.get("model_name")
    max_new_tokens = config.get("max_new_tokens")

    # Set responses as a list
    responses = []

    # Set the main file path
    main_file_path = file_path
    if mode == "podgpt":
        eval_pretrain = config.get("eval_pretrain")
        num_gpus_vllm = config.get("num_gpus_vllm")
        gpu_utilization_vllm = config.get("gpu_utilization_vllm")
        max_model_len_vllm = config.get("max_model_len_vllm")

        # Evaluate the original baseline model
        if eval_pretrain:
            start_t = time.time()
            file_path = main_file_path + "baseline.json"

            sampling_params = SamplingParams(
                temperature=0,
                top_p=1,
                max_tokens=max_new_tokens,
                # https://huggingface.co/astronomer/Llama-3-8B-Instruct-GPTQ-8-Bit
                # For loading this model onto vLLM, make sure all requests have
                # "stop_token_ids":[128001, 128009] to temporarily address the non-stop generation issue.
                # vLLM does not yet respect generation_config.json.
                # vLLM team is working on a fix for this https://github.com/vllm-project/vllm/issues/4180
                stop_token_ids=[128001, 128008, 128009],
            )
            # Initialize vLLM engine
            llm = LLM(
                model=model_name,
                tokenizer=model_name,
                dtype='float16',
                quantization="GPTQ",
                # Acknowledgement: Benjamin Kitor
                # https://github.com/vllm-project/vllm/issues/2794
                # Reference:
                # https://github.com/vllm-project/vllm/issues/1908
                distributed_executor_backend="mp",
                tensor_parallel_size=num_gpus_vllm,
                gpu_memory_utilization=gpu_utilization_vllm,
                # Note: We add this only to save the GPU Memories!
                max_model_len=max_model_len_vllm,
                disable_custom_all_reduce=True,
                enable_lora=False,
            )

            # Get the model's responses
            completions = llm.generate(prompts, sampling_params)
            for i, output in enumerate(completions):
                temp_gen = output.outputs[0].text
                responses.append(temp_gen)
            print('Successfully finished generating', len(prompts), 'samples!')

        # Evaluate our PodGPT
        else:
            lora_path = config.get("lora_path")
            podgpt_lora = snapshot_download(repo_id=lora_path)
            file_path = main_file_path + "PodGPT" + ".json"

            sampling_params = SamplingParams(
                temperature=0,
                top_p=1,
                max_tokens=max_new_tokens,
                # https://huggingface.co/astronomer/Llama-3-8B-Instruct-GPTQ-8-Bit
                # For loading this model onto vLLM, make sure all requests have
                # "stop_token_ids":[128001, 128009] to temporarily address the non-stop generation issue.
                # vLLM does not yet respect generation_config.json.
                # vLLM team is working on a fix for this https://github.com/vllm-project/vllm/issues/4180
                stop_token_ids=[128001, 128008, 128009],
            )
            llm = LLM(
                model=model_name,
                tokenizer=model_name,
                dtype='float16',
                quantization="GPTQ",
                # Acknowledgement: Benjamin Kitor
                # https://github.com/vllm-project/vllm/issues/2794
                # Reference:
                # https://github.com/vllm-project/vllm/issues/1908
                distributed_executor_backend="mp",
                tensor_parallel_size=num_gpus_vllm,
                gpu_memory_utilization=gpu_utilization_vllm,
                # Note: We add this only to save the GPU Memories!
                max_model_len=max_model_len_vllm,
                disable_custom_all_reduce=True,
                enable_lora=True,
            )

            # Get the model's responses
            completions = llm.generate(
                prompts,
                sampling_params,
                lora_request=LoRARequest("adapter", 1, podgpt_lora)
            )
            for i, output in enumerate(completions):
                temp_gen = output.outputs[0].text
                responses.append(temp_gen)
            print('Successfully finished generating', len(prompts), 'samples!')

    # Evaluating the OpenAI ChatGPT
    elif mode == "chatgpt":
        model_name = config.get("model_name")
        file_path = main_file_path + model_name + ".json"
        openai_api_key = config.get("openai_api_key")
        openai.api_key = openai_api_key

        # Please note that we are prompting a single question at a time, instead of a batch of questions.
        # If you wanna use batch processing, please refer to
        # https://github.com/meta-math/MetaMath/blob/main/code_for_generating_data/code/utils/parallel_utils.py
        for prompt in prompts:
            attempts = 0
            max_attempts = 100
            result = None
            while attempts < max_attempts:
                try:
                    message = openai.ChatCompletion.create(
                        model=model_name,
                        messages=[
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0,
                        max_tokens=max_new_tokens,
                    )
                    result = message['choices'][0]['message']['content']
                    # print("Received response:", result)
                    # print("Operation successful.")
                    break  # Exit the loop on success

                except Exception as e:
                    print(f"Attempt {attempts + 1} failed due to error: {e}")
                    result = None
                    print("Retrying in 10 seconds...")
                    time.sleep(10)
                    attempts += 1

            if result is None:
                responses.append("No response")
            else:
                responses.append(result)

        print('Successfully finished generating', len(responses), 'samples!')

    else:
        responses = None

    acc = []
    all_out = []  # Save all the model's responses
    invalid_out = []  # Only save the invalid responses
    for idx, (prompt, document, response, answer) in enumerate(zip(prompts, documents, responses, answers)):
        # Special Notice:
        # option_range="a-dA-D" means options A, B, C, D
        # option_range="a-eA-E" means options A, B, C, D, E
        option_range = "a-dA-D"  # 4 Options
        prediction = extract_answer(completion=response, option_range=option_range)
        if prediction is None:
            response = response_with_option(prompt=prompt, response=response)
            prediction = extract_answer(completion=response, option_range=option_range)

        if prediction is not None:
            acc.append(prediction == answer)
            temp = {
                'prompt': prompt,
                'retrieved document': document,
                'response': response,
                'extracted answer': prediction,
                'answer': answer
            }
            all_out.append(temp)
        else:
            acc.append(False)
            temp = {
                'prompt': prompt,
                'retrieved document': document,
                'response': response,
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
    with open(file_path.replace('.json', '_invalid_responses.json'), 'w', encoding='utf-8') as file:
        json.dump(invalid_out, file, ensure_ascii=False, indent=4)
    print('Successfully save the invalid output.')

    # Save all the responses in a JSON file
    with open(file_path.replace('.json', '_all_responses.json'), 'w', encoding='utf-8') as file:
        json.dump(all_out, file, ensure_ascii=False, indent=4)
    print('Successfully save all the output.')

    if mode == "podgpt":
        # Delete the llm object and free the memory
        # https://github.com/vllm-project/vllm/issues/1908#issuecomment-2461174904
        destroy_model_parallel()
        destroy_distributed_environment()
        del llm.llm_engine.model_executor
        del llm
        with contextlib.suppress(AssertionError):
            torch.distributed.destroy_process_group()
        gc.collect()
        torch.cuda.empty_cache()
        ray.shutdown()
        print("Successfully delete the llm pipeline and free the GPU memory.\n\n\n\n")
