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

import os
import time
import json

import ray
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from vllm.distributed.parallel_state import (
    destroy_model_parallel,
    destroy_distributed_environment,
)

from utils.utils import *
from utils.answer_utils import *


def performance_eval(config, mode, prompts, answers, documents, file_path):
    """
    Generate responses by vLLM and conduct performance evaluation
    :param config: the configuration file
    :param mode: one of "small", "large", or "chatgpt"
        "small": Evaluate the smaller models
        "large": Evaluate the larger models (with the LoRA Adapter)
    :param prompts: prompted questions
    :param answers: ground truth answers
    :param documents: the related documents to the query
    :param file_path: save file path and file name
    """
    # Load some configurations
    model_name = config.get("model_name")
    max_new_tokens = config.get("max_new_tokens")
    save_dir = config.get("save_dir")
    eval_pretrain = config.get("eval_pretrain")
    num_gpus_vllm = config.get("num_gpus_vllm")
    gpu_utilization_vllm = config.get("gpu_utilization_vllm")

    # Set the main file path
    main_file_path = file_path

    # List to store IDs
    checkpoint_ids = []
    # Walk through the folder
    for root, dirs, files in os.walk(save_dir):
        for dir_name in dirs:
            if dir_name.startswith("checkpoint-"):
                # Extract the ID part after "checkpoint-"
                checkpoint_id = dir_name.split("checkpoint-")[-1]
                checkpoint_ids.append(int(checkpoint_id))

    # Remove duplicates and sort the IDs
    checkpoint_ids = sorted(set(checkpoint_ids))

    # Evaluate the original baseline model
    if eval_pretrain:
        start_t = time.time()
        file_path = main_file_path + "baseline_model.json"
        responses = []
        # Evaluating the larger models
        if mode == "large":
            max_model_len_vllm = config.get("max_model_len_vllm")
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
                    stop_token_ids=[128001, 128004, 128008, 128009],
                )
                # Initialize vLLM engine
                llm = LLM(
                    model=save_dir,
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
            else:
                stop_tokens = stop_token_list()
                # https://github.com/vllm-project/vllm/blob/main/vllm/sampling_params.py#L38-L66
                sampling_params = SamplingParams(
                    temperature=0,
                    top_p=1,
                    max_tokens=max_new_tokens,
                    stop=stop_tokens
                )
                # Initialize vLLM engine
                llm = LLM(
                    model=save_dir,
                    tokenizer=model_name,
                    dtype='bfloat16',
                    # Acknowledgement: Benjamin Kitor
                    # https://github.com/vllm-project/vllm/issues/2794
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

        # Evaluating the smaller models
        elif mode == "small":
            stop_tokens = stop_token_list()
            # https://github.com/vllm-project/vllm/blob/main/vllm/sampling_params.py#L38-L66
            sampling_params = SamplingParams(
                temperature=0,
                top_p=1,
                # repetition_penalty=2.0,  # Only for the `Mistral 7B` model on the `Hindi Benchmarks`!
                max_tokens=max_new_tokens,
                stop=stop_tokens
            )

            llm = LLM(
                model=save_dir,
                tokenizer=model_name,
                dtype='bfloat16',
                # Acknowledgement: Benjamin Kitor
                # https://github.com/vllm-project/vllm/issues/2794
                # https://github.com/vllm-project/vllm/issues/1908
                distributed_executor_backend="mp",
                tensor_parallel_size=num_gpus_vllm,
                gpu_memory_utilization=gpu_utilization_vllm,
                disable_custom_all_reduce=True,
                enable_lora=False
            )

            # Get the model's responses
            completions = llm.generate(prompts, sampling_params)
            for i, output in enumerate(completions):
                temp_gen = output.outputs[0].text
                responses.append(temp_gen)
            print('Successfully finished generating', len(prompts), 'samples!')

        acc = []
        all_out = []  # Save all the model's responses
        invalid_out = []  # Only save the invalid responses
        for idx, (prompt, document, response, answer) in enumerate(zip(prompts, documents, responses, answers)):
            # Special Notice:
            # option_range="a-dA-D" means options A, B, C, D
            # option_range="a-eA-E" means options A, B, C, D, E
            # English Benchmarks
            if any(keyword in file_path.lower()
                   for keyword in ["english_mmlu", "english_medicine_medmcqa", "english_medicine_medqa"]):
                option_range = "a-dA-D"  # 4 Options
            elif "english_medicine_medexpqa" in file_path.lower():
                option_range = "a-eA-E"  # 5 Options
            else:
                option_range = None

            if option_range:
                prediction = extract_answer(completion=response, option_range=option_range)
                if prediction is None:
                    response = response_with_option(prompt=prompt, response=response)
                    prediction = extract_answer(completion=response, option_range=option_range)
            else:
                # We separately write codes to evaluate our models' performance
                # on the PubMedQA benchmark (yes/no/maybe)
                prediction = extract_answer_for_pubmedqa(completion=response)

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

    # Evaluate all the checkpoints of the models
    for id in checkpoint_ids:
        start_t = time.time()
        print("The current checkpoint is", id)
        responses = []
        # Evaluating the larger models
        if mode == "large":
            lora_path = save_dir + "/checkpoint-" + str(id)
            file_path = main_file_path + "checkpoint-" + str(id) + ".json"
            max_model_len_vllm = config.get("max_model_len_vllm")

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
                    stop_token_ids=[128001, 128004, 128008, 128009],
                )
                llm = LLM(
                    model=save_dir,
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
            else:
                stop_tokens = stop_token_list()
                # https://github.com/vllm-project/vllm/blob/main/vllm/sampling_params.py#L38-L66
                sampling_params = SamplingParams(
                    temperature=0,
                    top_p=1,
                    max_tokens=max_new_tokens,
                    stop=stop_tokens
                )
                # Initialize vLLM engine
                llm = LLM(
                    model=save_dir,
                    tokenizer=model_name,
                    dtype='bfloat16',
                    # Acknowledgement: Benjamin Kitor
                    # https://github.com/vllm-project/vllm/issues/2794
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
                lora_request=LoRARequest("adapter", 1, lora_path)
            )
            for i, output in enumerate(completions):
                temp_gen = output.outputs[0].text
                responses.append(temp_gen)
            print('Successfully finished generating', len(prompts), 'samples!')

        # Evaluating the smaller models
        elif mode == "small":
            model_path = save_dir + "/checkpoint-" + str(id)
            file_path = main_file_path + "checkpoint-" + str(id) + ".json"
            stop_tokens = stop_token_list()
            # https://github.com/vllm-project/vllm/blob/main/vllm/sampling_params.py#L38-L66
            sampling_params = SamplingParams(
                temperature=0,
                top_p=1,
                max_tokens=max_new_tokens,
                stop=stop_tokens
            )

            llm = LLM(
                model=model_path,
                tokenizer=model_name,
                dtype='bfloat16',
                # Acknowledgement: Benjamin Kitor
                # https://github.com/vllm-project/vllm/issues/2794
                # https://github.com/vllm-project/vllm/issues/1908
                distributed_executor_backend="mp",
                tensor_parallel_size=num_gpus_vllm,
                gpu_memory_utilization=gpu_utilization_vllm,
                disable_custom_all_reduce=True,
                enable_lora=False
            )

            # Get the model's responses
            completions = llm.generate(prompts, sampling_params)
            for i, output in enumerate(completions):
                temp_gen = output.outputs[0].text
                responses.append(temp_gen)
            print('Successfully finished generating', len(prompts), 'samples!')

        acc = []
        all_out = []  # Save all the model's responses
        invalid_out = []  # Only save the invalid responses
        for idx, (prompt, document, response, answer) in enumerate(zip(prompts, documents, responses, answers)):
            # Special Notice:
            # option_range="a-dA-D" means options A, B, C, D
            # option_range="a-eA-E" means options A, B, C, D, E
            # English Benchmarks
            if any(keyword in file_path.lower()
                   for keyword in ["english_mmlu", "english_medicine_medmcqa", "english_medicine_medqa"]):
                option_range = "a-dA-D"  # 4 Options
            elif "english_medicine_medexpqa" in file_path.lower():
                option_range = "a-eA-E"  # 5 Options
            else:
                option_range = None

            if option_range:
                prediction = extract_answer(completion=response, option_range=option_range)
                if prediction is None:
                    response = response_with_option(prompt=prompt, response=response)
                    prediction = extract_answer(completion=response, option_range=option_range)
            else:
                # We separately write codes to evaluate our models' performance
                # on the PubMedQA benchmark (yes/no/maybe)
                prediction = extract_answer_for_pubmedqa(completion=response)

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
