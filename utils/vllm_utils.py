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

import time
import json

import ray
import openai
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from vllm.distributed.parallel_state import (
    destroy_model_parallel,
    destroy_distributed_environment,
)

from utils.utils import *
from utils.answer_utils import *


def performance_eval(config, mode, prompts, answers, file_path):
    """
    Generate responses by vLLM and conduct performance evaluation
    :param config: the configuration file
    :param mode: one of "small", "large", or "chatgpt"
        "small": Evaluate the smaller models
        "large": Evaluate the larger models (with the LoRA Adapter)
        "chatgpt": Evaluate the OpenAI ChatGPT
    :param prompts: prompted questions
    :param answers: ground truth answers
    :param file_path: save file path and file name
    """
    # Load some configurations
    model_name = config.get("model_name")
    max_new_tokens = config.get("max_new_tokens")
    save_dir = config.get("save_dir")
    eval_pretrain = config.get("eval_pretrain")

    # The start time for evaluation
    start_t = time.time()

    responses = []
    # Evaluating the larger models
    if mode == "large":
        num_gpus_vllm = config.get("num_gpus_vllm")
        gpu_utilization_vllm = config.get("gpu_utilization_vllm")
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
            if eval_pretrain:
                # Initialize vLLM engine
                llm = LLM(
                    model=save_dir,
                    tokenizer=model_name,
                    # While using the GPTQ quantization, the current vLLM only supports float16, as of Dec. 14th, 2024
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
                # For the evaluation of the LoRA model
                completions = llm.generate(
                    prompts,
                    sampling_params,
                )
            else:
                # Get the LoRA adapter path
                lora_path = config.get("lora_path")

                # Initialize vLLM engine
                llm = LLM(
                    model=save_dir,
                    tokenizer=model_name,
                    # While using the GPTQ quantization, the current vLLM only supports float16, as of Dec. 14th, 2024
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
                # For the evaluation of the LoRA model
                completions = llm.generate(
                    prompts,
                    sampling_params,
                    lora_request=LoRARequest("adapter", 1, lora_path)
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
                # Initialize vLLM engine
                llm = LLM(
                    model=save_dir,
                    tokenizer=model_name,
                    dtype='bfloat16',
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
                # For the evaluation of the LoRA model
                completions = llm.generate(
                    prompts,
                    sampling_params,
                )
            else:
                # Get the LoRA adapter path
                lora_path = config.get("lora_path")

                # Initialize vLLM engine
                llm = LLM(
                    model=save_dir,
                    tokenizer=model_name,
                    dtype='bfloat16',
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

    # Evaluating the smaller models
    # Please take a look at the above quantization codes if you are using a quantized model.
    elif mode == "small":
        num_gpus_vllm = config.get("num_gpus_vllm")
        gpu_utilization_vllm = config.get("gpu_utilization_vllm")

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
            # Link: https://github.com/vllm-project/vllm/issues/2794
            # Reference:
            # https://github.com/vllm-project/vllm/issues/1908
            distributed_executor_backend="mp",
            tensor_parallel_size=num_gpus_vllm,
            gpu_memory_utilization=gpu_utilization_vllm,
            disable_custom_all_reduce=True,
            enable_lora=False
        )
        completions = llm.generate(
            prompts,
            sampling_params,
        )
        for i, output in enumerate(completions):
            temp_gen = output.outputs[0].text
            responses.append(temp_gen)
        print('Successfully finished generating', len(prompts), 'samples!')

    # Evaluating the ChatGPT
    elif mode == "chatgpt":
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
        if "english_medicine_pubmedqa" in file_path.lower():
            # We separately write codes to evaluate our models' performance
            # on the PubMedQA benchmark (yes/no/maybe)
            prediction = extract_answer_for_pubmedqa(completion=response)
        elif ("english_mmlu" in file_path.lower()
              or "english_medicine_medmcqa" in file_path.lower()
              or "english_medicine_medqa" in file_path.lower()):
            prediction = extract_answer(completion=response, option_range="a-dA-D")  # 4 Options
            if prediction is None:
                response = response_with_option(prompt=question, response=response)
                prediction = extract_answer(completion=response, option_range="a-dA-D")  # 4 Options
        elif "english_medicine_medexpqa" in file_path.lower():
            prediction = extract_answer(completion=response, option_range="a-eA-E")  # 5 Options
            if prediction is None:
                response = response_with_option(prompt=question, response=response)
                prediction = extract_answer(completion=response, option_range="a-eA-E")  # 5 Options
        elif "english_medicine_usmle" in file_path.lower():
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
        elif "french_medicine_medexpqa" in file_path.lower() or "french_medicine_medmcqa" in file_path.lower():
            prediction = extract_answer_for_french(completion=response, option_range="a-eA-E")
            if prediction is None:
                response = response_with_option(prompt=question, response=response)
                prediction = extract_answer_for_french(completion=response, option_range="a-eA-E")

        # Spanish Benchmarks - 8 Benchmarks
        elif "spanish_mmlu" in file_path.lower() or "spanish_medicine_headqa" in file_path.lower():
            prediction = extract_answer_for_spanish(completion=response, option_range="a-dA-D")
            if prediction is None:
                response = response_with_option(prompt=question, response=response)
                prediction = extract_answer_for_spanish(completion=response, option_range="a-dA-D")
        elif "spanish_medicine_medexpqa" in file_path.lower():
            prediction = extract_answer_for_spanish(completion=response, option_range="a-eA-E")
            if prediction is None:
                response = response_with_option(prompt=question, response=response)
                prediction = extract_answer_for_spanish(completion=response, option_range="a-eA-E")

        # Hindi Benchmarks - 6 Benchmarks
        elif "hindi_mmlu" in file_path.lower():
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
    with open(file_path.replace('.json', '_invalid_responses.json'), 'w', encoding='utf-8') as file:
        json.dump(invalid_out, file, ensure_ascii=False, indent=4)
    print('Successfully save the invalid output.')

    # Save all the responses in a JSON file
    with open(file_path.replace('.json', '_all_responses.json'), 'w', encoding='utf-8') as file:
        json.dump(all_out, file, ensure_ascii=False, indent=4)
    print('Successfully save all the output.')

    if mode != "chatgpt":
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
