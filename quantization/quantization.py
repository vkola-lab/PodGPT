#!/usr/bin/env python
# coding=utf-8
#
# GNU Affero General Public License v3.0 License
#
# PodGPT: An Audio-augmented Large Language Model for Research and Education
# Copyright (C) 2024 Kolachalama Laboratory at Boston University
#
# AutoGPTQ ACKNOWLEDGEMENT
# AutoGPTQ - An easy-to-use LLM quantization package with user-friendly APIs,
# based on GPTQ algorithm (weight-only quantization).
# https://github.com/AutoGPTQ/AutoGPTQ
#
# LICENSE OF THE AUTOGPTQ
# MIT License
# https://github.com/AutoGPTQ/AutoGPTQ/blob/main/LICENSE

####################################################################################

# SPECIAL NOTICE
# THE CODES ARE PROVIDED AND OPEN-SOURCED BY TOM JOBBINS
# https://github.com/AutoGPTQ/AutoGPTQ/issues/179#issuecomment-1611257490
# https://www.reddit.com/r/LocalLLaMA/comments/12rtg82/what_is_group_size_128_and_why_do_30b_models_give/
# THE CODES ARE MODIFIED BY THE TEAM FROM THE KOLACHALAMA LAB AT BOSTON UNIVERSITY

####################################################################################

import time
import os
import logging
import argparse
import random

import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig


def dataset_loader(nsamples, seed, seqlen, tokenizer):
    """
    Load and tokenize podcast data: `shuyuej/MedPodGPT-Demo-Data`.
    :param nsamples: Number of samples to extract.
    :param seed: Random seed for reproducibility.
    :param seqlen: Sequence length for each sample.
    :param tokenizer: Tokenizer instance to tokenize text.
    :return: List of tokenized input datasets.
    """
    logger = logging.getLogger(__name__)

    # Load the dataset from Hugging Face repository
    data = load_dataset("shuyuej/MedPodGPT-Demo-Data", split="train")
    datalist = [' \n' if s == '' else s for s in data['text']]  # Replace empty strings with newlines

    text = ''.join(datalist)  # Concatenate the dataset into a single text
    logger.info("Tokenising our medical dataset")
    trainenc = tokenizer(text, return_tensors='pt')  # Tokenize the entire text

    # Set random seeds for reproducibility
    random.seed(seed)
    np.random.seed(0)
    torch.random.manual_seed(0)

    # Create a list of training samples by slicing the tokenized text
    traindataset = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)  # Random start index
        j = i + seqlen  # End index for the sequence
        inp = trainenc.input_ids[:, i:j]  # Extract input IDs
        attention_mask = torch.ones_like(inp)  # Create an attention mask
        traindataset.append({'input_ids': inp, 'attention_mask': attention_mask})

    return traindataset


def quantization(model_dir, output_dir, quantdataset, bits, group_size, desc_act, damp,
                 batch_size=1, use_triton=False, trust_remote_code=False, dtype='float16'):
    """
    Quantizes a model using AutoGPTQ.
    :param model_dir: Path to the pretrained model directory.
    :param output_dir: Directory to save the quantized model.
    :param quantdataset: Dataset for calibration during quantization.
    :param bits: Bit-width for quantization (e.g., 4, 8).
    :param group_size: Group size for quantization.
    :param desc_act: Whether to use desc_act (True/False).
    :param damp: Dampening percentage for quantization.
    :param batch_size: Batch size for processing during quantization.
    :param use_triton: Use Triton backend for acceleration.
    :param trust_remote_code: Trust remote code while loading models.
    :param dtype: Data type for processing (e.g., float16, float32, bfloat16).
    """
    # Create a quantization configuration
    quantize_config = BaseQuantizeConfig(
        bits=bits,
        group_size=group_size,
        desc_act=desc_act,
        damp_percent=damp
    )

    # Map string dtype to PyTorch data types
    if dtype == 'float16':
        torch_dtype = torch.float16
    elif dtype == 'float32':
        torch_dtype = torch.float32
    elif dtype == 'bfloat16':
        torch_dtype = torch.bfloat16
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    # Load the model with specified quantization settings
    logger.info(f"Loading model from {model_dir} with trust_remote_code={trust_remote_code} and dtype={torch_dtype}")
    model = AutoGPTQForCausalLM.from_pretrained(
        model_dir,
        quantize_config=quantize_config,
        low_cpu_mem_usage=True,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code
    )

    # Perform the quantization process
    logger.info(f"Starting quantization to {output_dir} with use_triton={use_triton}")
    start_time = time.time()
    model.quantize(quantdataset, use_triton=use_triton, batch_size=batch_size)
    logger.info(f"Time to quantize model at {output_dir} with use_triton={use_triton}: {time.time() - start_time:.2f}")

    # Save the quantized model
    logger.info(f"Saving quantized model to {output_dir}")
    model.save_quantized(output_dir, use_safetensors=True)
    logger.info("Done.")


def mian(args):
    # Load the logger
    logger = logging.getLogger()
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_dir,
        use_fast=True,
        trust_remote_code=args.trust_remote_code
    )

    # Prepare the dataset for quantization
    quantdataset = dataset_loader(nsamples=128, seed=0, seqlen=args.seqlen, tokenizer=tokenizer)

    abort = False
    iterations = []
    for bits in args.bits:
        for group_size in args.group_size:
            for desc_act in args.desc_act:
                for damp in args.damp:
                    desc_act = desc_act == 1 and True or False  # Convert to boolean
                    iterations.append({"bits": bits, "group_size": group_size, "desc_act": desc_act, "damp": damp})

    # Log the number of quantization tasks
    num_iters = len(iterations)
    logger.info(f"Starting {num_iters} quantizations.")
    count = 1
    for iter in iterations:
        if not os.path.isfile(stop_file) and not abort:
            bits = iter['bits']
            group_size = iter['group_size']
            desc_act = iter['desc_act']
            damp = iter['damp']

            output_dir = args.output_dir_base
            try:
                os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists

                # Log details about the current task
                logger.info(
                    f"[{count} / {num_iters}] Quantizing: bits = {bits} - group_size = {group_size} - desc_act = "
                    f"{desc_act} - damp_percent = {damp} to {output_dir}"
                )
                try:
                    # Call the quantization function
                    quantization(
                        args.pretrained_model_dir,
                        output_dir,
                        quantdataset,
                        bits,
                        group_size,
                        desc_act,
                        damp,
                        args.batch_size,
                        args.use_triton,
                        trust_remote_code=args.trust_remote_code,
                        dtype=args.dtype
                    )
                except KeyboardInterrupt:
                    # Handle user interrupt
                    logger.error(f"Aborted. Will delete {output_dir}")
                    os.rmdir(output_dir)
                    abort = True
                except:
                    raise
            finally:
                count += 1
        else:
            logger.error(f"Aborting - told to stop!")
            break


if __name__ == "__main__":
    # Example Usage:
    # python quantization.py "./save_folder" "./gptq_model" \
    # --bits 4 \
    # --group_size 128 \
    # --damp 0.01 \
    # --desc_act 1 \
    # --dtype bfloat16
    parser = argparse.ArgumentParser(description='Quantize LLMs using the GPTQ Algorithm.')
    parser.add_argument('pretrained_model_dir', type=str, help='Repo name')
    parser.add_argument('output_dir_base', type=str, help='Output base folder')
    parser.add_argument('--trust_remote_code', action="store_true", help='Trust remote code')
    parser.add_argument('--use_triton', action="store_true", help='Use Triton for quantization')
    parser.add_argument('--bits', type=int, nargs='+', default=[4], help='Quantize bit(s)')
    parser.add_argument('--group_size', type=int, nargs='+', default=[32, 128, 1024, -1],
                        help='Quantize group size(s)')
    parser.add_argument('--damp', type=float, nargs='+', default=[0.01], help='Quantize damp_percent(s)')
    parser.add_argument('--desc_act', type=int, nargs='+', default=[0, 1],
                        help='Quantize desc_act(s) - 1 = True, 0 = False')
    parser.add_argument('--dtype', type=str, choices=['float16', 'float32', 'bfloat16'],
                        help='Quantize desc_act(s) - 1 = True, 0 = False')
    parser.add_argument('--seqlen', type=int, default=8192, help='Model sequence length')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Quantize batch size for processing dataset samples')
    parser.add_argument('--stop_file', type=str,
                        help='Filename to look for to stop inference, specific to this instance')
    args = parser.parse_args()

    stop_file = args.stop_file or ""  # Set stop file if provided

    # Quantize the model in the `pretrained_model_dir`
    mian(args=args)
