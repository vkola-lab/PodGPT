#!/usr/bin/env python
# coding=utf-8
#
# GNU Affero General Public License v3.0 License
#
# PodGPT: An Audio-augmented Large Language Model for Research and Education
# Copyright (C) 2024 Kolachalama Laboratory at Boston University
#
# GPTQModel ACKNOWLEDGEMENT
# Production ready LLM model compression/quantization toolkit with accelerated inference
# support for both cpu/gpu via HF, vLLM, and SGLang.
# https://github.com/ModelCloud/GPTQModel
#
# LICENSE OF THE GPTQModel
# Apache 2.0 License
# https://github.com/ModelCloud/GPTQModel?tab=Apache-2.0-1-ov-file

import os
import logging
import argparse

import torch
from transformers import AutoTokenizer
from datasets import load_dataset
from gptqmodel import GPTQModel as AutoGPTQForCausalLM
from gptqmodel import QuantizeConfig as BaseQuantizeConfig


def quantization(model_name, output_dir, bits, group_size, desc_act, damp, trust_remote_code=True, dtype='bfloat16'):
    """
    Quantizes a model using AutoGPTQ.
    :param model_name: The name of the model to quantize.
    :param output_dir: Directory to save the quantized model.
    :param bits: Bit-width for quantization (e.g., 4, 8).
    :param group_size: Group size for quantization.
    :param desc_act: Whether to use desc_act (True/False).
    :param damp: Dampening percentage for quantization.
    :param trust_remote_code: Trust remote code while loading models.
    :param dtype: Data type for processing (e.g., float16, float32, bfloat16).
    """
    # Create a quantization configuration
    quantize_config = BaseQuantizeConfig(
        bits=bits,
        group_size=group_size,
        desc_act=desc_act,
        damp_percent=damp,
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
    model = AutoGPTQForCausalLM.from_pretrained(
        model_name,
        quantize_config=quantize_config,
        low_cpu_mem_usage=True,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
        # device_map="auto"
    )

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        trust_remote_code=trust_remote_code
    )

    # Load the quantization dataset
    quantdataset = [
        tokenizer(example["text"])
        for example in load_dataset("shuyuej/MedPodGPT-Demo-Data", split="train").select(range(256))
    ]

    # Perform the quantization process
    model.quantize(quantdataset)

    # Save the quantized model
    model.save_quantized(output_dir)


def mian(args):
    # Load the logger
    logger = logging.getLogger()
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S"
    )

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
    logger.info(f"Starting {num_iters} quantization.")
    count = 1
    for iter in iterations:
        if not os.path.isfile(stop_file) and not abort:
            bits = iter['bits']
            group_size = iter['group_size']
            desc_act = iter['desc_act']
            damp = iter['damp']

            output_dir = args.output_dir
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
                        args.model_name,
                        output_dir,
                        bits,
                        group_size,
                        desc_act,
                        damp,
                        trust_remote_code=args.trust_remote_code,
                        dtype=args.dtype
                    )
                except KeyboardInterrupt:
                    # Handle user interrupt
                    logger.error("Aborted. Will delete {output_dir}")
                    os.rmdir(output_dir)
                    abort = True
                except Exception:
                    raise
            finally:
                count += 1
        else:
            logger.error("Aborting - told to stop!")
            break


if __name__ == "__main__":
    # Example Usage:
    # python quantization_GPTQModel.py "meta-llama/Llama-3.3-70B-Instruct" "./gptq_model" \
    # --bits 4 \
    # --group_size 128 \
    # --seqlen 2048 \
    # --damp 0.01 \
    # --desc_act 1 \
    # --dtype bfloat16
    parser = argparse.ArgumentParser(description='Quantize LLMs using the GPTQ Algorithm.')
    parser.add_argument('model_name', type=str, help='Repo name')
    parser.add_argument('output_dir', type=str, help='Output base folder')
    parser.add_argument('--trust_remote_code', action="store_true", help='Trust remote code')
    parser.add_argument('--bits', type=int, nargs='+', default=[4], help='Quantize bit(s)')
    parser.add_argument('--group_size', type=int, nargs='+', default=[32, 128, 1024, -1],
                        help='Quantize group size(s)')
    parser.add_argument('--damp', type=float, nargs='+', default=[0.01], help='Quantize damp_percent(s)')
    parser.add_argument('--desc_act', type=int, nargs='+', default=[0, 1],
                        help='Quantize desc_act(s) - 1 = True, 0 = False')
    parser.add_argument('--dtype', type=str, choices=['float16', 'float32', 'bfloat16'],
                        help='Quantize desc_act(s) - 1 = True, 0 = False')
    parser.add_argument('--seqlen', type=int, default=2048, help='Model sequence length')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Quantize batch size for processing dataset samples')
    parser.add_argument('--stop_file', type=str,
                        help='Filename to look for to stop inference, specific to this instance')
    args = parser.parse_args()

    stop_file = args.stop_file or ""  # Set stop file if provided

    # Quantize the model in the `pretrained_model_dir`
    mian(args=args)
