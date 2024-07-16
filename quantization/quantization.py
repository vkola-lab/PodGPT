#!/usr/bin/env python
# coding=utf-8
#
# GNU Affero General Public License v3.0 License
#
# MedPodGPT: A multilingual audio-augmented large language model for medical research and education
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


def get_medical_data(nsamples, seed, seqlen, tokenizer):
    """
    Load and tokenize the Multilingual podcasts: `shuyuej/MedPodGPT-Demo-Data`.
    :param nsamples: number of samples
    :param seed: random seed
    :param seqlen: sequence length
    :param tokenizer: tokenizer
    """
    logger = logging.getLogger(__name__)

    # Use this dataset in
    # https://github.com/AutoGPTQ/AutoGPTQ/issues/179#issuecomment-1611257490
    data = load_dataset(
        "shuyuej/MedPodGPT-Demo-Data",
        split="train",
    )
    datalist = [' \n' if s == '' else s for s in data['text']]

    text = ''.join(datalist)
    logger.info("Tokenising our medical dataset")
    trainenc = tokenizer(text, return_tensors='pt')

    random.seed(seed)
    np.random.seed(0)
    torch.random.manual_seed(0)

    traindataset = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        attention_mask = torch.ones_like(inp)
        traindataset.append({'input_ids': inp, 'attention_mask': attention_mask})

    return traindataset


def get_wikitext2(nsamples, seed, seqlen, tokenizer):
    """
    Load and tokenize the Wikitext-2 dataset.
    Reference:
    https://paperswithcode.com/dataset/wikitext-2
    https://github.com/huggingface/datasets/issues/4400#issuecomment-1288466795
    :param nsamples: number of samples
    :param seed: random seed
    :param seqlen: sequence length
    :param tokenizer: tokenizer
    """
    logger = logging.getLogger(__name__)

    # Use this dataset in
    # https://github.com/AutoGPTQ/AutoGPTQ/issues/179#issuecomment-1611257490
    wikidata = load_dataset(
        'wikitext',
        'wikitext-2-raw-v1',
        split='test'
    )
    # wikidata = load_dataset(path="wikitext", name="wikitext-103-v1", split="train")
    wikilist = [' \n' if s == '' else s for s in wikidata['text']]

    text = ''.join(wikilist)
    logger.info("Tokenising wikitext2")
    trainenc = tokenizer(text, return_tensors='pt')

    random.seed(seed)
    np.random.seed(0)
    torch.random.manual_seed(0)

    traindataset = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        attention_mask = torch.ones_like(inp)
        traindataset.append({'input_ids': inp, 'attention_mask': attention_mask})

    return traindataset


def get_c4(nsamples, seed, seqlen, tokenizer):
    """
    Load and tokenize the C4 dataset.
    :param nsamples: number of samples
    :param seed: random seed
    :param seqlen: sequence length
    :param tokenizer: tokenizer
    """
    traindata = load_dataset(
        'allenai/c4',
        data_files={'train': 'en/c4-train.00000-of-01024.json.gz'},
        split='train',
        use_auth_token=False
    )

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        attention_mask = torch.ones_like(inp)
        trainloader.append({'input_ids': inp, 'attention_mask': attention_mask})

    return trainloader


def quantize(model_dir, output_dir, traindataset, bits, group_size, desc_act, damp,
             batch_size=1, use_triton=False, trust_remote_code=False, dtype='float16'):
    """
    Reference:
    https://github.com/AutoGPTQ/AutoGPTQ/issues/179#issuecomment-1611257490
    https://github.com/AutoGPTQ/AutoGPTQ?tab=readme-ov-file#supported-models
    https://huggingface.co/blog/gptq-integration
    https://x.com/aniketmaurya/status/1753106224772731384
    """
    quantize_config = BaseQuantizeConfig(
        bits=bits,
        group_size=group_size,
        desc_act=desc_act,
        damp_percent=damp
    )

    if dtype == 'float16':
        torch_dtype = torch.float16
    elif dtype == 'float32':
        torch_dtype = torch.float32
    elif dtype == 'bfloat16':
        torch_dtype = torch.bfloat16
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    logger.info(f"Loading model from {model_dir} with trust_remote_code={trust_remote_code} and dtype={torch_dtype}")
    model = AutoGPTQForCausalLM.from_pretrained(
        model_dir,
        quantize_config=quantize_config,
        low_cpu_mem_usage=True,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code
    )

    logger.info(f"Starting quantization to {output_dir} with use_triton={use_triton}")
    start_time = time.time()
    model.quantize(traindataset, use_triton=use_triton, batch_size=batch_size)
    logger.info(f"Time to quantize model at {output_dir} with use_triton={use_triton}: {time.time() - start_time:.2f}")
    logger.info(f"Saving quantized model to {output_dir}")
    model.save_quantized(output_dir, use_safetensors=True)
    logger.info("Done.")


if __name__ == "__main__":
    # Example Usage:
    # python quantization.py "./save_folder" "./gptq_model" "medical" \
    # --bits 4 \
    # --group_size 128 \
    # --desc_act 1 \
    # --dtype float16
    logger = logging.getLogger()
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    parser = argparse.ArgumentParser(description='quantise')
    parser.add_argument('pretrained_model_dir', type=str, help='Repo name')
    parser.add_argument('output_dir_base', type=str, help='Output base folder')
    parser.add_argument('dataset', type=str, help='Output base folder')
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

    stop_file = args.stop_file or ""

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_dir,
        use_fast=True,
        trust_remote_code=args.trust_remote_code
    )

    # Select the dataset for GPTQ quantization
    if args.dataset == 'wikitext':
        traindataset = get_wikitext2(128, 0, args.seqlen, tokenizer)
    elif args.dataset == 'c4':
        traindataset = get_c4(128, 0, args.seqlen, tokenizer)
    elif args.dataset == 'medical':
        traindataset = get_medical_data(96, 0, args.seqlen, tokenizer)
    else:
        logger.error(f"Unsupported dataset: {args.dataset}")
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    abort = False
    iterations = []
    for bits in args.bits:
        for group_size in args.group_size:
            for desc_act in args.desc_act:
                for damp in args.damp:
                    desc_act = desc_act == 1 and True or False
                    iterations.append({"bits": bits, "group_size": group_size, "desc_act": desc_act, "damp": damp})

    num_iters = len(iterations)
    logger.info(f"Starting {num_iters} quantizations.")
    count = 1
    for iter in iterations:
        if not os.path.isfile("/workspace/gptq-ppl-test/STOP") and not os.path.isfile(stop_file) and not abort:
            bits = iter['bits']
            group_size = iter['group_size']
            desc_act = iter['desc_act']
            damp = iter['damp']

            output_dir = args.output_dir_base
            try:
                os.makedirs(output_dir, exist_ok=True)

                # Log file has same name as directory + .quantize.log,
                # and is placed alongside model directory, not inside it
                # This ensures that we can delete the output_dir in case of error or abort, without losing the logfile.
                # Therefore the existence of the output_dir is a reliable indicator
                # of whether a model has started or not.
                logger.info(
                    f"[{count} / {num_iters}] Quantizing: bits = {bits} - group_size = {group_size} - desc_act = "
                    f"{desc_act} - damp_percent = {damp} to {output_dir}"
                )
                try:
                    quantize(
                        args.pretrained_model_dir,
                        output_dir,
                        traindataset,
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
