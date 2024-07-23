#!/usr/bin/env python
# coding=utf-8
#
# GNU Affero General Public License v3.0 License
#
# MedPodGPT: A multilingual audio-augmented large language model for medical research and education
# Copyright (C) 2024 Kolachalama Laboratory at Boston University

import argparse

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig
from huggingface_hub import login

from utils.utils import load_config


def load_medical_data():
    """
    Load and tokenize the `Multilingual podcasts: shuyuej/Multilingual-Pretraining-Dataset` dataset.
    :return dataset: the loaded dataset
    """
    dataset = load_dataset("shuyuej/MedPodGPT-Demo-Data", split="train")
    dataset = [s for s in dataset['text']]

    return dataset


def prepare_dataset(dataset, tokenizer, seqlen: int = 2048, nsamples: int = 128):
    """
    Prepare the dataset for quantization
    :param dataset: the loaded dataset
    :param tokenizer: the tokenizer
    :param seqlen: the sequence length
    :param nsamples: the number of samples
    :return dataset: the prepared dataset
    """
    print("Start to tokenize the dataset...")
    enc = tokenizer([d + "\n" for d in dataset], add_special_tokens=False)
    enc = torch.tensor(sum(enc.input_ids, [])).unsqueeze(0)
    print(f"Finished tokenizing {enc.shape[1]} tokens!")

    dataset = []
    for i in range(nsamples):
        input_ids = enc[:, i * seqlen: (i + 1) * seqlen]
        attention_mask = torch.ones_like(input_ids)
        dataset.append(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask
            }
        )

    return dataset


def quantize(repo, bits, group_size, act_order):
    """
    Quantize the model and save the quantized weights.
    :param repo: the model id
    :param bits: the number of bits for quantization
    :param group_size: the group_size in the GPTQ algorithm
    :param act_order: a.k.a, desc_act, quantizing columns in order of decreasing activation size
    :param hf_read_token: the Hugging Face READ Token
    """
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(repo)

    # Load the dataset
    dataset = load_medical_data()
    dataset = prepare_dataset(dataset, tokenizer)

    # Config the GPTQ algorithm
    gptq_config = GPTQConfig(
        bits=bits,
        dataset=dataset,
        group_size=group_size,
        desc_act=act_order,
        use_cuda_fp16=True,
    )

    # Load the original quantized model and conduct quantization
    model = AutoModelForCausalLM.from_pretrained(
        repo,
        quantization_config=gptq_config,
        # Please install FlashAttention by using `pip install flash-attn`
        # attn_implementation="flash_attention_2",
        # Please use your Hugging Face READ Token
        # `read` token: for downloading models
        # For your information: https://huggingface.co/settings/tokens
        # token="YOUR_HUGGING_FACE_READ_TOKEN",
        # device_map="auto"
    )
    model.config.quantization_config.dataset = None

    # Save the quantized model and the tokenizer
    model.save_pretrained(f"{repo}_{bits}bit")
    tokenizer.save_pretrained(f"{repo}_{bits}bit")


if __name__ == "__main__":
    # Example Usage:
    # python quantization_HF.py --repo "CohereForAI/c4ai-command-r-v01" --bits 4 --group_size 128
    parser = argparse.ArgumentParser(description="Quantize LLMs using GPTQ Algorithm")
    parser.add_argument("--repo", type=str, help="The pretrained model ID.")
    parser.add_argument("--bits", default=4, type=int, help="Number of bits for quantization.")
    parser.add_argument("--group_size", default=128, type=int, help="Group size for quantization.")
    parser.add_argument("--act_order", action="store_true", help="Enable act-order")
    args = parser.parse_args()
    
    # Conduct the GPTQ quantization
    quantize(
        config=config,
        model_id=args.model_id,
        bits=args.bits,
        group_size=args.group_size,
        act_order=args.act_order
    )
    