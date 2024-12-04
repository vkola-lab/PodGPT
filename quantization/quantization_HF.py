#!/usr/bin/env python
# coding=utf-8
#
# GNU Affero General Public License v3.0 License
#
# PodGPT: An Audio-augmented Large Language Model for Research and Education
# Copyright (C) 2024 Kolachalama Laboratory at Boston University

import argparse

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig
from huggingface_hub import login

from utils.utils import load_config


def dataset_loader():
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
        dataset.append({"input_ids": input_ids, "attention_mask": attention_mask})

    return dataset


def main(repo, bits, group_size, act_order, hf_read_token):
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
    dataset = dataset_loader()
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
        # This is my Hugging Face `read` token. Please replace it to yours.
        # https://huggingface.co/settings/tokens
        token=hf_read_token,
        # device_map="auto"
    )
    model.config.quantization_config.dataset = None

    # Save the quantized model and the tokenizer
    model.save_pretrained(f"{repo}_{bits}bit")
    tokenizer.save_pretrained(f"{repo}_{bits}bit")

    # Create the index file for the quantized model
    state_dict = model.state_dict()
    total_size = sum(tensor.numel() * tensor.element_size() for tensor in state_dict.values())

    # Index file content
    index = {
        "metadata": {
            "total_size": total_size,
        },
        "weight_map": {key: "model.safetensors" for key in state_dict.keys()},  # Map all weights to a single file
    }

    index_file_path = os.path.join(model_save_path, "model.safetensors.index.json")
    with open(index_file_path, "w") as f:
        json.dump(index, f, indent=2)
    print("Saved index file to", index_file_path)


if __name__ == "__main__":
    # Example Usage:
    # python quantization_HF.py --repo "meta-llama/Meta-Llama-3-70B-Instruct" --bits 4 --group_size 128
    parser = argparse.ArgumentParser(description="Quantize LLMs using the GPTQ Algorithm.")
    parser.add_argument("--repo", type=str, help="The pretrained model ID.")
    parser.add_argument("--bits", default=4, type=int, help="Number of bits for quantization.")
    parser.add_argument("--group_size", default=128, type=int, help="Group size for quantization.")
    parser.add_argument("--act_order", action="store_true", help="Enable act-order")
    args = parser.parse_args()

    # Load the configuration
    config = load_config(file_name="config_quantization.yml")
    hf_read_token = config.get("hf_read_token")
    
    # Conduct the GPTQ quantization
    main(
        config=config,
        model_id=args.model_id,
        bits=args.bits,
        group_size=args.group_size,
        act_order=args.act_order,
        hf_read_token=hf_read_token,
    )
