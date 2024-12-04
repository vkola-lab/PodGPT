#!/usr/bin/env python
# coding=utf-8
#
# GNU Affero General Public License v3.0 License
#
# PodGPT: An Audio-augmented Large Language Model for Research and Education
# Copyright (C) 2024 Kolachalama Laboratory at Boston University
#
# LICENSE OF THE FOLLOWING MODELS
#
# LLAMA 3 COMMUNITY LICENSE AGREEMENT
# https://llama.meta.com/llama3/license/
#
# LLAMA 3.1 COMMUNITY LICENSE AGREEMENT
# https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/LICENSE
#
# Mistral LICENSE
# https://www.apache.org/licenses/LICENSE-2.0
#
# GEMMA TERMS OF USE
# https://ai.google.dev/gemma/terms

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.utils import load_config


def main(config):
    """
    Download and save the model into local folder for further usage
    :param config: the YAML configuration file
    """
    model_name = config.get("model_name")
    device_map = config.get("device_map")
    hf_read_token = config.get("hf_read_token")
    save_dir = config.get("save_dir")

    # Load the base model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        use_auth_token=hf_read_token,
        device_map=device_map,
        torch_dtype=torch.bfloat16,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        return_tensors="pt",
        use_fast=False,
        use_auth_token=hf_read_token,
        device_map=device_map,
    )

    if "llama" in model_name or "mistralai" in model_name:
        tokenizer.pad_token = tokenizer.eos_token

    # Save the pre-trained model and tokenizer
    tokenizer.save_pretrained(save_dir)
    print('Successfully save the tokenizer!')

    if "Mixtral-8x" in model_name:
        model.save_pretrained(save_dir, safe_serialization=False)
    else:
        model.save_pretrained(save_dir)
    print('Successfully save the model!\n\n')


if __name__ == "__main__":
    # Load the configuration
    config = load_config(file_name="config_large.yml")
    main(config=config)
