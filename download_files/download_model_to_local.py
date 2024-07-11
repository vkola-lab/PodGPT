#!/usr/bin/env python
# coding=utf-8
#
# MIT License
#
# Copyright (c) 2024 Kolachalama Lab at Boston University
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# LICENSE OF THE FOLLOWING MODELS
#
# LLAMA 2 COMMUNITY LICENSE AGREEMENT
# https://github.com/facebookresearch/llama/blob/main/LICENSE
#
# LLAMA 3 COMMUNITY LICENSE AGREEMENT
# https://llama.meta.com/llama3/license/
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
