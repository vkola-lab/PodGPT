#!/usr/bin/env python
# coding=utf-8
#
# GNU Affero General Public License v3.0 License
#
# PodGPT: An Audio-augmented Large Language Model for Research and Education
# Copyright (C) 2024 Kolachalama Laboratory at Boston University

####################################################################################

# SPECIAL NOTICE
# THE CODES ARE PROVIDED AND OPEN-SOURCED BY TIMOTHEE LACROIX AT Mistral AI
# https://huggingface.co/mistralai/Mistral-Large-Instruct-2407/blob/main/test.py
# THE CODES ARE MODIFIED BY THE TEAM FROM THE KOLACHALAMA LAB AT BOSTON UNIVERSITY

####################################################################################

import os
import json
import argparse
from typing import Dict

import torch
import safetensors.torch
from huggingface_hub import split_torch_state_dict_into_shards


def save_state_dict(state_dict: Dict[str, torch.Tensor], save_directory: str, max_size_gb: int):
    """
    Saves the state dictionary into multiple shards with each shard having a maximum size specified by max_size_gb.
    :param state_dict: The state dictionary to be saved.
    :param save_directory: The directory where the shards will be saved.
    :param max_size_gb: The maximum size of each shard in gigabytes.
    """
    # Convert max size from gigabytes to bytes
    max_size_bytes = max_size_gb * 1024 ** 3

    # Split the state dictionary into shards
    state_dict_split = split_torch_state_dict_into_shards(
        state_dict,
        max_shard_size=max_size_bytes,
        filename_pattern='model-{part_idx:05d}-of-{num_parts:05d}.safetensors'
    )

    # Save each shard to the specified directory
    for filename, tensors in state_dict_split.filename_to_tensors.items():
        shard = {tensor: state_dict[tensor] for tensor in tensors}
        print("Saving", save_directory, filename)
        safetensors.torch.save_file(shard, os.path.join(save_directory, filename))

    # If the state dictionary is sharded, create an index file
    if state_dict_split.is_sharded:
        index = {
            "metadata": {
                "total_size": sum(tensor.numel() * tensor.element_size() for tensor in state_dict.values())
            },
            "weight_map": state_dict_split.tensor_to_filename,
        }
        with open(os.path.join(save_directory, "model.safetensors.index.json"), "w") as f:
            json.dump(index, f, indent=2)


def main(input_file: str, output_dir: str, max_size_gb: int):
    """
    Splits a large SafeTensor file into smaller shards and saves them to the specified directory.
    :param input_file: The path to the input SafeTensor file.
    :param output_dir: The directory where the output shards will be saved.
    :param max_size_gb: The maximum size of each shard in gigabytes.
    """
    # Create the output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load the original safetensor
    state_dict = safetensors.torch.load_file(input_file)

    # Save the state dict into shards
    save_state_dict(state_dict, save_directory=output_dir, max_size_gb=max_size_gb)


if __name__ == "__main__":
    # Example Usage:
    # python model_split.py --large_file "gptq_model/model.safetensors" --output_dir "split_model" --max_size_gb 5
    parser = argparse.ArgumentParser(description="Splits a large SafeTensor file into smaller shards.")
    parser.add_argument("--large_file", type=str, help="The path to the large SafeTensor file.")
    parser.add_argument("--output_dir", type=int, help="The path to save the smaller shards.")
    parser.add_argument("--max_size_gb", default=5, type=int,
                        help="The maximum size of each shard in gigabytes.")
    args = parser.parse_args()

    # Split the safetensor file into shards
    main(args.input_file, args.output_dir, args.max_size_gb)
