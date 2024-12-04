#!/usr/bin/env python
# coding=utf-8
#
# GNU Affero General Public License v3.0 License
#
# PodGPT: An Audio-augmented Large Language Model for Research and Education
# Copyright (C) 2024 Kolachalama Laboratory at Boston University

from argparse import ArgumentParser

from huggingface_hub import HfApi

from utils.utils import load_config


def main(api, args, hf_write_token):
    """
    Upload the trained model to Hugging Face Model Repo
    :param api: Hugging Face API
    :param args: The user arguments
    """
    api.upload_folder(
        # repo_type: 'model', 'dataset', 'space'
        repo_type="model",
        repo_id=args.repo,
        folder_path=args.folder_path,
        # This is my Hugging Face `write` token. Please replace it to yours.
        # https://huggingface.co/settings/tokens
        token=hf_write_token
    )


if __name__ == "__main__":
    # Example Usage:
    # python upload_quantized_model.py \
    # --repo "shuyuej/MedLLaMA3-70B-BASE-MODEL-QUANT" \
    # --folder_path "./gptq_model"
    parser = ArgumentParser(description='User arguments')
    parser.add_argument("--repo", type=str,
                        default='shuyuej/MedLLaMA3-70B-BASE-MODEL-QUANT', help="HF Repo")
    parser.add_argument("--folder_path", type=str, default='./gptq_model', help="The path of the model")
    args = parser.parse_args()

    # Load the configuration
    config = load_config(file_name="config_quantization.yml")
    hf_write_token = config.get("hf_write_token")

    # Upload the Checkpoints
    api = HfApi()
    main(api=api, args=args, hf_write_token=hf_write_token)
