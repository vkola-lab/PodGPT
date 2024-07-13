#!/usr/bin/env python
# coding=utf-8
#
# GNU Affero General Public License v3.0 License
#
# MedPodGPT: A multilingual audio-augmented large language model for medical research and education
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
    for id in args.id:
        api.upload_folder(
            # repo_type: 'model', 'dataset', 'space'
            repo_type="model",
            repo_id=args.repo,
            folder_path="./save_folder/checkpoint-" + str(id),
            path_in_repo="./checkpoint-" + str(id) + "/",
            # This is my Hugging Face `write` token. Please replace it to yours.
            # https://huggingface.co/settings/tokens
            token=hf_write_token
        )


if __name__ == "__main__":
    # Example Usage:
    # python upload_model.py --repo "shuyuej/DrGemma2B" --id 35166 52749 70332 87915
    parser = ArgumentParser(description='User arguments')
    parser.add_argument("--repo", type=str, default='shuyuej/DrGemma2B', help="HF Repo")
    parser.add_argument("--id", type=int, nargs="+", default=[],
                        help="Checkpoint IDs (You can input more IDs "
                             "and the checkpoints will be sequentially uploaded!)")
    args = parser.parse_args()

    # Load the configuration
    config = load_config(file_name="config_small.yml")
    hf_write_token = config.get("hf_write_token")

    # Upload the Checkpoints
    api = HfApi()
    main(api=api, args=args, hf_write_token=hf_write_token)
