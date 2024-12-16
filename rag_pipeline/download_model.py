#!/usr/bin/env python
# coding=utf-8
#
# GNU Affero General Public License v3.0 License
#
# PodGPT: An Audio-augmented Large Language Model for Research and Education
# Copyright (C) 2024 Kolachalama Laboratory at Boston University

from argparse import ArgumentParser

from huggingface_hub import snapshot_download

from utils.utils import load_config


def main(repo_id, save_dir="./", hf_read_token=None):
    """
    Download the models from Hugging Face repo
    :param repo_id: the id of Hugging Face repo
    :param save_dir: the directory to save the model files
    :param hf_read_token: the Hugging face `READ` token
    """
    snapshot_download(
        repo_id=repo_id,
        # By default, we set the "repo_type" to "model"
        repo_type="model",
        local_dir=save_dir,
        cache_dir=save_dir,
        # This is my Hugging Face `read` token. Please replace it to yours.
        # https://huggingface.co/settings/tokens
        token=hf_read_token
    )


if __name__ == "__main__":
    # Example Usage:
    # python download_model.py --repo "shuyuej/gemma-2b-it-2048" --save_dir "./save_folder"
    parser = ArgumentParser(description='User arguments')
    parser.add_argument("--repo", type=str, default='shuyuej/gemma-2b-it-2048', help="HF Repo")
    parser.add_argument("--save_dir", type=str, default='./save_folder', help="Local save directory")
    args = parser.parse_args()

    # Load the configuration
    config = load_config(file_name="config_small.yml")
    hf_read_token = config.get("hf_read_token")

    # Download the Checkpoints
    main(repo_id=args.repo, save_dir=args.save_dir, hf_read_token=hf_read_token)
