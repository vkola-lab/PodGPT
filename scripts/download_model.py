#!/usr/bin/env python
# coding=utf-8
#
# GNU Affero General Public License v3.0 License
#
# MedPodGPT: A multilingual audio-augmented large language model for medical research and education
# Copyright (C) 2024 Kolachalama Laboratory at Boston University

from argparse import ArgumentParser

from huggingface_hub import snapshot_download

from utils.utils import load_config


def download_hf_files(repo_id, repo_type="model", save_dir="./", hf_read_token=None):
    """
    Download Hugging Face files, including models
    :param repo_id: your Hugging Face repo ID
    :param repo_type: the type of your repo, i.e., model or dataset
    :param save_dir: the local save path
    :param hf_read_token: your Hugging Face read token
    """
    snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        local_dir=save_dir,
        # This is my Hugging Face `read` token. Please replace it to yours.
        # https://huggingface.co/settings/tokens
        token=hf_read_token
    )


if __name__ == "__main__":
    # Example Usage:
    # python download_model.py --repo "shuyuej/DrGemma2B" --repo_type "model" --save_dir "./save_folder"
    parser = ArgumentParser(description='User arguments')
    parser.add_argument("--repo", type=str, default='shuyuej/DrGemma2B', help="HF Repo")
    parser.add_argument("--repo_type", type=str, default='model', help="Repo Type: model, dataset")
    parser.add_argument("--save_dir", type=str, default='./save_folder', help="local save path")
    args = parser.parse_args()

    # Load the configuration
    config = load_config(file_name="config_small.yml")
    hf_read_token = config.get("hf_read_token")

    # Download the Checkpoints
    download_hf_files(
        repo_id=args.repo,
        repo_type=args.repo_type,
        save_dir=args.save_dir,
        hf_read_token=hf_read_token
    )
