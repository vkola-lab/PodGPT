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

from argparse import ArgumentParser

from huggingface_hub import snapshot_download

from utils.utils import load_config


def main(repo_id, repo_type="model", save_dir="./", hf_read_token=None):
    """
    Download the models from Hugging Face repo
    :param repo_id: the id of Hugging Face repo
    :param repo_type: the type of Hugging Face repo: model, datasets, or space
    :param save_dir: the directory to save the model files
    :param hf_read_token: the Hugging face `READ` token
    """
    snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        local_dir=save_dir,
        cache_dir=save_dir,
        # This is my Hugging Face `read` token. Please replace it to yours.
        # https://huggingface.co/settings/tokens
        token=hf_read_token
    )


if __name__ == "__main__":
    # Example Usage:
    # python download_model.py --repo "shuyuej/MedGemma7B-Multilingual" --repo_type "model" --save_dir "./save_folder"
    parser = ArgumentParser(description='User arguments')
    parser.add_argument("--repo", type=str, default='shuyuej/MedGemma7B-Multilingual', help="HF Repo")
    parser.add_argument("--repo_type", type=str, default='model', help="Repo Type: model, dataset")
    parser.add_argument("--save_dir", type=str, default='./save_folder', help="Local save directory")
    args = parser.parse_args()

    # Load the configuration
    config = load_config(file_name="config_small.yml")
    hf_read_token = config.get("hf_read_token")

    # Download the Checkpoints
    main(
        repo_id=args.repo,
        repo_type=args.repo_type,
        save_dir=args.save_dir,
        hf_read_token=hf_read_token
    )
