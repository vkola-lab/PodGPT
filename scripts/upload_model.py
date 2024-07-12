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
