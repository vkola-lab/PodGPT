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


def download_hf_files(repo_id, repo_type="model", save_dir="./"):
    snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        local_dir=save_dir,
        # This is my Hugging Face `read` token. Please replace it to yours.
        # https://huggingface.co/settings/tokens
        token="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"
    )


if __name__ == "__main__":
    # Example Usage:
    # python download_model.py --repo "shuyuej/DrGemma2B" --repo_type "model" --save_dir "./save_folder"
    parser = ArgumentParser(description='User arguments')
    parser.add_argument("--repo", type=str, default='shuyuej/DrGemma2B', help="HF Repo")
    parser.add_argument("--repo_type", type=str, default='model', help="Repo Type: model, dataset")
    parser.add_argument("--save_dir", type=str, default='./save_folder', help="local save path")
    args = parser.parse_args()

    # Download the Checkpoints
    download_hf_files(repo_id=args.repo, repo_type=args.repo_type, save_dir=args.save_dir)
