# coding=utf-8
#
# GNU Affero General Public License v3.0 License
#
# PodGPT: An Audio-augmented Large Language Model for Research and Education
# Copyright (C) 2024 Kolachalama Laboratory at Boston University

from datasets import load_dataset


def data_loader(hf_repo):
    """
    Load the pre-training dataset from Hugging Face
    :param hf_repo: the dataset in the Hugging Face
    :return dataset: the training dataset
    """
    dataset = load_dataset(hf_repo, split="train")

    return dataset
