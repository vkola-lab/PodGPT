#!/usr/bin/env python
# coding=utf-8
#
# GNU Affero General Public License v3.0 License
#
# PodGPT: An Audio-augmented Large Language Model for Research and Education
# Copyright (C) 2024 Kolachalama Laboratory at Boston University

import os
import re
import json

import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer


def remove_extra_spaces(text):
    """
    Remove extra spaces from a string.

    :param text: Input string to be processed.
    :return: Processed string with extra spaces removed.
    """
    sentence = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space.
    return sentence


def main(files_path, tokenizer_name):
    """
    Calculate statistics of tokens in text files, such as total, mean, variance, and standard deviation.

    :param files_path: List of file paths for text files to process.
    :param tokenizer_name: Name of the tokenizer to use for processing.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        # This is my Hugging Face read and write tokens. Please replace it to yours.
        # read token: for downloading models
        # For your information: https://huggingface.co/settings/tokens
        token="YOUR_HUGGING_FACE_READ_TOKEN"  # Replace with your Hugging Face read token.
    )

    num_long = 0  # Count of long sentences removed.
    cutting_ratio = 0.95  # Threshold ratio for sentence grouping.
    train_max_len = 2048  # Maximum token length per sentence.
    data = []  # List to store processed data.
    length = []  # List to store token lengths of all samples.
    num = 0  # Counter for the number of samples processed.
    length_per_episode = []  # List to store token lengths per episode.

    for file_path in files_path:
        tokens_per_episode = 0

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

            sentences = sent_tokenize(content, language='english')
            total_tokens = 0
            grouped_sen = ""
            num_of_sentences = len(sentences)

            for count, sen in enumerate(sentences):
                tokens = tokenizer(sen, return_tensors="pt")
                len_tokens = len(tokens['input_ids'][0])

                if cutting_ratio * train_max_len < len_tokens < train_max_len:
                    sen = remove_extra_spaces(sen.strip())
                    data.append({"text": sen})
                    length.append(len_tokens)
                    tokens_per_episode += len_tokens
                    num += 1

                elif len_tokens > train_max_len:
                    num_long += 1

                else:
                    total_tokens += len_tokens
                    if total_tokens <= cutting_ratio * train_max_len:
                        grouped_sen += " " + sen
                        if count == num_of_sentences - 1:
                            grouped_sen = remove_extra_spaces(grouped_sen.strip())
                            data.append({"text": grouped_sen})
                            length.append(total_tokens)
                            tokens_per_episode += total_tokens
                            num += 1
                    else:
                        grouped_sen = remove_extra_spaces(grouped_sen.strip())
                        data.append({"text": grouped_sen})
                        length.append(total_tokens - len_tokens)
                        tokens_per_episode += total_tokens - len_tokens
                        num += 1

                        grouped_sen = remove_extra_spaces(sen.strip())
                        total_tokens = len_tokens

        length_per_episode.append(tokens_per_episode)

    with open("cc_podcast_transcripts.json", 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)

    print('Number of samples: ', num)
    print('Number of long sentences (removed): ', num_long)
    print('The total number of text tokens: ', np.sum(length))
    print("The averaged (mean) text tokens per episode: ", np.mean(length_per_episode))
    print("The std text tokens per episode: ", np.std(length_per_episode))


if __name__ == "__main__":
    file_dir = 'podcasts_transcripts'
    files_path = [
        os.path.join(root, filename)
        for root, _, files in os.walk(file_dir)
        for filename in files
        if filename.endswith(".txt")
    ]
    print("The number of Episode Transcript files: ", len(files_path))

    tokenizer = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    nltk.download('punkt')

    main(files_path=files_path, tokenizer_name=tokenizer)
