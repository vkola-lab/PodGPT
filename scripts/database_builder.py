#!/usr/bin/env python
# coding=utf-8
#
# GNU Affero General Public License v3.0 License
#
# MedPodGPT: A multilingual audio-augmented large language model for medical research and education
# Copyright (C) 2024 Kolachalama Laboratory at Boston University

import os
import re
import json

import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer


def has_successive_three_repeats(text):
    """
    Determine if a string has three repeated words.
    :param text: the input string
    :return: True if the string has three repeated words, False otherwise
    """
    # Split text into sentences
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a.z]\.)(?<=\.|\?|\!)\s', text.replace('\n', ' '))

    # Function to find consecutive repeated phrases
    def find_successive_repeats(sentence):
        words = sentence.split()

        # Initialize variables to track repetitions
        previous_word = None
        repeat_count = 1

        # Iterate through words to find successive repetitions
        for word in words:
            if word == previous_word:
                repeat_count += 1
            else:
                if repeat_count >= 3:  # Check if the last tracked word was repeated three times or more
                    return True
                # Reset for new word
                previous_word = word
                repeat_count = 1

        # Handle case where the last set of words might be a repeat
        if repeat_count >= 3:
            return True

        return False

    # Check each sentence for successive repeated phrases
    for sentence in sentences:
        if find_successive_repeats(sentence):
            return True
    return False


def remove_extra_spaces(text):
    """
    Remove extra spaces from a string.
    :param text: the input string
    :return sentence: the processed string
    """
    # Remove extra spaces from each texts
    sentence = re.sub(r'\s+', ' ', text)

    return sentence


def detect_languages(text):
    """
    Determine the language of a string.
    :param text: the input string
    :return: True if the input string is in Chinese, Korean, Arabic, or Japanese, False otherwise
    """
    # Define regex patterns for Chinese, Korean, Arabic, Russian, and Japanese
    patterns = {
        'Chinese': '[\u4e00-\u9fff]',
        'Korean': '[\uac00-\ud7af]',
        'Arabic': '[\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff\uFB50-\uFDFF\uFE70-\uFEFF]',
        'Russian': '[\u0400-\u04ff\u0500-\u052f]',
        'Japanese': '[\u3040-\u309f\u30a0-\u30ff\u4e00-\u9faf\u3000-\u303f]'
    }

    # Check for matches
    for language, pattern in patterns.items():
        if re.search(pattern, text):
            return True

    return False


def main(folder_path, tokenizer_name, language):
    """
    Calculate some statistics of tokens in txt files,
    such as the total number of tokens, the averaged number of tokens, the variance of tokens, etc.
    :param folder_path: the path to the txt files
    :param tokenizer_name: which tokenizer to use
    :param language: the language of the text files, i.e., english, spanish, or french
    """
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        token="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"
    )

    num_long = 0
    cutting_ratio = 0.95
    train_max_len = 2048
    data = []
    length = []
    num = 0
    num_removed = 0
    length_per_episode = []

    # List all files in the given directory
    files = os.listdir(folder_path)
    for file in files:
        tokens_per_episode = 0

        # Check if the file is a text file
        if file.endswith('.txt'):
            file_path = os.path.join(folder_path, file)

            # Open and read the contents of the file
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

                # Tokenize the content into sentences
                sentences = sent_tokenize(content, language=language)

                # Group sentences
                total_tokens = 0
                grouped_sen = ""
                num_of_sentences = len(sentences)
                for count, sen in enumerate(sentences):
                    non_eng = detect_languages(sen)
                    repeated = has_successive_three_repeats(sen)

                    # Check if there is non-english language or repeated words inside this sentence
                    if not non_eng and not repeated:
                        tokens = tokenizer(sen, return_tensors="pt")
                        len_tokens = len(tokens['input_ids'][0])

                        if cutting_ratio * train_max_len < len_tokens < train_max_len:
                            data.append({"text": sen})
                            length.append(len_tokens)
                            tokens_per_episode += len_tokens
                            num += 1

                        elif len_tokens > train_max_len:
                            num_long += 1
                            pass

                        else:
                            total_tokens += len_tokens
                            if total_tokens <= cutting_ratio * train_max_len:
                                grouped_sen = grouped_sen + " " + sen
                                if count < num_of_sentences - 1:
                                    pass
                                else:
                                    data.append({"text": grouped_sen})
                                    length.append(total_tokens)
                                    tokens_per_episode += total_tokens
                                    num += 1
                            else:
                                grouped_sen = grouped_sen.strip()
                                grouped_sen = remove_extra_spaces(grouped_sen)

                                if total_tokens <= train_max_len:
                                    data.append({"text": grouped_sen})
                                    length.append(total_tokens - len_tokens)
                                    tokens_per_episode += total_tokens - len_tokens
                                    num += 1

                                    grouped_sen = ""
                                    total_tokens = 0
                                else:
                                    # Tokenize the content into sentences
                                    sens = sent_tokenize(grouped_sen, language=language)
                                    grouped_sen = " ".join(sens[:-1])
                                    grouped_sen = grouped_sen.strip()
                                    grouped_sen = remove_extra_spaces(grouped_sen)

                                    tokens_group = tokenizer(grouped_sen, return_tensors="pt")
                                    len_tokens_group = len(tokens_group['input_ids'][0])

                                    data.append({"text": grouped_sen})
                                    length.append(len_tokens_group)
                                    tokens_per_episode += len_tokens_group
                                    num += 1

                                    grouped_sen = sen
                                    total_tokens = len_tokens
                    else:
                        num_removed += 1
                        print('Remove this sentence because it contains repeated or other language words!')

        length_per_episode.append(tokens_per_episode)

    # Write data to a JSON file
    with open(language + "_transcripts.json", 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)

    print('Number of samples: ', num)
    print('Number of removed samples: ', num_removed)
    print('Number of long sentences (removed): ', num_long)

    # Calculate some statistics
    print('The total number of text tokens: ', np.sum(length))
    print("The averaged (mean) text tokens per episode: ", np.mean(length_per_episode))
    print("The std text tokens per episode: ", np.std(length_per_episode))

    return float(np.sum(length)), float(np.mean(length_per_episode)), float(np.std(length_per_episode))


if __name__ == "__main__":
    # Specify the path to the txt files
    folder_path = 'all_the_transcripts_english'
    language = 'english'  # english, french, and spanish
    # folder_path = 'all_the_transcripts_spanish'
    # language = 'spanish'  # english, french, and spanish
    # folder_path = 'all_the_transcripts_french'
    # language = 'french'  # english, french, and spanish

    # Specify which tokenizer we will use
    tokenizer = "meta-llama/Meta-Llama-3-8B-Instruct"

    # Ensure that the punkt tokenizer models are downloaded
    nltk.download('punkt')

    # Execute the main file
    main(folder_path=folder_path, tokenizer_name=tokenizer, language=language)
