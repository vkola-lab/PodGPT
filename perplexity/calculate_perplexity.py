#!/usr/bin/env python
# coding=utf-8
#
# GNU Affero General Public License v3.0 License
#
# PodGPT: An Audio-augmented Large Language Model for Research and Education
# Copyright (C) 2024 Kolachalama Laboratory at Boston University

import os
import re
import sys
import logging
import chardet
from tqdm import tqdm
from datetime import datetime
from argparse import ArgumentParser

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.utils import CustomStream, str2bool


def detect_encoding(file_path):
    """
    Detects the encoding of a given file using the `chardet` library.

    :param file_path: str, path to the file whose encoding needs to be detected.
    :return: str, detected encoding name.
    """
    # Open the file in binary mode ('rb') to read raw bytes
    with open(file_path, 'rb') as f:
        raw_data = f.read()  # Read the entire file content as bytes

    # Use chardet to analyze the byte content and detect the encoding
    return chardet.detect(raw_data)['encoding']


def clean_text(text):
    """
    Cleans up text by removing certain prefixes and extra spaces.

    :param text: str, input text to be cleaned.
    :return: str, cleaned text.
    """
    # Remove occurrences of 'D: ' or 'P:' at the beginning of lines
    text = re.sub(r'^[DP]: ?', '', text, flags=re.MULTILINE)

    # Replace multiple spaces with a single space and remove leading/trailing spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def process_files(model, tokenizer, input_folder, output_folder):
    """
    Processes text files from the input folder by:
    1. Detecting file encoding.
    2. Reading and cleaning text.
    3. Saving cleaned text to the output folder.
    4. Calculating perplexity of the combined cleaned texts.

    :param model: Pretrained language model for perplexity calculation.
    :param tokenizer: Tokenizer corresponding to the model.
    :param input_folder: str, path to the directory containing input text files.
    :param output_folder: str, path to the directory where cleaned files will be saved.
    """

    # Ensure the output folder exists; create if not
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cleaned_texts = []  # List to store cleaned texts from all files

    # Iterate over all files in the input folder
    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)  # Full path to input file
        output_path = os.path.join(output_folder, filename)  # Full path to output file

        # Process only files (ignore directories)
        if os.path.isfile(input_path):
            try:
                # Detect file encoding dynamically
                encoding = detect_encoding(input_path)

                # Read file content using the detected encoding
                with open(input_path, 'r', encoding=encoding) as file:
                    content = file.read()

                # Clean text using the clean_text function
                cleaned_text = clean_text(content)
                cleaned_texts.append(cleaned_text)  # Store cleaned text

                # Save cleaned text to output folder
                with open(output_path, 'w', encoding='utf-8') as file:
                    file.write(cleaned_text)

                # Uncomment for debugging/logging:
                # print(f"Processed: {filename}")

            except Exception as e:
                print(f"Error processing {filename}: {e}")  # Handle errors gracefully

    # Compute perplexity if there are cleaned texts
    if cleaned_texts:
        texts = "\n\n".join(cleaned_texts)  # Combine texts with spacing
        calculate_perplexity(model, tokenizer, texts)  # Compute perplexity


def calculate_perplexity(model, tokenizer, texts):
    """
    Computes the perplexity of given texts using a language model.

    :param model: Pretrained language model (e.g., from Hugging Face).
    :param tokenizer: Tokenizer corresponding to the model.
    :param texts: str, input text for which perplexity is calculated.
    :return: torch.Tensor, calculated perplexity.

    Reference: https://huggingface.co/docs/transformers/perplexity
    """
    max_length = 1024  # Maximum sequence length processed at a time
    stride = 512  # Overlapping stride between consecutive windows

    # Tokenize the input text and return as PyTorch tensors
    encodings = tokenizer(texts, return_tensors="pt")
    seq_len = encodings.input_ids.size(1)  # Total number of tokens in the text

    # Initialize negative log-likelihood sum and token count
    nll_sum = 0.0
    n_tokens = 0
    prev_end_loc = 0  # Track previous segment end position

    # Iterate over the text in chunks using a sliding window approach
    for begin_loc in tqdm(range(0, seq_len, stride), desc="Calculating Perplexity"):
        end_loc = min(begin_loc + max_length, seq_len)  # Define chunk boundaries
        trg_len = end_loc - prev_end_loc  # Target length for loss calculation

        # Set device (GPU if available, else CPU)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Extract the chunk of input tokens and move to the appropriate device
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)

        # Adding this for Gemma 7B model
        # https://github.com/huggingface/transformers/issues/29250#issuecomment-1966149282
        # https://github.com/huggingface/transformers/issues/29250
        # input_ids[:, 0] = 2  # bos token

        # Clone input_ids for use as target_ids and mask non-target tokens
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100  # Mask non-target tokens with -100

        # Forward pass through the model without gradient computation
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss  # Cross-entropy loss

        # Compute the number of valid tokens in the current chunk
        num_valid_tokens = (target_ids != -100).sum().item()
        batch_size = target_ids.size(0)

        # Adjust for internal label shift (model shifts labels left by 1)
        num_loss_tokens = num_valid_tokens - batch_size

        # Accumulate total negative log-likelihood and token count
        nll_sum += neg_log_likelihood * num_loss_tokens
        n_tokens += num_loss_tokens

        # Update previous segment end location
        prev_end_loc = end_loc

        # Exit loop if the entire text has been processed
        if end_loc == seq_len:
            break

    # Compute average negative log-likelihood per token
    avg_nll = nll_sum / n_tokens

    # Compute perplexity (exp of average negative log-likelihood)
    ppl = torch.exp(avg_nll)

    print("The perplexity is", ppl)
    return ppl


if __name__ == "__main__":
    # Examples usage
    # Evaluate the baseline models
    # python calculate_perplexity.py --evaluate baseline --model google/gemma-2b-it
    # python calculate_perplexity.py --evaluate baseline --model google/gemma-7b-it
    # python calculate_perplexity.py --evaluate baseline --model shuyuej/Llama-3.3-70B-Instruct-GPTQ
    # python calculate_perplexity.py --evaluate baseline --model mistralai/Mixtral-8x7B-Instruct-v0.1
    # python calculate_perplexity.py --evaluate baseline --model meta-llama/Llama-3.3-70B-Instruct

    # Evaluate PodGPT models
    # python download_model_from_hf.py shuyuej/gemma-2b-it-2048
    # python calculate_perplexity.py --evaluate PodGPT --model shuyuej/gemma-2b-it-2048 --id 9456 18912 28368 37824 47280
    #
    # python download_model_from_hf.py shuyuej/gemma-7b-it-2048
    # python calculate_perplexity.py --evaluate PodGPT --model shuyuej/gemma-7b-it-2048 --id 18912 37824 56736 75648 94560
    #
    # python download_model_from_hf.py shuyuej/PodGPT-v0.1
    # python calculate_perplexity.py --evaluate PodGPT --model shuyuej/Llama-3.3-70B-Instruct-GPTQ --lora True --id 18912 37824 56736 75648 94560
    #
    # python download_model_from_hf.py shuyuej/Mixtral-8x7B-Instruct-v0.1-2048
    # python calculate_perplexity.py --evaluate PodGPT --model mistralai/Mixtral-8x7B-Instruct-v0.1 --lora True -id 20481 40962 61443 81924 102405
    #
    # python download_model_from_hf.py shuyuej/Llama-3.3-70B-Instruct-2048
    # python calculate_perplexity.py --evaluate PodGPT--model meta-llama/Llama-3.3-70B-Instruct --lora True --id 18640

    # Parse command-line arguments
    parser = ArgumentParser(description="User arguments")
    parser.add_argument(
        "--evaluate", type=str, default="baseline",
        help="Specify whether to evaluate PodGPT or baseline"
    )
    parser.add_argument(
        "--model", type=str, default="google/gemma-2b-it",
        help="Specify which model to evaluate"
    )
    parser.add_argument(
        "--lora", type=str2bool, default=False,
        help="Whether there is LoRA adapter in this model"
    )
    parser.add_argument(
        "--id", type=int, nargs="+", default=[],
        help="Checkpoint IDs (multiple IDs can be provided to upload sequentially)"
    )
    args = parser.parse_args()

    # Hugging Face authentication token (Replace with actual token)
    hf_read_token = "YOUR_HUGGINGFACE_READ_TOKEN"

    # Disable parallelism in tokenizers to avoid potential deadlocks
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Get the current date and time for logging and output organization
    time = datetime.now()

    # Define input and output directories for processing
    input_folder = "Clean Transcripts"  # Folder containing text files
    output_folder = "output_texts"  # Folder to save processed files

    # Device map for model deployment (auto-distributes across available hardware)
    device_map = "auto"

    # Define result directory for storing logs
    result_dir = "record_" + args.model.replace("/", "-")
    log_subdir = os.path.join(result_dir, time.strftime("%Y-%m-%d"))  # Date-based subfolder
    os.makedirs(log_subdir, exist_ok=True)  # Create directory if it doesn't exist

    # Define log file with timestamp
    log_file = os.path.join(log_subdir, time.strftime("%H-%M-%S.log.txt"))

    # Configure logging to store outputs in a log file
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,  # Adjust log level as needed (e.g., DEBUG, WARNING, ERROR)
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Redirect stdout to a custom stream for logging
    stream = CustomStream(log_file, sys.stdout)

    # Check if the model name contains "shuyuej" (indicating a specific checkpoint evaluation)
    if args.evaluate == "baseline":
        # If a custom model path is provided instead of a checkpoint ID
        model_path = args.model

        # Load the model
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            use_auth_token=hf_read_token,
            device_map=device_map,
            torch_dtype=torch.bfloat16,
        )

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            return_tensors="pt",
            use_auth_token=hf_read_token,
            device_map=device_map,
        )

        # Redirect output to log file
        sys.stdout = stream
        process_files(model, tokenizer, input_folder, output_folder)  # Process input files
        sys.stdout = sys.__stdout__  # Restore standard output

    else:  # args.evaluate == "PodGPT"
        if args.lora is True:
            for id in args.id:
                model_path = args.model
                lora_path = os.path.join("save_folder", "checkpoint-" + str(id))

                # Load the base model and LoRA from the checkpoint
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    use_auth_token=hf_read_token,
                    device_map=device_map,
                    torch_dtype=torch.bfloat16,
                )
                if args.model == "mistralai/Mixtral-8x7B-Instruct-v0.1":
                    # "mistralai/Mixtral-8x7B-Instruct-v0.1" doesn't work using the model.load_adapter(lora_path)
                    # So we load it using the PeftModel.from_pretrained() method
                    model = PeftModel.from_pretrained(
                        model,
                        lora_path
                    )
                else:
                    model.load_adapter(lora_path)

                # Load the corresponding tokenizer
                tokenizer = AutoTokenizer.from_pretrained(
                    model_path,
                    return_tensors="pt",
                    use_auth_token=hf_read_token,
                    device_map=device_map,
                )
                # Redirect output to log file
                sys.stdout = stream
                process_files(model, tokenizer, input_folder, output_folder)  # Process input files
                sys.stdout = sys.__stdout__  # Restore standard output
        else:
            for id in args.id:
                model_path = os.path.join("save_folder", f"checkpoint-{id}")  # Construct checkpoint path

                # Load the model from the checkpoint
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    use_auth_token=hf_read_token,
                    device_map=device_map,
                    torch_dtype=torch.bfloat16,  # Use BF16 precision for efficiency
                )

                # Load the corresponding tokenizer
                tokenizer = AutoTokenizer.from_pretrained(
                    model_path,
                    return_tensors="pt",
                    use_auth_token=hf_read_token,
                    device_map=device_map,
                )

                # Redirect output to log file
                sys.stdout = stream
                process_files(model, tokenizer, input_folder, output_folder)  # Process input files
                sys.stdout = sys.__stdout__  # Restore standard output
