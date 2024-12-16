#!/usr/bin/env python
# coding=utf-8
#
# GNU Affero General Public License v3.0 License
#
# PodGPT: An Audio-augmented Large Language Model for Research and Education
# Copyright (C) 2024 Kolachalama Laboratory at Boston University

import os
import sys
import logging
from argparse import ArgumentParser

import yaml
import torch
from datetime import datetime
from huggingface_hub import login

from lib.evaluation import evaluation
from utils.utils import CustomStream, load_config


def main(config, args):
    """
    Performance Evaluation
    :param config: The configurations
    :param args: The user arguments
    """
    # Load and evaluate the original pre-trained model and your fine-tuned checkpoints
    if args.eval_pretrain:
        config['eval_pretrain'] = True
    else:
        config['eval_pretrain'] = False

    evaluation(config=config, mode=args.mode)


if __name__ == "__main__":
    # Example Usage:
    # python main.py --mode small --eval_pretrain True
    parser = ArgumentParser(description='User arguments')
    parser.add_argument("--mode", type=str, default="small")
    parser.add_argument("--eval_pretrain", type=bool, default=True)
    args = parser.parse_args()

    # Load and merge the configuration
    if args.mode == "small":
        config = load_config(file_name="config_small.yml")
    else:
        config = load_config(file_name="config_large.yml")
    benchmark_config = load_config(file_name="config_benchmark.yml")
    config.update(benchmark_config)

    # Load other configurations
    result_dir = config.get("result_dir")
    hf_read_token = config.get("hf_read_token")

    # get the current working directory
    cwd = os.getcwd()
    login(token=hf_read_token)  # Hugging Face Login

    # print output to the console
    print('\n\nThe current working directory is', cwd, '\n\n')

    # Check out the system assigned GPU id
    count = torch.cuda.device_count()
    print('There are', count, 'GPU/GPUs available!',
          'The devices are:', os.getenv("CUDA_VISIBLE_DEVICES"), '\n')

    # Get the current date and time
    time = datetime.now()

    # Create a subdirectory with the current date
    dir = os.path.join(result_dir, time.strftime("%Y-%m-%d"))
    os.makedirs(dir, exist_ok=True)

    # Create a log file with the exact time as the file name
    name = time.strftime("%H-%M-%S.log.txt")
    path = os.path.join(dir, name)

    # Configure the logging module to write to the log file
    logging.basicConfig(
        filename=path,
        level=logging.INFO,  # Adjust the log level as needed
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Redirect sys.stdout to the custom stream
    stream = CustomStream(path, sys.stdout)
    sys.stdout = stream
    print(yaml.dump(config, default_flow_style=False), '\n\n')
    main(config=config, args=args)
    sys.stdout = sys.__stdout__
