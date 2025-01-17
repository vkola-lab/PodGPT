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
from utils.utils import CustomStream, load_config, str2bool


def main(config, args):
    """
    Performance Evaluation
    :param config: The configurations
    :param args: The user arguments
    """
    if args.mode == "chatgpt":
        evaluation(config=config, mode="chatgpt")
    else:
        # Load and evaluate the original pre-trained model
        if args.eval_pretrain:
            config['eval_pretrain'] = True
            print("Perform evaluation on the original pre-trained model!")
            evaluation(config=config, mode=args.mode, eval_pretrain=True, checkpoint_id=None)

        # Load and evaluate the specific saved checkpoint
        for id in args.id:
            config['eval_pretrain'] = False
            print("The current evaluation checkpoint is ", "checkpoint-" + str(id))
            evaluation(config=config, mode=args.mode, eval_pretrain=False, checkpoint_id=id)


if __name__ == "__main__":
    # Example Usage:
    # python inference.py --mode small --eval_pretrain True --id 35166 52749 70332 87915
    parser = ArgumentParser(description='User arguments')
    parser.add_argument("--mode", type=str, default="small",
                        help="Evaluate the smaller model or larger model or ChatGPT: small/large/quant/chatgpt.")
    parser.add_argument("--eval_pretrain", type=str2bool, default=True,
                        help="Evaluate the original pretrained model: True/False")
    parser.add_argument("--id", type=int, nargs="+", default=[],
                        help="Checkpoint IDs (You can input more IDs "
                             "and the checkpoints will be sequentially evaluated!)")
    args = parser.parse_args()

    # Load and merge the configuration
    if args.mode == "small":  # Evaluate the smaller models
        config = load_config(file_name="config_small.yml")
    elif args.mode == "large":  # Evaluate the larger models
        config = load_config(file_name="config_large.yml")
    elif args.mode == "quant":  # Evaluate the quantized models
        config = load_config(file_name="config_quantization.yml")
    else:
        config = load_config(file_name="config_chatgpt.yml")
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
