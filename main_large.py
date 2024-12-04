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

import yaml
import torch
from datetime import datetime
from huggingface_hub import login

from lib.model_loader_large import model_loader, trainer_loader
from lib.data_manager import data_loader
from utils.utils import CustomStream, load_config


def main(config):
    """Run the program"""
    # Retrieve the pathes of needed hyperparameters
    epochs = config.get("epochs")
    dataset_hf = config.get("dataset_hf")

    # Load the model and tokenizer
    print("Start the Pre-training process......")
    model, tokenizer = model_loader(config)

    # Load dataset
    dataset = data_loader(hf_repo=dataset_hf)

    # Load Trainer
    trainer = trainer_loader(
        config,
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        num_train_epochs=epochs
    )

    # Start the training process
    trainer.train()


if __name__ == "__main__":
    # Load the configuration
    config = load_config(file_name="config_large.yml")
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
    main(config=config)
    sys.stdout = sys.__stdout__
    