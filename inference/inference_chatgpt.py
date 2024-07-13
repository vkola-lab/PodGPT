#!/usr/bin/env python
# coding=utf-8
#
# GNU Affero General Public License v3.0 License
#
# MedPodGPT: A multilingual audio-augmented large language model for medical research and education
# Copyright (C) 2024 Kolachalama Laboratory at Boston University

import os
import sys
import logging

import yaml
from datetime import datetime

from lib.evaluation_chatgpt import evaluation
from utils.utils import CustomStream, load_config


def main(config):
    """
    Performance Evaluation
    :param config: The configurations
    """
    evaluation(config=config)


if __name__ == "__main__":
    # Example Usage:
    # python inference_chatgpt.py

    # Load the configuration
    config = load_config(file_name="config_chatgpt.yml")
    result_dir = config.get("result_dir")

    # get the current working directory
    cwd = os.getcwd()

    # print output to the console
    print('\n\nThe current working directory is', cwd, '\n\n')

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
