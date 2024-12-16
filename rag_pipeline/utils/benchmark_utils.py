# coding=utf-8
#
# GNU Affero General Public License v3.0 License
#
# PodGPT: An Audio-augmented Large Language Model for Research and Education
# Copyright (C) 2024 Kolachalama Laboratory at Boston University


def medqa_format(item):
    """
    The format the input of MedQA Benchmark
    :param item: the input one example
    :return full_question: the formatted question
    """
    # Start with the question part
    question = item["question"] + "\n"

    # Initialize an empty string for the options
    options_str = ""

    # Check if 'options' is indeed a dictionary and contains items
    if isinstance(item["options"], dict) and item["options"]:
        # Retrieve all keys to handle the last key differently
        keys = list(item["options"].keys())

        # Loop through each key-value pair in the dictionary
        for key in keys:  # Iterate over all keys except the last one
            value = item["options"][key]
            # Append the key and value to the options string with a newline for all except the last one
            options_str += f"{key}. {value}\n"

    full_question = question + options_str

    return full_question


def pubmedqa_format(item):
    """
    The format the input of PubMedQA Benchmark
    :param item: the input one example
    :return full_question: the formatted question
    """
    # Start with the question part
    abstract = "\n".join(item["CONTEXTS"])
    full_question = item["QUESTION"]

    return abstract, full_question


def medmcqa_format(item):
    """
    The format the input of MedMCQA Benchmark
    :param item: the input one example
    :return full_question: the formatted question
    """
    # Start with the question part
    question = item["question"] + "\n"

    # Initialize an empty string for the options
    options_str = "A. " + item["opa"] + "\n"
    options_str += "B. " + item["opb"] + "\n"
    options_str += "C. " + item["opc"] + "\n"
    options_str += "D. " + item["opd"] + "\n"

    full_question = question + options_str

    return full_question


def medexpqa_format(item):
    """
    The format the input of MedExpQA Benchmark
    :param item: the input one example
    :return full_question: the formatted question
    """
    # Start with the question part
    question = item["full_question"] + "\n"

    # Initialize an empty string for the options
    options_str = ""

    # Check if 'options' is indeed a dictionary and contains items
    if isinstance(item["options"], dict) and item["options"]:
        # Retrieve all keys to handle the last key differently
        keys = list(item["options"].keys())

        # Loop through each key-value pair in the dictionary
        for key in keys:  # Iterate over all keys except the last one
            value = item["options"][key]

            if key == "1":
                key = "A"
            elif key == "2":
                key = "B"
            elif key == "3":
                key = "C"
            elif key == "4":
                key = "D"
            elif key == "5":
                key = "E"

            if value != "None":
                # Append the key and value to the options string with a newline for all except the last one
                options_str += f"{key}. {value}\n"

    full_question = question + options_str

    return full_question


def mmlu_format(item):
    """
    The format the input of the MMLU Benchmark
    (1) anatomy_test.csv
    (2) clinical_knowledge_test.csv
    (3) college_biology_test.csv
    (4) college_medicine_test.csv
    (5) medical_genetics_test.csv
    (6) professional_medicine_test.csv
    :param item: the input one example
    :return full_question: the formatted question
    """
    # Check if each one is already a string and only convert it to a string if it’s not
    item = list(map(lambda i: str(i) if not isinstance(i, str) else i, item))

    # Start with the question part
    question = item[0] + "\n"

    # Initialize an empty string for the options
    options_str = "A. " + item[1] + "\n"
    options_str += "B. " + item[2] + "\n"
    options_str += "C. " + item[3] + "\n"
    options_str += "D. " + item[4] + "\n"

    full_question = question + options_str

    return full_question
