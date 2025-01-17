# coding=utf-8
#
# GNU Affero General Public License v3.0 License
#
# PodGPT: An Audio-augmented Large Language Model for Research and Education
# Copyright (C) 2024 Kolachalama Laboratory at Boston University


def database_format(item):
    """
    The format the input of the Database
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
