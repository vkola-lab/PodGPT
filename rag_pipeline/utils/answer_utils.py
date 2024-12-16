# coding=utf-8
#
# GNU Affero General Public License v3.0 License
#
# PodGPT: An Audio-augmented Large Language Model for Research and Education
# Copyright (C) 2024 Kolachalama Laboratory at Boston University

import re


def extract_answer(completion, option_range="a-eA-E"):
    """
    Extract answer ("A", "B", "C", "D", etc.) from the response.
    :param completion: model's response
    :param option_range: range of options to look for (default is 'a-eA-E')
    :return: the best option to the multi-option problem
    """
    # The range of letters to look for (default is '\u4e00-\u9fffa-zA-ZÀ-ÖØ-öø-ÿÁÉÍÓÚÜáéíóúüÑñ\u0900-\u097F')
    potential_letters = "\u4e00-\u9fffa-zA-ZÀ-ÖØ-öø-ÿÁÉÍÓÚÜáéíóúüÑñ\u0900-\u097F"

    # Replace asterisks with newlines for better processing
    completion = completion.replace("*", "\n")

    # Replace letter\n\n to letter.
    for letter in ['a', 'b', 'c', 'd', 'e', 'A', 'B', 'C', 'D', 'E']:
        completion = completion.replace(letter + "\n\n", letter + ".")

    # Determine the last letter in the option range
    last_letter = option_range[-1]

    # Adjust the option range for patterns excluding 'A'
    if last_letter.lower() == 'b':
        opt_wo_a = "bA-B"
    else:
        opt_wo_a = "b-" + last_letter.lower() + "A-" + last_letter.upper()

    # Extend potential letters to include numbers
    letter_and_num = potential_letters + "0-9"

    # Define patterns to match the correct answer format
    patterns = [
        # Matches "A.", "B.", etc. at the beginning of a line
        re.compile(rf'^([{option_range}])\.'),

        # Matches "correct answer is (A)" and similar formats
        re.compile(rf'[cC]orrect answer is[^{potential_letters}]*\(([{option_range}])\)[^{potential_letters}]'),
        re.compile(rf'[cC]orrect answer is[^{potential_letters}]*\[([{option_range}])\][^{potential_letters}]'),
        re.compile(rf'[cC]orrect answer is[^{potential_letters}]*\{{([{option_range}])\}}[^{potential_letters}]'),
        re.compile(rf'[cC]orrect answer is[^{potential_letters}]*([{option_range}])\)[^{potential_letters}]'),
        re.compile(rf'[cC]orrect answer is[^{potential_letters}]*([{option_range}])[^{letter_and_num}]'),
        re.compile(rf'[cC]orrect answer is[^{potential_letters}]*\(([{option_range}])\)'),
        re.compile(rf'[cC]orrect answer is[^{potential_letters}]*\[([{option_range}])\]'),
        re.compile(rf'[cC]orrect answer is[^{potential_letters}]*\{{([{option_range}])\}}'),
        re.compile(rf'[cC]orrect answer is[^{potential_letters}]*([{option_range}])\)'),
        re.compile(rf'[cC]orrect answer is[^{potential_letters}]*([{option_range}])$'),

        # Matches "best answer is (A)" and similar formats
        re.compile(rf'[bB]est answer is[^{potential_letters}]*\(([{option_range}])\)[^{potential_letters}]'),
        re.compile(rf'[bB]est answer is[^{potential_letters}]*\[([{option_range}])\][^{potential_letters}]'),
        re.compile(rf'[bB]est answer is[^{potential_letters}]*\{{([{option_range}])\}}[^{potential_letters}]'),
        re.compile(rf'[bB]est answer is[^{potential_letters}]*([{option_range}])\)[^{potential_letters}]'),
        re.compile(rf'[bB]est answer is[^{potential_letters}]*([{option_range}])[^{letter_and_num}]'),
        re.compile(rf'[bB]est answer is[^{potential_letters}]*\(([{option_range}])\)'),
        re.compile(rf'[bB]est answer is[^{potential_letters}]*\[([{option_range}])\]'),
        re.compile(rf'[bB]est answer is[^{potential_letters}]*\{{([{option_range}])\}}'),
        re.compile(rf'[bB]est answer is[^{potential_letters}]*([{option_range}])\)'),
        re.compile(rf'[bB]est answer is[^{potential_letters}]*([{option_range}])$'),

        # Matches "correct option is (A)" and similar formats
        re.compile(rf'[cC]orrect option is[^{potential_letters}]*\(([{option_range}])\)[^{potential_letters}]'),
        re.compile(rf'[cC]orrect option is[^{potential_letters}]*\[([{option_range}])\][^{potential_letters}]'),
        re.compile(rf'[cC]orrect option is[^{potential_letters}]*\{{([{option_range}])\}}[^{potential_letters}]'),
        re.compile(rf'[cC]orrect option is[^{potential_letters}]*([{option_range}])\)[^{potential_letters}]'),
        re.compile(rf'[cC]orrect option is[^{potential_letters}]*([{option_range}])[^{letter_and_num}]'),
        re.compile(rf'[cC]orrect option is[^{potential_letters}]*\(([{option_range}])\)'),
        re.compile(rf'[cC]orrect option is[^{potential_letters}]*\[([{option_range}])\]'),
        re.compile(rf'[cC]orrect option is[^{potential_letters}]*\{{([{option_range}])\}}'),
        re.compile(rf'[cC]orrect option is[^{potential_letters}]*([{option_range}])\)'),
        re.compile(rf'[cC]orrect option is[^{potential_letters}]*([{option_range}])$'),

        # Matches "best option is (A)" and similar formats
        re.compile(rf'[bB]est option is[^{potential_letters}]*\(([{option_range}])\)[^{potential_letters}]'),
        re.compile(rf'[bB]est option is[^{potential_letters}]*\[([{option_range}])\][^{potential_letters}]'),
        re.compile(rf'[bB]est option is[^{potential_letters}]*\{{([{option_range}])\}}[^{potential_letters}]'),
        re.compile(rf'[bB]est option is[^{potential_letters}]*([{option_range}])\)[^{potential_letters}]'),
        re.compile(rf'[bB]est option is[^{potential_letters}]*([{option_range}])[^{letter_and_num}]'),
        re.compile(rf'[bB]est option is[^{potential_letters}]*\(([{option_range}])\)'),
        re.compile(rf'[bB]est option is[^{potential_letters}]*\[([{option_range}])\]'),
        re.compile(rf'[bB]est option is[^{potential_letters}]*\{{([{option_range}])\}}'),
        re.compile(rf'[bB]est option is[^{potential_letters}]*([{option_range}])\)'),
        re.compile(rf'[bB]est option is[^{potential_letters}]*([{option_range}])$'),

        # Matches "correct choice is (A)" and similar formats
        re.compile(rf'[cC]orrect choice is[^{potential_letters}]*\(([{option_range}])\)[^{potential_letters}]'),
        re.compile(rf'[cC]orrect choice is[^{potential_letters}]*\[([{option_range}])\][^{potential_letters}]'),
        re.compile(rf'[cC]orrect choice is[^{potential_letters}]*\{{([{option_range}])\}}[^{potential_letters}]'),
        re.compile(rf'[cC]orrect choice is[^{potential_letters}]*([{option_range}])\)[^{potential_letters}]'),
        re.compile(rf'[cC]orrect choice is[^{potential_letters}]*([{option_range}])[^{letter_and_num}]'),
        re.compile(rf'[cC]orrect choice is[^{potential_letters}]*\(([{option_range}])\)'),
        re.compile(rf'[cC]orrect choice is[^{potential_letters}]*\[([{option_range}])\]'),
        re.compile(rf'[cC]orrect choice is[^{potential_letters}]*\{{([{option_range}])\}}'),
        re.compile(rf'[cC]orrect choice is[^{potential_letters}]*([{option_range}])\)'),
        re.compile(rf'[cC]orrect choice is[^{potential_letters}]*([{option_range}])$'),

        # Matches "best choice is (A)" and similar formats
        re.compile(rf'[bB]est choice is[^{potential_letters}]*\(([{option_range}])\)[^{potential_letters}]'),
        re.compile(rf'[bB]est choice is[^{potential_letters}]*\[([{option_range}])\][^{potential_letters}]'),
        re.compile(rf'[bB]est choice is[^{potential_letters}]*\{{([{option_range}])\}}[^{potential_letters}]'),
        re.compile(rf'[bB]est choice is[^{potential_letters}]*([{option_range}])\)[^{potential_letters}]'),
        re.compile(rf'[bB]est choice is[^{potential_letters}]*([{option_range}])[^{letter_and_num}]'),
        re.compile(rf'[bB]est choice is[^{potential_letters}]*\(([{option_range}])\)'),
        re.compile(rf'[bB]est choice is[^{potential_letters}]*\[([{option_range}])\]'),
        re.compile(rf'[bB]est choice is[^{potential_letters}]*\{{([{option_range}])\}}'),
        re.compile(rf'[bB]est choice is[^{potential_letters}]*([{option_range}])\)'),
        re.compile(rf'[bB]est choice is[^{potential_letters}]*([{option_range}])$'),

        # Matches "answer is (A)" and similar formats
        re.compile(rf'[aA]nswer is[^{potential_letters}]*\(([{option_range}])\)[^{potential_letters}]'),
        re.compile(rf'[aA]nswer is[^{potential_letters}]*\[([{option_range}])\][^{potential_letters}]'),
        re.compile(rf'[aA]nswer is[^{potential_letters}]*\{{([{option_range}])\}}[^{potential_letters}]'),
        re.compile(rf'[aA]nswer is[^{potential_letters}]*([{option_range}])\)[^{potential_letters}]'),
        re.compile(rf'[aA]nswer is[^{potential_letters}]*([{option_range}])[^{letter_and_num}]'),
        re.compile(rf'[aA]nswer is[^{potential_letters}]*\(([{option_range}])\)'),
        re.compile(rf'[aA]nswer is[^{potential_letters}]*\[([{option_range}])\]'),
        re.compile(rf'[aA]nswer is[^{potential_letters}]*\{{([{option_range}])\}}'),
        re.compile(rf'[aA]nswer is[^{potential_letters}]*([{option_range}])\)'),
        re.compile(rf'[aA]nswer is[^{potential_letters}]*([{option_range}])$'),

        # Matches "option is (A)" and similar formats
        re.compile(rf'[oO]ption is[^{potential_letters}]*\(([{option_range}])\)[^{potential_letters}]'),
        re.compile(rf'[oO]ption is[^{potential_letters}]*\[([{option_range}])\][^{potential_letters}]'),
        re.compile(rf'[oO]ption is[^{potential_letters}]*\{{([{option_range}])\}}[^{potential_letters}]'),
        re.compile(rf'[oO]ption is[^{potential_letters}]*([{option_range}])\)[^{potential_letters}]'),
        re.compile(rf'[oO]ption is[^{potential_letters}]*([{option_range}])[^{letter_and_num}]'),
        re.compile(rf'[oO]ption is[^{potential_letters}]*\(([{option_range}])\)'),
        re.compile(rf'[oO]ption is[^{potential_letters}]*\[([{option_range}])\]'),
        re.compile(rf'[oO]ption is[^{potential_letters}]*\{{([{option_range}])\}}'),
        re.compile(rf'[oO]ption is[^{potential_letters}]*([{option_range}])\)'),
        re.compile(rf'[oO]ption is[^{potential_letters}]*([{option_range}])$'),

        # Matches "choice is (A)" and similar formats
        re.compile(rf'[cC]hoice is[^{potential_letters}]*\(([{option_range}])\)[^{potential_letters}]'),
        re.compile(rf'[cC]hoice is[^{potential_letters}]*\[([{option_range}])\][^{potential_letters}]'),
        re.compile(rf'[cC]hoice is[^{potential_letters}]*\{{([{option_range}])\}}[^{potential_letters}]'),
        re.compile(rf'[cC]hoice is[^{potential_letters}]*([{option_range}])\)[^{potential_letters}]'),
        re.compile(rf'[cC]hoice is[^{potential_letters}]*([{option_range}])[^{letter_and_num}]'),
        re.compile(rf'[cC]hoice is[^{potential_letters}]*\(([{option_range}])\)'),
        re.compile(rf'[cC]hoice is[^{potential_letters}]*\[([{option_range}])\]'),
        re.compile(rf'[cC]hoice is[^{potential_letters}]*\{{([{option_range}])\}}'),
        re.compile(rf'[cC]hoice is[^{potential_letters}]*([{option_range}])\)'),
        re.compile(rf'[cC]hoice is[^{potential_letters}]*([{option_range}])$'),

        # Matches "is: (A)" and similar formats
        re.compile(
            rf'is[^{potential_letters}]*:+[^{potential_letters}]*\n+[^{potential_letters}]*([{option_range}])\)'
        ),
        re.compile(rf'is[^{potential_letters}]*\n+[^{potential_letters}]*([{option_range}])\)'),
        # Matches "would be (A)" and similar formats
        re.compile(rf'would be[^{potential_letters}]*\(([{option_range}])\)[^{potential_letters}]'),
        re.compile(rf'would be[^{potential_letters}]*\[([{option_range}])\][^{potential_letters}]'),
        re.compile(rf'would be[^{potential_letters}]*\{{([{option_range}])\}}[^{potential_letters}]'),
        re.compile(rf'would be[^{letter_and_num}]*([{option_range}])\)[^{potential_letters}]'),
        re.compile(
            rf'would be[^{letter_and_num}]*([{opt_wo_a}])[^'
            rf'{letter_and_num}]'
        ),
        re.compile(rf'would be[^{potential_letters}]*\(([{option_range}])\)'),
        re.compile(rf'would be[^{potential_letters}]*\[([{option_range}])\]'),
        re.compile(rf'would be[^{potential_letters}]*\{{([{option_range}])\}}'),
        re.compile(rf'would be[^{potential_letters}]*([{option_range}])\)'),
        re.compile(rf'would be[^{potential_letters}]*([{option_range}])$'),

        # Matches "is (A)" and similar formats
        re.compile(
            rf'is[^{potential_letters}]*:+[^{potential_letters}]*\n+[^{potential_letters}]*\(([{option_range}])\)'
        ),
        re.compile(rf'is[^{potential_letters}]*\n+[^{potential_letters}]*\(([{option_range}])\)'),
        re.compile(
            rf'is[^{potential_letters}]*:+[^{potential_letters}]*\n+[^{potential_letters}]*\[([{option_range}])\]'
        ),
        re.compile(rf'is[^{potential_letters}]*\n+[^{potential_letters}]*\[([{option_range}])\]'),
        re.compile(
            rf'is[^{potential_letters}]*:+[^{potential_letters}]*\n+[^{potential_letters}]*\{{([{option_range}])\}}'
        ),
        re.compile(rf'is[^{potential_letters}]*\n+[^{potential_letters}]*\{{([{option_range}])\}}'),
        re.compile(
            rf'is[^{potential_letters}]*:+[^{potential_letters}]*\n+[^{potential_letters}]*([{option_range}])\)'
        ),
        re.compile(rf'is[^{potential_letters}]*\n+[^{potential_letters}]*([{option_range}])\)'),

        # Matches "be (A)" and similar formats
        re.compile(rf'is[^{letter_and_num}]+([{option_range}])\)'),
        re.compile(rf'be[^{letter_and_num}]+([{option_range}])\)'),
        re.compile(rf'[^{letter_and_num}]+([{option_range}])\)[^{potential_letters}]*is'),
        re.compile(rf'[^{letter_and_num}]+([{option_range}])\)[^{potential_letters}]*would'),
        re.compile(rf'[^{letter_and_num}]+([{option_range}])\)[^{potential_letters}]*could'),
        re.compile(rf'[^{letter_and_num}]+([{option_range}])\)[^{potential_letters}]*will'),

        # Matches "(A)" followed by any other characters
        re.compile(rf':+[^{letter_and_num}]*([{option_range}])\)[^{potential_letters}]'),
        re.compile(rf':+[^{letter_and_num}]*([{option_range}])\)$'),
        re.compile(rf':+[^{letter_and_num}]*([{option_range}])\)'),
        re.compile(rf'[^{letter_and_num}]+([{option_range}])\)$'),
        re.compile(rf'[^{letter_and_num}]+([{option_range}])\)[^{potential_letters}]'),
    ]

    # Check for matches with defined patterns
    for pattern in patterns:
        match = pattern.search(completion)
        if match:
            return match.group(1).upper()

    # Count occurrences of left and right parentheses
    count_left_paren = completion.count('(')
    count_right_paren = completion.count(')')

    # Adjust patterns if right parentheses are more frequent
    if count_right_paren > count_left_paren:
        patterns = [
            # Matches "(A)" followed by any other characters, adjusted for more right parentheses
            re.compile(rf'^[^{letter_and_num}]*([{option_range}])\)[^{potential_letters}]*is'),
            re.compile(rf'^[^{letter_and_num}]*([{option_range}])\)[^{potential_letters}]*would'),
            re.compile(rf'^[^{letter_and_num}]*([{option_range}])\)[^{potential_letters}]*could'),
            re.compile(rf'^[^{letter_and_num}]*([{option_range}])\)[^{potential_letters}]*will'),
            re.compile(rf'^[^{potential_letters}]*[^{letter_and_num}]([{option_range}])\)'),
            re.compile(rf'^[^{letter_and_num}]*([{option_range}])\)'),
            re.compile(rf':+[^{potential_letters}]*\n+[^{potential_letters}]*([{option_range}])\)'),
            re.compile(rf'\n+[^{potential_letters}]*([{option_range}])\)[^{potential_letters}]*is'),
            re.compile(rf'\n+[^{potential_letters}]*([{option_range}])\)[^{potential_letters}]*would'),
            re.compile(rf'\n+[^{potential_letters}]*([{option_range}])\)[^{potential_letters}]*could'),
            re.compile(rf'\n+[^{potential_letters}]*([{option_range}])\)[^{potential_letters}]*will'),
            re.compile(rf'\n+[^{potential_letters}]*([{option_range}])\)[^{potential_letters}]'),
            re.compile(rf'\n+[^{potential_letters}]*([{option_range}])\)$'),
            re.compile(rf'\n+[^{potential_letters}]*([{option_range}])\)'),
            re.compile(rf'is[^{letter_and_num}]+([{option_range}])\)'),
            re.compile(rf'be[^{letter_and_num}]+([{option_range}])\)'),
            re.compile(rf'[^{letter_and_num}]+([{option_range}])\)[^{potential_letters}]*is'),
            re.compile(rf'[^{letter_and_num}]+([{option_range}])\)[^{potential_letters}]*would'),
            re.compile(rf'[^{letter_and_num}]+([{option_range}])\)[^{potential_letters}]*could'),
            re.compile(rf'[^{letter_and_num}]+([{option_range}])\)[^{potential_letters}]*will'),
            re.compile(rf':+[^{letter_and_num}]*([{option_range}])\)[^{potential_letters}]'),
            re.compile(rf':+[^{letter_and_num}]*([{option_range}])\)$'),
            re.compile(rf':+[^{letter_and_num}]*([{option_range}])\)'),
            re.compile(rf'[^{letter_and_num}]+([{option_range}])\)$'),
            re.compile(rf'[^{letter_and_num}]+([{option_range}])\)[^{potential_letters}]'),
        ]

        # Check for matches with the adjusted patterns
        for pattern in patterns:
            match = pattern.search(completion)
            if match:
                return match.group(1).upper()

    # Define additional patterns to match correct answers
    additional_patterns = [
        # Matches "A"
        re.compile(rf"^[^{letter_and_num}]*([{option_range}])[^{letter_and_num}]*$"),

        # Matches "(A) is", "[A] is", "{A} is", and similar formats
        re.compile(rf'\(([{option_range}])\)[^{potential_letters}]*is'),
        re.compile(rf'\[([{option_range}])\][^{potential_letters}]*is'),
        re.compile(rf'\{{([{option_range}])\}}[^{potential_letters}]*is'),
        re.compile(rf'[^{letter_and_num}]([{option_range}])\)[^{potential_letters}]*is'),
        re.compile(
            rf'[^{letter_and_num}]([{option_range}])[^{letter_and_num}][^'
            rf'{potential_letters}]*is'
        ),
        re.compile(rf'^([{option_range}])\)[^{potential_letters}]*is'),
        re.compile(rf'^([{option_range}])[^{letter_and_num}][^{potential_letters}]*is'),

        # Matches "(A) would", "[A] would", "{A} would", and similar formats
        re.compile(rf'\(([{option_range}])\)[^{potential_letters}]*would'),
        re.compile(rf'\[([{option_range}])\][^{potential_letters}]*would'),
        re.compile(rf'\{{([{option_range}])\}}[^{potential_letters}]*would'),
        re.compile(rf'[^{letter_and_num}]([{option_range}])\)[^{potential_letters}]*would'),
        re.compile(
            rf'[^{letter_and_num}]([{option_range}])[^{letter_and_num}][^'
            rf'{potential_letters}]*would'
        ),
        re.compile(rf'^([{option_range}])\)[^{potential_letters}]*would'),
        re.compile(rf'^([{option_range}])[^{letter_and_num}][^{potential_letters}]*would'),

        # Matches "(A) could", "[A] could", "{A} could", and similar formats
        re.compile(rf'\(([{option_range}])\)[^{potential_letters}]*could'),
        re.compile(rf'\[([{option_range}])\][^{potential_letters}]*could'),
        re.compile(rf'\{{([{option_range}])\}}[^{potential_letters}]*could'),
        re.compile(rf'[^{letter_and_num}]([{option_range}])\)[^{potential_letters}]*could'),
        re.compile(
            rf'[^{letter_and_num}]([{option_range}])[^{letter_and_num}][^'
            rf'{potential_letters}]*could'
        ),
        re.compile(rf'^([{option_range}])\)[^{potential_letters}]*could'),
        re.compile(rf'^([{option_range}])[^{letter_and_num}][^{potential_letters}]*could'),

        # Matches "(A) will", "[A] will", "{A} will", and similar formats
        re.compile(rf'\(([{option_range}])\)[^{potential_letters}]*will'),
        re.compile(rf'\[([{option_range}])\][^{potential_letters}]*will'),
        re.compile(rf'\{{([{option_range}])\}}[^{potential_letters}]*will'),
        re.compile(rf'[^{letter_and_num}]([{option_range}])\)[^{potential_letters}]*will'),
        re.compile(
            rf'[^{letter_and_num}]([{option_range}])[^{letter_and_num}][^'
            rf'{potential_letters}]*will'
        ),
        re.compile(rf'^([{option_range}])\)[^{potential_letters}]*will'),
        re.compile(rf'^([{option_range}])[^{letter_and_num}][^{potential_letters}]*will'),

        # Matches "option: (A)" and similar formats
        re.compile(rf'[oO]ption:+[^{potential_letters}]*\(([{option_range}])\)'),
        re.compile(rf'[oO]ption:+[^{potential_letters}]*\[([{option_range}])\]'),
        re.compile(rf'[oO]ption:+[^{potential_letters}]*\{{([{option_range}])\}}'),
        re.compile(
            rf'[oO]ption:+[^{letter_and_num}]*([{option_range}])\)[^{letter_and_num}]'
        ),
        re.compile(rf'[oO]ption:+[^{letter_and_num}]*([{option_range}])\)$'),
        re.compile(
            rf'[oO]ption:+[^{potential_letters}]*[^{letter_and_num}]([{option_range}])\)[^'
            rf'{letter_and_num}]'
        ),
        re.compile(rf'[oO]ption:+[^{potential_letters}]*[^{letter_and_num}]([{option_range}])\)$'),
        re.compile(
            rf'[oO]ption:+[^{letter_and_num}]*([{option_range}])[^{letter_and_num}]'
        ),
        re.compile(rf'[oO]ption:+[^{letter_and_num}]*([{option_range}])$'),
        re.compile(
            rf'[oO]ption:+[^{potential_letters}]*[^{letter_and_num}]([{option_range}])[^'
            rf'{letter_and_num}]'
        ),
        re.compile(rf'[oO]ption:+[^{potential_letters}]*[^{letter_and_num}]([{option_range}])$'),

        # Matches "choice: (A)" and similar formats
        re.compile(rf'[cC]hoice:+[^{potential_letters}]*\(([{option_range}])\)'),
        re.compile(rf'[cC]hoice:+[^{potential_letters}]*\[([{option_range}])\]'),
        re.compile(rf'[cC]hoice:+[^{potential_letters}]*\{{([{option_range}])\}}'),
        re.compile(
            rf'[cC]hoice:+[^{letter_and_num}]*([{option_range}])\)[^{letter_and_num}]'
        ),
        re.compile(rf'[cC]hoice:+[^{letter_and_num}]*([{option_range}])\)$'),
        re.compile(
            rf'[cC]hoice:+[^{potential_letters}]*[^{letter_and_num}]([{option_range}])\)[^'
            rf'{letter_and_num}]'
        ),
        re.compile(rf'[cC]hoice:+[^{potential_letters}]*[^{letter_and_num}]([{option_range}])\)$'),
        re.compile(
            rf'[cC]hoice:+[^{letter_and_num}]*([{option_range}])[^{letter_and_num}]'
        ),
        re.compile(rf'[cC]hoice:+[^{letter_and_num}]*([{option_range}])$'),
        re.compile(
            rf'[cC]hoice:+[^{potential_letters}]*[^{letter_and_num}]([{option_range}])[^'
            rf'{letter_and_num}]'
        ),
        re.compile(rf'[cC]hoice:+[^{potential_letters}]*[^{letter_and_num}]([{option_range}])$'),

        # Matches "answer: (A)" and similar formats
        re.compile(rf' is[^{potential_letters}]+\(([{option_range}])\)[^{potential_letters}]'),
        re.compile(rf' is[^{potential_letters}]+\[([{option_range}])\][^{potential_letters}]'),
        re.compile(rf' is[^{potential_letters}]+\{{([{option_range}])\}}[^{potential_letters}]'),
        re.compile(
            rf' is[^{potential_letters}]*[^{letter_and_num}]([{option_range}])\)[^{potential_letters}]'
        ),
        re.compile(rf' is[^{potential_letters}]+\(([{option_range}])\)$'),
        re.compile(rf' is[^{potential_letters}]+\[([{option_range}])\]$'),
        re.compile(rf' is[^{potential_letters}]+\{{([{option_range}])\}}$'),
        re.compile(rf' is[^{potential_letters}]*[^{letter_and_num}]([{option_range}])\)$'),
        re.compile(rf' is([{option_range}])\)[^{potential_letters}]'),
        re.compile(rf' is([{option_range}])\)$'),
        re.compile(
            rf' is[^{potential_letters}]*[^{letter_and_num}]([{opt_wo_a}])[^'
            rf'{letter_and_num}]'
        ),
        re.compile(rf' is[^{potential_letters}]*[^{letter_and_num}]([{option_range}])$'),
        re.compile(rf' is([{opt_wo_a}])[^{letter_and_num}]'),
        re.compile(rf' is([{option_range}])$'),
        re.compile(rf' is[^{potential_letters}]+\(([{option_range}])\)'),
        re.compile(rf' is[^{potential_letters}]+\[([{option_range}])\]'),
        re.compile(rf' is[^{potential_letters}]+\{{([{option_range}])\}}'),
        re.compile(rf' is[^{potential_letters}]*[^{letter_and_num}]([{option_range}])\)'),
        re.compile(rf' is[^{letter_and_num}]*([{option_range}])\)'),

        # Matches "choice (A)" and similar formats
        re.compile(rf'[cC]hoice[^{potential_letters}]*\(([{option_range}])\)'),
        re.compile(rf'[cC]hoice[^{potential_letters}]*\[([{option_range}])\]'),
        re.compile(rf'[cC]hoice[^{potential_letters}]*\{{([{option_range}])\}}'),
        re.compile(
            rf'[cC]hoice[^{letter_and_num}]*([{option_range}])\)[^{letter_and_num}]'
        ),
        re.compile(rf'[cC]hoice[^{letter_and_num}]*([{option_range}])\)$'),
        re.compile(
            rf'[cC]hoice[^{potential_letters}]*[^{letter_and_num}]([{option_range}])\)[^'
            rf'{letter_and_num}]'
        ),
        re.compile(rf'[cC]hoice[^{potential_letters}]*[^{letter_and_num}]([{option_range}])\)$'),
        re.compile(
            rf'[cC]hoice[^{letter_and_num}]*([{option_range}])[^{letter_and_num}]'
        ),
        re.compile(rf'[cC]hoice[^{letter_and_num}]*([{option_range}])$'),
        re.compile(
            rf'[cC]hoice[^{potential_letters}]*[^{letter_and_num}]([{option_range}])[^'
            rf'{letter_and_num}]'
        ),
        re.compile(rf'[cC]hoice[^{potential_letters}]*[^{letter_and_num}]([{option_range}])$'),

        # Matches "answer (A)" and similar formats
        re.compile(rf'[aA]nswer[^{potential_letters}]*\(([{option_range}])\)'),
        re.compile(rf'[aA]nswer[^{potential_letters}]*\[([{option_range}])\]'),
        re.compile(rf'[aA]nswer[^{potential_letters}]*\{{([{option_range}])\}}'),
        re.compile(
            rf'[aA]nswer[^{letter_and_num}]*([{option_range}])\)[^{letter_and_num}]'
        ),
        re.compile(rf'[aA]nswer[^{letter_and_num}]*([{option_range}])\)$'),
        re.compile(
            rf'[aA]nswer[^{potential_letters}]*[^{letter_and_num}]([{option_range}])\)[^'
            rf'{letter_and_num}]'
        ),
        re.compile(rf'[aA]nswer[^{potential_letters}]*[^{letter_and_num}]([{option_range}])\)$'),
        re.compile(
            rf'[aA]nswer[^{letter_and_num}]*([{option_range}])[^{letter_and_num}]'),
        re.compile(rf'[aA]nswer[^{letter_and_num}]*([{option_range}])$'),
        re.compile(
            rf'[aA]nswer[^{potential_letters}]*[^{letter_and_num}]([{option_range}])[^'
            rf'{letter_and_num}]'
        ),
        re.compile(rf'[aA]nswer[^{potential_letters}]*[^{letter_and_num}]([{option_range}])$'),

        # Matches "option (A)" and similar formats
        re.compile(rf'[Oo]ption[^{potential_letters}]*\(([{option_range}])\)'),
        re.compile(rf'[Oo]ption[^{potential_letters}]*\[([{option_range}])\]'),
        re.compile(rf'[Oo]ption[^{potential_letters}]*\{{([{option_range}])\}}'),
        re.compile(
            rf'[Oo]ption[^{letter_and_num}]*([{option_range}])\)[^{letter_and_num}]'
        ),
        re.compile(rf'[Oo]ption[^{letter_and_num}]*([{option_range}])\)$'),
        re.compile(
            rf'[Oo]ption[^{potential_letters}]*[^{letter_and_num}]([{option_range}])\)[^'
            rf'{letter_and_num}]'
        ),
        re.compile(rf'[Oo]ption[^{potential_letters}]*[^{letter_and_num}]([{option_range}])\)$'),
        re.compile(
            rf'[Oo]ption[^{letter_and_num}]*([{option_range}])[^{letter_and_num}]'
        ),
        re.compile(rf'[Oo]ption[^{letter_and_num}]*([{option_range}])$'),
        re.compile(
            rf'[Oo]ption[^{potential_letters}]*[^{letter_and_num}]([{option_range}])[^'
            rf'{letter_and_num}]'
        ),
        re.compile(rf'[Oo]ption[^{potential_letters}]*[^{letter_and_num}]([{option_range}])$'),
    ]

    # Check for matches with additional patterns
    for pattern in additional_patterns:
        match = pattern.search(completion)
        if match:
            return match.group(1).upper()

    # Define a pattern to identify multiple options in a sequence
    multi_pattern = re.compile(
        rf'[aA][^{potential_letters}]*[bB][^{potential_letters}]*[cC][^{potential_letters}]*[dD]')
    match_multi = multi_pattern.search(completion)
    if match_multi:
        return None

    # Define sliding window patterns to match correct answers within a short text window
    sliding_window_patterns = [
        re.compile(rf'\(([{option_range}])\)'),
        re.compile(rf'\[([{option_range}])\]'),
        re.compile(rf'\{{([{option_range}])\}}'),
        re.compile(
            rf'[^{letter_and_num}]([{opt_wo_a}])[^{letter_and_num}]'),
    ]

    # Use sliding window approach to find matches within the completion text
    for i in range(len(completion) - 2):
        window = completion[i:i + 3]
        for pattern in sliding_window_patterns:
            match = pattern.search(window)
            if match:
                return match.group(1).upper()

    # If no patterns match, return None
    return None


def helper(str):
    """
    A helper function for PubMedQA Benchmark
    """
    if str == "YES":
        return 'A'
    elif str == "NO":
        return 'B'
    elif str == "MAYBE":
        return "C"
    else:
        return None


def extract_answer_for_pubmedqa(completion):
    """
    Extract the best option from responses for PubMedQA Benchmark
    :param completion: model's response
    :return: the best option to the multi-option problem
    """
    # [aA]nswer is [yY]es/no/maybe
    # [aA]nswer: Yes[^a-zA-Z]
    # ^Yes. ^No^a-zA-Z\s] Maybe.
    # is [yY]es[^a-zA-Z]
    # [^a-zA-Z]Yes.
    # [mM]aybe

    length_1 = len("Yes/No/Maybe")
    length_2 = len("Yes/no")

    pattern_y_n_m = re.compile(r'^([yY][eE][sS])/([nN][oO])/([mM][aA][yY][bB][eE])')
    pattern_y_or_n = re.compile(r'^([yY][eE][sS])/([nN][oO])')

    match_1 = pattern_y_n_m.search(completion)
    match_2 = pattern_y_or_n.search(completion)

    if match_1:
        completion = completion[length_1:]

        patterns = [
            re.compile(r'^[^a-zA-Z]*([yY][eE][sS])'),
            re.compile(r'^[^a-zA-Z]*([nN][oO])'),
            re.compile(r'^[^a-zA-Z]*([mM][aA][yY][bB][eE])'),
        ]

        # Loop through the patterns and search for a match at the beginning of the string
        for pattern in patterns:
            match = pattern.match(completion)
            if match:
                return helper(match.group(1).upper())

    elif match_2:
        completion = completion[length_2:]

    patterns = [
        re.compile(r'^[^a-zA-Z]*([yY][eE][sS])[^a-zA-Z]'),
        re.compile(r'^[^a-zA-Z]*([nN][oO])[^a-zA-Z\s]'),
        re.compile(r'^[^a-zA-Z]*([mM][aA][yY][bB][eE])[^a-zA-Z\s]'),
    ]

    # Loop through the patterns and search for a match at the beginning of the string
    for pattern in patterns:
        match = pattern.match(completion)
        if match:
            return helper(match.group(1).upper())

    patterns = [
        re.compile(r'[aA]nswer is[^a-zA-Z]*([yY][eE][sS])[^a-zA-Z\s]'),
        re.compile(r'[aA]nswer is[^a-zA-Z]*([nN][oO])[^a-zA-Z\s]'),
        re.compile(r'[aA]nswer is[^a-zA-Z]*([mM][aA][yY][bB][eE])[^a-zA-Z\s]'),

        re.compile(r'[aA]nswer[^a-zA-Z]*([yY][eE][sS])[^a-zA-Z]'),
        re.compile(r'[aA]nswer[^a-zA-Z]*([nN][oO])[^a-zA-Z\s]'),
        re.compile(r'[aA]nswer[^a-zA-Z]*([mM][aA][yY][bB][eE])[^a-zA-Z\s]'),

        re.compile(r'[aA]nswer is[^a-zA-Z]*([yY][eE][sS])[^a-zA-Z]'),
        re.compile(r'[aA]nswer is[^a-zA-Z]*([nN][oO])[^a-zA-Z]'),
        re.compile(r'[aA]nswer is[^a-zA-Z]*([mM][aA][yY][bB][eE])[^a-zA-Z]'),

        re.compile(r'[aA]nswer[^a-zA-Z]*([nN][oO])[^a-zA-Z]'),
        re.compile(r'[aA]nswer[^a-zA-Z]*([mM][aA][yY][bB][eE])[^a-zA-Z]'),

        re.compile(r'[oO]ption is[^a-zA-Z]*([yY][eE][sS])[^a-zA-Z\s]'),
        re.compile(r'[oO]ption is[^a-zA-Z]*([nN][oO])[^a-zA-Z\s]'),
        re.compile(r'[oO]ption is[^a-zA-Z]*([mM][aA][yY][bB][eE])[^a-zA-Z\s]'),

        re.compile(r'[oO]ption[^a-zA-Z]*([yY][eE][sS])[^a-zA-Z]'),
        re.compile(r'[oO]ption[^a-zA-Z]*([nN][oO])[^a-zA-Z\s]'),
        re.compile(r'[oO]ption[^a-zA-Z]*([mM][aA][yY][bB][eE])[^a-zA-Z\s]'),

        re.compile(r'[oO]ption is[^a-zA-Z]*([yY][eE][sS])[^a-zA-Z]'),
        re.compile(r'[oO]ption is[^a-zA-Z]*([nN][oO])[^a-zA-Z]'),
        re.compile(r'[oO]ption is[^a-zA-Z]*([mM][aA][yY][bB][eE])[^a-zA-Z]'),

        re.compile(r'[oO]ption[^a-zA-Z]*([nN][oO])[^a-zA-Z]'),
        re.compile(r'[oO]ption[^a-zA-Z]*([mM][aA][yY][bB][eE])[^a-zA-Z]'),

        re.compile(r'[cC]hoice is[^a-zA-Z]*([yY][eE][sS])[^a-zA-Z\s]'),
        re.compile(r'[cC]hoice is[^a-zA-Z]*([nN][oO])[^a-zA-Z\s]'),
        re.compile(r'[cC]hoice is[^a-zA-Z]*([mM][aA][yY][bB][eE])[^a-zA-Z\s]'),

        re.compile(r'[cC]hoice[^a-zA-Z]*([yY][eE][sS])[^a-zA-Z]'),
        re.compile(r'[cC]hoice[^a-zA-Z]*([nN][oO])[^a-zA-Z\s]'),
        re.compile(r'[cC]hoice[^a-zA-Z]*([mM][aA][yY][bB][eE])[^a-zA-Z\s]'),

        re.compile(r'[cC]hoice is[^a-zA-Z]*([yY][eE][sS])[^a-zA-Z]'),
        re.compile(r'[cC]hoice is[^a-zA-Z]*([nN][oO])[^a-zA-Z]'),
        re.compile(r'[cC]hoice is[^a-zA-Z]*([mM][aA][yY][bB][eE])[^a-zA-Z]'),

        re.compile(r'[cC]hoice[^a-zA-Z]*([nN][oO])[^a-zA-Z]'),
        re.compile(r'[cC]hoice[^a-zA-Z]*([mM][aA][yY][bB][eE])[^a-zA-Z]'),
    ]

    # Loop through the patterns and search for a match
    for pattern in patterns:
        match = pattern.search(completion)
        if match:
            return helper(match.group(1).upper())

    patterns = [
        re.compile(r'^[^a-zA-Z]*([yY][eE][sS])$'),
        re.compile(r'^[^a-zA-Z]*([nN][oO])$'),
        re.compile(r'^[^a-zA-Z]*([mM][aA][yY][bB][eE])$'),
        re.compile(r'^[^a-zA-Z]*([nN][oO])[^a-zA-Z]'),
        re.compile(r'^[^a-zA-Z]*([mM][aA][yY][bB][eE])[^a-zA-Z]'),
    ]

    # Loop through the patterns and search for a match at the beginning of the string
    for pattern in patterns:
        match = pattern.match(completion)
        if match:
            return helper(match.group(1).upper())

    patterns = [
        re.compile(r' is[^a-zA-Z]*:+[^a-zA-Z]*([yY][eE][sS])[^a-zA-Z\s]'),
        re.compile(r' is[^a-zA-Z]*:+[^a-zA-Z]*([nN][oO])[^a-zA-Z\s]'),
        re.compile(r' is[^a-zA-Z]*:+[^a-zA-Z]*([mM][aA][yY][bB][eE])[^a-zA-Z\s]'),

        re.compile(r' is[^a-zA-Z]*:+\s+[^a-zA-Z]*([yY][eE][sS])[^a-zA-Z]'),
        re.compile(r' is[^a-zA-Z]*:+\s+[^a-zA-Z]*([nN][oO])[^a-zA-Z]'),
        re.compile(r' is[^a-zA-Z]*:+\s+[^a-zA-Z]*([mM][aA][yY][bB][eE])[^a-zA-Z]'),

        re.compile(r' is[^a-zA-Z]*([yY][eE][sS])[^a-zA-Z\s]'),

        re.compile(r':+[^a-zA-Z]*([yY][eE][sS])[^a-zA-Z\s]'),
        re.compile(r':+[^a-zA-Z]*([nN][oO])[^a-zA-Z\s]'),
        re.compile(r':+[^a-zA-Z]*([mM][aA][yY][bB][eE])[^a-zA-Z\s]'),

        re.compile(r':+\s+[^a-zA-Z]*([yY][eE][sS])[^a-zA-Z]'),
        re.compile(r':+\s+[^a-zA-Z]*([nN][oO])[^a-zA-Z]'),
        re.compile(r':+\s+[^a-zA-E]*([mM][aA][yY][bB][eE])[^a-zA-Z]'),

        re.compile(r' is[^a-zA-Z]*\s+[^a-zA-Z]*([yY][eE][sS])[^a-zA-Z]'),
        re.compile(r' is[^a-zA-Z]*([nN][oO])[^a-zA-Z\s]'),
        re.compile(r' is[^a-zA-Z]*([mM][aA][yY][bB][eE])[^a-zA-Z\s]'),

        re.compile(r' is[^a-zA-Z]*\s+[^a-zA-Z]*([nN][oO])[^a-zA-Z]'),
        re.compile(r' is[^a-zA-Z]*\s+[^a-zA-Z]*([mM][aA][yY][bB][eE])[^a-zA-Z]'),

        re.compile(r'[^a-zA-Z]([yY][eE][sS])[^a-zA-Z\s]'),
        re.compile(r'[^a-zA-Z]([mM][aA][yY][bB][eE])[^a-zA-Z\s]'),
        re.compile(r'[^a-zA-Z]([nN][oO])[^a-zA-Z\s]'),
        re.compile(r'[^a-zA-Z]([yY][eE][sS])$'),
        re.compile(r'[^a-zA-Z]([mM][aA][yY][bB][eE])$'),
        re.compile(r'[^a-zA-Z]([nN][oO])$'),

        re.compile(r'[^a-zA-Z]([yY][eE][sS])[^a-zA-Z]'),
        re.compile(r'[^a-zA-Z]([mM][aA][yY][bB][eE])[^a-zA-Z]'),
        re.compile(r'[^a-zA-Z]([nN][oO])[^a-zA-Z]'),

        re.compile(r'[^a-zA-Z]([yY][eE][sS])[^a-zA-Z]*'),
        re.compile(r'[^a-zA-Z]([mM][aA][yY][bB][eE])[^a-zA-Z]*'),

        re.compile(r'[^a-zA-Z]*([yY][eE][sS])[^a-zA-Z]*'),
        re.compile(r'[^a-zA-Z]*([mM][aA][yY][bB][eE])[^a-zA-Z]*'),
    ]

    for pattern in patterns:
        match = pattern.search(completion)
        if match:
            return helper(match.group(1).upper())

    patterns = [
        re.compile(r'([nN][oO])[a-zA-Z]+'),
        re.compile(r'[a-zA-Z]+([nN][oO])'),
    ]
    for pattern in patterns:
        match = pattern.search(completion)
        if match:
            return None

    extra_pattern_no = re.compile(r'[^a-zA-Z]*([nN][oO])[^a-zA-Z]*')
    match = extra_pattern_no.search(completion)
    if match:
        return helper(match.group(1).upper())

    # If no match is found, return None
    return None


def response_with_option(prompt, response):
    """
    Add the corresponding A/B/C/D option to the prompt
    :param prompt: the prompt to the model
    :param response: the model's response
    :return response: the model's response with the option
    """
    terms = {}
    option_range = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    for option in option_range:
        pattern = re.compile(rf'([{option}]\.\s.*?)\n')
        match = pattern.search(prompt)
        if match:
            terms[option] = match.group(1)

    for v in terms.values():
        answer = v[3:].lower()
        response = response.lower()
        response = response.replace(answer, v)

    return response
