#!/usr/bin/env python
# coding=utf-8
#
# GNU Affero General Public License v3.0 License
#
# MedPodGPT: A multilingual audio-augmented large language model for medical research and education
# Copyright (C) 2024 Kolachalama Laboratory at Boston University

from answer_utils import extract_answer

completion = ("The correct next action for the resident to take is (B): Disclose the error to the patient "
              "and put it in the operative report.\n\nThe patient has a right to be informed about "
              "any significant medical or surgical complications that may occur during a procedure. "
              "The fact that the resident has cut a flexor tendon is a significant medical or surgical "
              "complication that should be disclosed to the patient. Therefore, the correct next action "
              "for the resident to take is to disclose the error to the patient and put it in the operative report.")
completion = " C"
completion = (" The most appropriate pharmacotherapy for this patient would be a A. cholinesterase inhibitor, "
              "such as donepezil, galantamine, or rivastigmine. "
              "These medications are first-line treatment for Alzheimer's dementia and other dementias, "
              "as they can improve cognition and daily functioning. The symptoms described in the question, "
              "such as memory loss, repetitive behavior, and forgetfulness, are consistent with dementia. "
              "The other options, such as a beta-adrenergic agonist, dopamine agonist, or prednisone, "
              "are not indicated for the treatment of dementia.")
result = extract_answer(completion=completion)
print(result)
