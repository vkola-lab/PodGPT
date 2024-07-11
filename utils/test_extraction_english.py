#!/usr/bin/env python
# coding=utf-8
#
# MIT License
#
# Copyright (c) 2024 Kolachalama Lab at Boston University
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

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
