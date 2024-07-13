#!/usr/bin/env python
# coding=utf-8
#
# GNU Affero General Public License v3.0 License
#
# MedPodGPT: A multilingual audio-augmented large language model for medical research and education
# Copyright (C) 2024 Kolachalama Laboratory at Boston University

import json

# Path to your JSON file
data_path = [
    'english_transcripts.json',
    'spanish_transcripts.json',
    'french_transcripts.json'
]

data = []
num = 0
for path in data_path:
    with open(path, "r", encoding="utf8") as f:
        content = json.load(f)
        for item in content:
            text = item['text']
            data.append({"text": text})
            num += 1

# Write data to a JSON file
file_name = "multilingual_transcripts.json"
with open(file_name, 'w', encoding='utf-8') as json_file:
    json.dump(data, json_file, ensure_ascii=False, indent=4)

print(f"Modified data saved to {file_name}")
print("The total number of samples: ", num)
