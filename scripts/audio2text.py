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
#
# LICENSE OF THE FOLLOWING MODELS
# openai/whisper-large-v3
# https://huggingface.co/openai/whisper-large-v3
# https://github.com/openai/whisper
#
# MIT License
# https://github.com/openai/whisper/blob/main/LICENSE

import os

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


class AudioDataset(Dataset):
    """
    Custom Dataset class to load audio files
    """

    def __init__(self, file_dir):
        self.file_dir = file_dir
        self.audio_files = [
            os.path.join(self.file_dir, filename) for filename in os.listdir(self.file_dir)
            if (filename.endswith(".mp3") or filename.endswith(".m4a"))
        ]

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        filepath = self.audio_files[idx]
        return filepath


def transcribe_audio(loader, save_dir):
    """
    Transcribe audio files to text
    :param loader: the data loader
    :param save_dir: the save directory
    """
    for batch in loader:
        try:
            # Transcribe the MP3 files in the batch
            results = pipe(batch, generate_kwargs={"language": "english"})

            # Iterate over results of each file in the batch
            for index, result in enumerate(results):
                text = result["text"]

                # Get the file path from the loader
                filepath = batch[index]

                # Save the transcribed text to a file in the saved_transcripts folder
                filename = os.path.basename(filepath)
                text_filename = os.path.splitext(filename)[0] + ".txt"
                with open(os.path.join(save_dir, text_filename), "w") as text_file:
                    text_file.write(text)

                print(f"Transcribed '{filename}' successfully.")
        except Exception as e:
            print(f"Error transcribing batch: {str(e)}")


if __name__ == "__main__":
    # Check if CUDA is available, set device accordingly
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Current PyTorch device is set to", device)
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # Load the Whisper model
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        "openai/whisper-large-v3",
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=False,
        use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(
        "openai/whisper-large-v3"
    )

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=384,
        chunk_length_s=30,
        stride_length_s=5,
        batch_size=96,
        return_timestamps=False,
        torch_dtype=torch_dtype,
        device=device,
    )

    # Directory containing the audio files
    file_dir = "Annals_On_Call"

    # Create a directory to save transcripts if it doesn't exist
    save_dir = os.path.join(file_dir, "saved_transcripts")
    os.makedirs(save_dir, exist_ok=True)

    # Create dataset and DataLoader
    dataset = AudioDataset(file_dir)
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=8)

    # Transcribe audio files in the directory
    transcribe_audio(pipeline=pipe, loader=loader, save_dir=save_dir)