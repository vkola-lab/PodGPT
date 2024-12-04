#!/usr/bin/env python
# coding=utf-8
#
# GNU Affero General Public License v3.0 License
#
# PodGPT: An Audio-augmented Large Language Model for Research and Education
# Copyright (C) 2024 Kolachalama Laboratory at Boston University
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
    Custom Dataset class to load audio files from a directory.

    :param file_dir: Directory containing audio files.
    """
    def __init__(self, file_dir):
        """
        Initialize the dataset with a directory containing audio files.

        :param file_dir: Path to the directory containing audio files.
        """
        self.file_dir = file_dir
        self.audio_files = [
            os.path.join(self.file_dir, filename) for filename in os.listdir(self.file_dir)
            if filename.endswith((".mp3", ".m4a", ".wav"))
        ]

    def __len__(self):
        """
        Get the total number of audio files in the dataset.

        :return: Number of audio files in the directory.
        """
        return len(self.audio_files)

    def __getitem__(self, idx):
        """
        Get the file path of the audio file at the specified index.

        :param idx: Index of the audio file in the dataset.
        :return: File path of the audio file.
        """
        filepath = self.audio_files[idx]
        return filepath


def main(loader, save_dir):
    """
    Transcribe audio files to text and save them to a specified directory.

    :param loader: DataLoader object for batching audio file paths.
    :param save_dir: Directory where transcriptions will be saved.
    """
    for batch in loader:
        try:
            # Transcribe the audio files in the batch using the pipeline.
            results = pipe(batch, generate_kwargs={"language": "english"})

            # Iterate over each transcription result in the batch.
            for index, result in enumerate(results):
                text = result["text"]  # Extract the transcription text.

                # Get the corresponding file path for the transcription.
                filepath = batch[index]

                # Save the transcription to a text file in the specified directory.
                filename = os.path.basename(filepath)
                text_filename = os.path.splitext(filename)[0] + ".txt"
                with open(os.path.join(save_dir, text_filename), "w") as text_file:
                    text_file.write(text)

                print(f"Transcribed '{filename}' successfully.")
        except Exception as e:
            # Log any errors encountered during transcription.
            print(f"Error transcribing batch: {str(e)}")


if __name__ == "__main__":
    # Determine the computing device: use CUDA if available, otherwise CPU.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Current PyTorch device is set to", device)

    # Use float16 if CUDA is available, otherwise float32.
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # Load the Whisper model for speech-to-text transcription.
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        "openai/whisper-large-v3",
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=False,
        use_safetensors=True
    )
    model.to(device)  # Move the model to the selected device.

    # Load the processor (tokenizer and feature extractor) for the model.
    processor = AutoProcessor.from_pretrained("openai/whisper-large-v3")

    # Create a transcription pipeline with specified parameters.
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=384,  # Limit tokens per transcription.
        chunk_length_s=30,  # Audio chunk length for processing.
        stride_length_s=5,  # Overlap between audio chunks.
        batch_size=96,  # Batch size for the pipeline.
        return_timestamps=False,  # Disable timestamps in the transcription output.
        torch_dtype=torch_dtype,
        device=device,  # Set the device for computation.
    )

    # Directory containing the audio files to be transcribed.
    file_dir = "The_New_England_Journal_of_Medicine_Podcasts"

    # Directory to save transcriptions. Create the directory if it doesn't exist.
    save_dir = os.path.join(file_dir, "saved_transcripts")
    os.makedirs(save_dir, exist_ok=True)

    # Create a dataset and a DataLoader for batching audio file paths.
    dataset = AudioDataset(file_dir)
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=8)

    # Transcribe audio files and save the transcriptions to the directory.
    main(loader=loader, save_dir=save_dir)
