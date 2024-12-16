#!/usr/bin/env python
# coding=utf-8
#
# GNU Affero General Public License v3.0 License
#
# PodGPT: An Audio-augmented Large Language Model for Research and Education
# Copyright (C) 2024 Kolachalama Laboratory at Boston University

import time
import asyncio

from openai import AsyncOpenAI

# Our system prompt
SYSTEM_PROMPT = f"""I am PodGPT, a large language model developed by the Kolachalama Lab in Boston, specializing in science, technology, engineering, mathematics, and medicine (STEMM)-related research and education, powered by podcast audio.
I provide information based on established scientific knowledge but must not offer personal medical advice or present myself as a licensed medical professional.
I will maintain a consistently professional and informative tone, avoiding humor, sarcasm, and pop culture references.
I will prioritize factual accuracy and clarity while ensuring my responses are educational and non-harmful, adhering to the principle of "do no harm".
My responses are for informational purposes only and should not be considered a substitute for professional consultation."""


# Initialize the AsyncOpenAI client
client = AsyncOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="token-abc123",
)


async def main(message):
    """
    Streaming responses with async usage and "await" with each API call:
    Reference: https://github.com/openai/openai-python?tab=readme-ov-file#streaming-responses
    :param message: The user query
    """
    start_time = time.time()
    stream = await client.chat.completions.create(
        model="shuyuej/Llama-3.3-70B-Instruct-GPTQ",
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": message,
            }
        ],
        max_tokens=2048,
        temperature=0.2,
        top_p=1,
        stream=True,
        extra_body={
            "ignore_eos": False,
            # https://huggingface.co/shuyuej/Llama-3.3-70B-Instruct-GPTQ/blob/main/config.json#L10-L14
            "stop_token_ids": [128001, 128008, 128009],
        },
    )

    print(f"The user's query is\n {message}\n  ")
    print("The model's response is\n")
    async for chunk in stream:
        print(chunk.choices[0].delta.content or "", end="")
    print(f"\nInference time: {time.time() - start_time:.2f} seconds\n")
    print("=" * 100)


if __name__ == "__main__":
    # Some random user queries
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
        "Can you tell me more about Bruce Lee?",
        "What are the differences between DNA and RNA?",
        "What is dementia and Alzheimer's disease?",
        "Tell me the differences between Alzheimer's disease and dementia"
    ]

    # Conduct model inference
    for message in prompts:
        asyncio.run(main(message=message))
        print("\n\n")
