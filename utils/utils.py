# coding=utf-8

import yaml


def stop_token_list():
    """
    The stop token list for vLLM engine
    Note: You can add more stop tokens
    if you are using other LLMs that have stop tokens
    """
    stop_tokens = [
        "Question:",
    ]

    return stop_tokens


def load_config(file_name):
    """
    Load parameters and path from the YAML file
    :param file_name: the name of the YAML file
    :return config: The configuration info
    """
    fopen = open(file_name)
    config = yaml.load(fopen, Loader=yaml.FullLoader)
    fopen.close()

    return config


def print_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    # Retrieve a list of all named parameters in the model
    model_parameters = list(model.named_parameters())

    # Calculate the total number of parameters using a generator expression
    all_param = sum(p.numel() for _, p in model_parameters)

    # Calculate the total number of trainable parameters using a generator expression
    # that filters parameters which require gradients
    trainable_params = sum(p.numel() for _, p in model_parameters if p.requires_grad)

    # Print out the number of trainable parameters, total parameters,
    # and the percentage of parameters that are trainable
    # The percentage is formatted to two decimal places
    print(
        f"Trainable params: {trainable_params:,} | "
        f"All params: {all_param:,} | "
        f"Trainable%: {100 * trainable_params / all_param:.2f}%"
    )


class CustomStream:
    """
    Save all the running logs
    """

    def __init__(self, filename, console_stream):
        self.filename = filename
        self.console_stream = console_stream

    def write(self, text):
        with open(self.filename, 'a') as file:
            file.write(text)
        self.console_stream.write(text)

    def flush(self):
        pass


def prompt_template(tokenizer=None, input=None):
    """
    Prompt Template
    Gemma: https://ai.google.dev/gemma/docs/formatting
    LLaMA 3: https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3/
    Mistrial: https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2
    :param tokenizer: the tokenizer
    :param input: User input question and content
    :return prompt: Prompt Template with query
    """
    messages = [
        {
            "role": "user", "content": input
        }
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        # We add the generation prompt ("<start_of_turn>model") in the Prompt during training
        # to be consistent with Model Inference
        add_generation_prompt=True
    )

    return prompt


def prompt_template_medAlpaca(input=None):
    """
    Prompt Template for the medAlpaca model
    Evaluation codes: https://github.com/kbressem/medAlpaca/blob/main/eval/eval_usmle.py#L157-L162
    Prompt example: https://github.com/kbressem/medAlpaca/blob/main/medalpaca/handler.py#L170-L191
    :param input: User input question and content
    :return prompt: Prompt Template with query
    """
    prompt = (f"Below is an instruction that describes a task, paired with an input that provides further context. "
              f"Write a response that appropriately completes the request.\n\n"
              f"### Instruction:\nAnswer this multiple choice question.\n\n"
              f"### Input:\n{input}\n\n"
              f"### Response:\nThe Answer to the question is:")

    return prompt


def prompt_template_MMedLM(input=None, language="English"):
    """
    Prompt Template for the MMed-Llama-3-8B-EnIns model
    Evaluation codes: https://github.com/MAGIC-AI4Med/MMedLM/blob/main/inference/inference.py#L86-L91
    Prompt example: https://github.com/MAGIC-AI4Med/MMedLM/blob/main/inference/inference.py#L10-L21
    :param input: User input question and content
    :param language: The language of the benchmark, as used in the instruction,
                     "You're a {language} doctor, kindly address ..."
    :return prompt: Prompt Template with query
    """
    # Pre-defined prompts references
    # Leaderboard: https://huggingface.co/spaces/openlifescienceai/open_medical_llm_leaderboard
    # Language Model Evaluation Harness: https://github.com/EleutherAI/lm-evaluation-harness
    # Special Notice: We use `"Directly answer the best option:"` instead of `Answer:`
    # to better guide LLMs to generate the best option and to easier extract the best option from the responses
    english_prompt = "Directly answer the best option:"
    english_prompt_pubmedqa = "Directly answer yes/no/maybe:"
    hindi_prompt = "सीधे सबसे अच्छे विकल्प के साथ जवाब दें:"
    french_prompt = "Répondez directement avec la meilleure option:"
    spanish_prompt = "Responde directamente con la mejor opción:"
    chinese_prompt = "直接回答最优选项:"

    # https://github.com/MAGIC-AI4Med/MMedLM/blob/main/inference/inference.py#L87
    instruction = (f"You're a {language} doctor, kindly address the medical queries according to the patient's "
                   f"account. Answer with the best option directly.")

    # https://github.com/MAGIC-AI4Med/MMedLM/blob/main/inference/inference.py#L88
    if english_prompt_pubmedqa in input:
        # For the PubMedQA Benchmark
        question = input
        question = question.replace(english_prompt_pubmedqa, "")
        # Special Notice: We don't use "A. yes B. no C. maybe"
        # to align with our answer extraction codes
        options = "yes, no, or maybe"
    else:
        question = input.split("\nA.")[0]
        options = "\nA." + input.split("\nA.")[1]
        
        options = options.replace(english_prompt, "")
        options = options.replace(hindi_prompt, "")
        options = options.replace(spanish_prompt, "")
        options = options.replace(french_prompt, "")
        options = options.replace(chinese_prompt, "")

    input = (f"###Question: {question} Which of the following is the best treatment for this patient? "
             f"###Options: {options}")

    # https://github.com/MAGIC-AI4Med/MMedLM/blob/main/inference/inference.py#L11-L14
    prompt = (f"Below is an instruction that describes a task, paired with an input that provides further context. "
              f"Write a response that appropriately completes the request.\n\n"
              f"### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:")

    return prompt


def response_template(model_name, output):
    """
    Response Template
    Gemma: https://ai.google.dev/gemma/docs/formatting
    LLaMA 3: https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3/
    Mistrial: https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2
    :param output: Answer response
    :return: Response Template with response
    """
    if "gemma" in model_name:
        response = f"{output}<end_of_turn><eos>"
    elif "mistralai" in model_name:
        response = f"{output}</s>"
    elif "llama" in model_name:
        response = f"{output}<|eot_id|><|end_of_text|>"
    else:
        response = f"{output}"

    return response
