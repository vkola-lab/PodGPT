# coding=utf-8
#
# GNU Affero General Public License v3.0 License
#
# PodGPT: An Audio-augmented Large Language Model for Research and Education
# Copyright (C) 2024 Kolachalama Laboratory at Boston University
#
# LICENSE OF THE FOLLOWING MODELS
#
# LLAMA 3 COMMUNITY LICENSE AGREEMENT
# https://llama.meta.com/llama3/license/
#
# LLAMA 3.1 COMMUNITY LICENSE AGREEMENT
# https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/LICENSE
#
# Mistral LICENSE
# https://www.apache.org/licenses/LICENSE-2.0
#
# GEMMA TERMS OF USE
# https://ai.google.dev/gemma/terms

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from peft import LoraConfig, TaskType
from trl import SFTTrainer

from utils.utils import print_parameters


def model_loader(config):
    """
    Initialize model
    :param config: the YAML configuration file
    :return model: the pre-trained model
    :return tokenizer: the tokenizer of the pre-trained model
    """
    model_name = config.get("model_name")
    device_map = config.get("device_map")
    train_max_len = config.get("train_max_len")
    hf_read_token = config.get("hf_read_token")
    lora_r = config.get("lora_r")
    lora_alpha = config.get("lora_alpha")
    lora_dropout = config.get("lora_dropout")

    # Load the base model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        use_auth_token=hf_read_token,
        device_map=device_map,
        torch_dtype=torch.bfloat16,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        ############################################################
        # IMPORTANT - Please note that this is for model training
        # We DON'T need this during performance evaluation
        model_max_length=train_max_len,
        padding='longest',
        padding_side="right",
        truncation=True,
        ############################################################
        return_tensors="pt",
        # use_fast=False,
        use_auth_token=hf_read_token,
        device_map=device_map,
    )
    if "llama" in model_name.lower() or "mistralai" in model_name.lower():
        tokenizer.pad_token = tokenizer.eos_token

    # LoRA configuration
    # `google/gemma-2b-it`:
    # https://huggingface.co/google/gemma-2b-it/blob/main/model.safetensors.index.json
    # `google/gemma-7b-it`:
    # https://huggingface.co/google/gemma-7b-it/blob/main/model.safetensors.index.json
    # `meta-llama/Meta-Llama-3-8B-Instruct`:
    # https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct/blob/main/model.safetensors.index.json
    # `meta-llama/Meta-Llama-3-70B-Instruct`:
    # https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct/blob/main/model.safetensors.index.json
    # `mistralai/Mistral-7B-Instruct-v0.1`:
    # https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1/blob/main/model.safetensors.index.json
    # `mistralai/Mistral-7B-Instruct-v0.2`:
    # https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2/blob/main/model.safetensors.index.json
    # `mistralai/Mistral-7B-Instruct-v0.3`:
    # https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3/blob/main/model.safetensors.index.json
    # `mistralai/Mixtral-8x7B-Instruct-v0.1`:
    # https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1/blob/main/model.safetensors.index.json
    # `mistralai/Mixtral-8x22B-Instruct-v0.1`:
    # https://huggingface.co/mistralai/Mixtral-8x22B-Instruct-v0.1/blob/main/model.safetensors.index.json
    if "Mixtral-8x" in model_name:
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            # Please note that the current vLLM is not supporting 
            # the modules "w1", "w2", "w3", and "gate" at this point (June 20, 2024)
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj"
            ],
            task_type=TaskType.CAUSAL_LM,
        )
    else:
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            task_type=TaskType.CAUSAL_LM,
        )
    model.add_adapter(lora_config, adapter_name="adapter")
    # If you wanna load the existing LoRA adapter, please comment the above `model.add_adapter`
    # And use the `model.load_adapter`
    # model.load_adapter("./save_folder/YOUR_LORA_CHECKPOINT_FOLDER_PATH")
    model.enable_adapters()
    print_parameters(model=model)

    return model, tokenizer


def trainer_loader(config, model, tokenizer, dataset, num_train_epochs):
    """
    Load training pipeline
    :param config: the configurations
    :param model: the pre-trained model
    :param tokenizer: the tokenizer of the pre-trained model
    :param dataset: the training dataset
    :param num_train_epochs: the number of training epochs
    :return trainer: SFTTrainer
    """
    train_batch_size = config.get("train_batch_size")
    gradient_accumulation_steps = config.get("gradient_accumulation_steps")
    optim = config.get("optim")
    logging_steps = config.get("logging_steps")
    learning_rate = config.get("learning_rate")
    weight_decay = config.get("weight_decay")
    warmup_ratio = config.get("warmup_ratio")
    lr_scheduler_type = config.get("lr_scheduler_type")
    fp16 = config.get("fp16")
    bf16 = config.get("bf16")
    save_dir = config.get("save_dir")
    train_max_len = config.get("train_max_len")
    gradient_checkpointing = config.get("gradient_checkpointing")
    log_save_platform = config.get("log_save_platform")
    save_strategy = config.get("save_strategy")
    save_only_model = config.get("save_only_model")
    save_total_limit = config.get("save_total_limit")

    # Different models have different `eos_token_id`
    # By default, the SFTTrainer has set the
    # append_concat_token = True
    # add_special_tokens = True
    # eos_token_id=tokenizer.eos_token_id
    # So, we don't have to pass a specific `eos_token_id` to the SFTTrainer
    # https://github.com/huggingface/trl/blob/main/trl/trainer/sft_trainer.py#L586-L614

    # https://huggingface.co/docs/trl/en/sft_trainer#trl.trainer.ConstantLengthDataset
    # https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct/blob/main/config.json#L8
    # https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct/blob/main/generation_config.json#L3
    # eos_token_id=128009,  # meta-llama/Meta-Llama-3-70B-Instruct

    # https://huggingface.co/docs/trl/en/sft_trainer#trl.trainer.ConstantLengthDataset
    # https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1/blob/main/config.json#L7
    # https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1/blob/main/generation_config.json#L4
    # https://huggingface.co/mistralai/Mixtral-8x22B-Instruct-v0.1/blob/main/config.json#L7
    # https://huggingface.co/mistralai/Mixtral-8x22B-Instruct-v0.1/blob/main/generation_config.json#L4
    # eos_token_id=2,  # mistralai/Mixtral-8x7B-Instruct-v0.1, mistralai/Mixtral-8x22B-Instruct-v0.1

    # Set training parameters
    arguments = TrainingArguments(
        output_dir=save_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=gradient_checkpointing,
        optim=optim,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        lr_scheduler_type=lr_scheduler_type,
        weight_decay=weight_decay,
        fp16=fp16,
        bf16=bf16,
        warmup_ratio=warmup_ratio,
        report_to=log_save_platform,
        save_strategy=save_strategy,
        save_only_model=save_only_model,
        save_total_limit=save_total_limit,
    )

    # Set supervised fine-tuning parameters
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        dataset_text_field="text",
        train_dataset=dataset,
        max_seq_length=train_max_len,
        args=arguments,
        packing=True,
    )

    return trainer
