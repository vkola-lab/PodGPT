# coding=utf-8
#
# GNU Affero General Public License v3.0 License
#
# PodGPT: An Audio-augmented Large Language Model for Research and Education
# Copyright (C) 2024 Kolachalama Laboratory at Boston University

import os

from transformers import AutoTokenizer, TrainingArguments
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig, get_gptq_peft_model
from auto_gptq.utils.peft_utils import GPTQLoraConfig
from peft import TaskType
from trl import SFTTrainer

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def model_initializer(config):
    """
    Initialize model
    :param config: the YAML configuration file
    :return model: the pre-trained model
    :return tokenizer: the tokenizer of the pre-trained model
    """
    model_name = config.get("model_name")
    train_max_len = config.get("train_max_len")
    hf_read_token = config.get("hf_read_token")
    lora_r = config.get("lora_r")
    lora_alpha = config.get("lora_alpha")
    lora_dropout = config.get("lora_dropout")
    device_map = config.get("device_map")

    model = AutoGPTQForCausalLM.from_quantized(
        model_name,
        use_safetensors=True,
        use_triton=True,
        warmup_triton=False,
        trainable=True,
        quantize_config=None,
        # https://github.com/AutoGPTQ/AutoGPTQ/issues/210#issuecomment-1694269325
        # https://huggingface.co/TheBloke/Llama-2-70B-Chat-GPTQ/discussions/24
        # https://github.com/AutoGPTQ/AutoGPTQ/pull/237/commits/11afc47f7f9ab1671df4a81a9e91d6153d5d958e
        inject_fused_attention=False,
        inject_fused_mlp=False,
    )
    model.warmup_triton()

    # https://gist.github.com/eusip/de8fadb761741b56d5d9a6232bf979ed#file-oasst-pythia-12b-05-03-2023-py-L68-L87
    # NOTE: https://github.com/lvwerra/trl/blob/a2749d9e0c96198486b788875eda3b325f76a5c8/examples/sentiment/scripts/gpt-neox-20b_peft/gpt-neo-20b_sentiment_peft.py#L181
    for param in model.parameters():
        # freeze base model's layers
        param.requires_grad = False

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)

        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

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
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # LoRA configurations
    peft_config = GPTQLoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
    )
    model = get_gptq_peft_model(model, peft_config=peft_config, auto_find_all_linears=True, train_mode=True)
    model.print_trainable_parameters()

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
    save_dir = config.get("save_dir")
    train_max_len = config.get("train_max_len")
    gradient_checkpointing = config.get("gradient_checkpointing")
    log_save_platform = config.get("log_save_platform")
    save_strategy = config.get("save_strategy")
    save_steps = config.get("save_steps")
    save_only_model = config.get("save_only_model")
    save_total_limit = config.get("save_total_limit")

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
        warmup_ratio=warmup_ratio,
        report_to=log_save_platform,
        remove_unused_columns=False,
        # Model saving settings
        save_strategy=save_strategy,
        save_steps=save_steps,
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
