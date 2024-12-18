# coding=utf-8
#
# GNU Affero General Public License v3.0 License
#
# PodGPT: An Audio-augmented Large Language Model for Research and Education
# Copyright (C) 2024 Kolachalama Laboratory at Boston University
#
# BENCHMARKS AND CREDITS
#
# English Benchmarks
# MedQA-USMLE
# We conducted performance evaluation on the MedQA-USMLE Benchmark
# Notice: We use the 4-option version of the dataset
# Paper: What Disease does this Patient Have? A Large-scale Open Domain Question Answering Dataset from Medical Exams
# Link: https://arxiv.org/abs/2009.13081
# Database and Codes: https://github.com/jind11/MedQA
#
# MedMCQA
# We conducted performance evaluation on the MedMCQA Benchmark
# Paper: MedMCQA : A Large-scale Multi-Subject Multi-Choice Dataset for Medical domain Question Answering
# Link: https://arxiv.org/abs/2203.14371
# Homepage: https://medmcqa.github.io/
# Database and Codes: https://github.com/MedMCQA/MedMCQA
#
# PubMedQA
# We conducted performance evaluation on the PubMedQA Benchmark
# Paper: PubMedQA: A Dataset for Biomedical Research Question Answering
# Link: https://arxiv.org/abs/1909.06146
# Homepage: https://pubmedqa.github.io/
# Database and Codes: https://github.com/pubmedqa/pubmedqa
#
# USMLE - STEP 1, STEP 2, STEP 3
# Credits:
# Lindsey Claus, Vascular Surgery, University of Pennsylvania (laclaus@bu.edu)
# Divya Veerapaneni, Neurology, UT Southwestern Medical Center (divyavee@bu.edu)
# Meagan Lauber, Ph.D. in Neuroscience, Boston University (mlauber@bu.edu)
#
# MedExpQA English
# We conducted performance evaluation on the MedExpQA (English) Benchmark
# Paper: MedExpQA: Multilingual Benchmarking of Large Language Models for Medical Question Answering
# Link: https://arxiv.org/abs/2404.05590
# Database and Codes: https://huggingface.co/datasets/HiTZ/MedExpQA
#
# MMLU Medicine Subset
# We conducted performance evaluation on medicine-related 6 databases
# anatomy_test, clinical_knowledge_test, college_biology_test
# college_medicine_test, medical_genetics_test, professional_medicine_test
# Paper: Measuring Massive Multitask Language Understanding
# Link: https://arxiv.org/abs/2009.03300
# Database and Codes: https://www.kaggle.com/datasets/lizhecheng/mmlu-dataset
#
# MMLU STEM Subsets
# We conducted performance evaluation on the STEM subsets, including
# Physics (4)
# Database: astronomy, college_physics, conceptual_physics, high_school_physics
# Chemistry (2)
# Database: college_chemistry, high_school_chemistry
# Biology (2)
# Database: college_biology, high_school_biology
# Computer Science (4)
# Database: college_computer_science, computer_security, high_school_computer_science, machine_learning
# Math (5)
# Database: abstract_algebra, college_mathematics, elementary_mathematics,
# high_school_mathematics, high_school_statistics
# Engineering (1)
# Database: electrical_engineering
# Link: https://arxiv.org/abs/2009.03300
# Category Link: https://github.com/hendrycks/test/blob/master/categories.py
# Database and Codes: https://www.kaggle.com/datasets/lizhecheng/mmlu-dataset
#
# Chinese Mandarin Benchmarks
# MedQA-MCMLE
# We conducted performance evaluation on the MedQA-MCMLE Benchmark
# Paper: What Disease does this Patient Have? A Large-scale Open Domain Question Answering Dataset from Medical Exams
# Link: https://arxiv.org/abs/2009.13081
# Database and Codes: https://github.com/jind11/MedQA
#
# CMMLU
# We conducted performance evaluation on medicine-related 7 databases
# anatomy, clinical_knowledge, college_medicine
# genetics, nutrition, traditional_chinese_medicine, virology
# Paper: CMMLU: Measuring massive multitask language understanding in Chinese
# Link: https://arxiv.org/abs/2306.09212
# Database and Codes: https://huggingface.co/datasets/haonan-li/cmmlu
#
# CMMLU STEM Subsets
# We conducted performance evaluation on the STEM subsets, including
# Physics (3)
# Database: astronomy, conceptual_physics, high_school_physics,
# Chemistry (1)
# Database: high_school_chemistry
# Biology (1)
# Database: high_school_biology
# Note: Due to the overlap with the CMMLU Medicine Subset, we only used high_school_biology here.
# and remove the `anatomy, genetics, virology`
# Computer Science (3)
# Database: computer_science, machine_learning, computer_security
# Math (4)
# Database: college_actuarial_science, college_mathematics, elementary_mathematics, high_school_mathematics,
# Engineering (2)
# Database: college_engineering_hydrology, electrical_engineering,
# Statistics (1)
# Database: college_medical_statistics,
# Link: https://arxiv.org/abs/2306.09212
# Category: https://github.com/haonan-li/CMMLU/blob/master/src/categories.py
# Database and Codes: https://huggingface.co/datasets/haonan-li/cmmlu
#
# French Benchmarks
# FrenchMedMCQA
# We conducted performance evaluation on the FrenchMedMCQA Benchmark
# Notice: We only use the single-answer subset of this benchmark
# Paper: FrenchMedMCQA: A French Multiple-Choice Question Answering Dataset for Medical domain
# Link: https://arxiv.org/abs/2304.04280
# Database and Codes: https://github.com/qanastek/FrenchMedMCQA
#
# MMLU French
# We conducted performance evaluation on medicine-related 6 databases
# Clinical knowledge, Medical genetics, Anatomy,
# Professional medicine, College biology, and College medicine.
# Paper: Apollo: An Lightweight Multilingual Medical LLM towards Democratizing Medical AI to 6B People
# Link: https://arxiv.org/abs/2403.03640
# Database and Codes: https://huggingface.co/datasets/FreedomIntelligence/MMLU_French
#
# MedExpQA French
# We conducted performance evaluation on the MedExpQA (French) Benchmark
# Paper: MedExpQA: Multilingual Benchmarking of Large Language Models for Medical Question Answering
# Link: https://arxiv.org/abs/2404.05590
# Database and Codes: https://huggingface.co/datasets/HiTZ/MedExpQA
#
# MMLU French STEM Subsets
# We conducted performance evaluation on the STEM subsets, including
# Physics (4)
# Database: astronomy, college_physics, conceptual_physics, high_school_physics
# Chemistry (2)
# Database: college_chemistry, high_school_chemistry
# Biology (2)
# Database: college_biology, high_school_biology
# Computer Science (4)
# Database: college_computer_science, computer_security, high_school_computer_science, machine_learning
# Math (5)
# Database: abstract_algebra, college_mathematics, elementary_mathematics,
# high_school_mathematics, high_school_statistics
# Engineering (1)
# Database: electrical_engineering
# Link: https://arxiv.org/abs/2009.03300
# Category Link: https://github.com/hendrycks/test/blob/master/categories.py
# Database and Codes: https://huggingface.co/datasets/FreedomIntelligence/MMLU_French
#
# Spanish Benchmarks
# HEAD-QA
# We conducted performance evaluation on the HEAD-QA Benchmark
# Paper: HEAD-QA: A Healthcare Dataset for Complex Reasoning
# Link: https://arxiv.org/abs/1906.04701
# Homepage: https://aghie.github.io/head-qa/
# Database and Codes: https://github.com/aghie/head-qa
#
# MedExpQA Spanish
# We conducted performance evaluation on the MedExpQA (Spanish) Benchmark
# Paper: MedExpQA: Multilingual Benchmarking of Large Language Models for Medical Question Answering
# Link: https://arxiv.org/abs/2404.05590
# Database and Codes: https://huggingface.co/datasets/HiTZ/MedExpQA
#
# MMLU Spanish
# We conducted performance evaluation on medicine-related 6 databases
# Clinical knowledge, Medical genetics, Anatomy,
# Professional medicine, College biology, and College medicine.
# Paper: Apollo: An Lightweight Multilingual Medical LLM towards Democratizing Medical AI to 6B People
# Link: https://arxiv.org/abs/2403.03640
# Database and Codes: https://huggingface.co/datasets/FreedomIntelligence/MMLU_Spanish
#
# MMLU Spanish STEM Subsets
# We conducted performance evaluation on the STEM subsets, including
# Physics (4)
# Database: astronomy, college_physics, conceptual_physics, high_school_physics
# Chemistry (2)
# Database: college_chemistry, high_school_chemistry
# Biology (2)
# Database: college_biology, high_school_biology
# Computer Science (4)
# Database: college_computer_science, computer_security, high_school_computer_science, machine_learning
# Math (5)
# Database: abstract_algebra, college_mathematics, elementary_mathematics
# high_school_mathematics, high_school_statistics
# Engineering (1)
# Database: electrical_engineering
# Link: https://arxiv.org/abs/2009.03300
# Category Link: https://github.com/hendrycks/test/blob/master/categories.py
# Database and Codes: https://huggingface.co/datasets/FreedomIntelligence/MMLU_Spanish
#
# Hindi Benchmarks
# MMLU Hindi
# We conducted performance evaluation on medicine-related 6 databases
# Clinical knowledge, Medical genetics, Anatomy,
# Professional medicine, College biology, and College medicine
# Paper: Apollo: An Lightweight Multilingual Medical LLM towards Democratizing Medical AI to 6B People
# Link: https://arxiv.org/abs/2403.03640
# Database and Codes: https://huggingface.co/datasets/FreedomIntelligence/MMLU_Hindi
#
# MMLU Hindi STEM Subsets
# We conducted performance evaluation on the STEM subsets, including
# Physics (4)
# Database: astronomy, college_physics, conceptual_physics, high_school_physics
# Chemistry (2)
# Database: college_chemistry, high_school_chemistry
# Biology (2)
# Database: college_biology, high_school_biology
# Computer Science (4)
# Database: college_computer_science, computer_security, high_school_computer_science, machine_learning
# Math (5)
# Database: abstract_algebra, college_mathematics, elementary_mathematics,
# high_school_mathematics, high_school_statistics
# Engineering (1)
# Database: electrical_engineering
# Link: https://arxiv.org/abs/2009.03300
# Category Link: https://github.com/hendrycks/test/blob/master/categories.py
# Database and Codes: https://huggingface.co/datasets/FreedomIntelligence/MMLU_Hindi

import os

from transformers import AutoTokenizer

from utils.eval_utils import *
from utils.utils import download_pretrained_model


def evaluation(config, mode="small", eval_pretrain=False, checkpoint_id=None):
    """
    Conduct model inference, get the model's responses, and evaluate performance
    :param config: the configuration file
    :param mode: one of "small", "large", or "chatgpt"
        "small": Evaluate the smaller models
        "large": Evaluate the larger models (with the LoRA Adapter)
        "chatgpt": Evaluate the OpenAI ChatGPT
    :param eval_pretrain: is this evaluating the pre-trained model?
    :param checkpoint_id: checkpoint ID
    """
    # Retrieve the pathes of needed hyperparameters
    model_name = config.get("model_name")
    result_dir = config.get("result_dir")

    # Evaluating the larger models
    if mode == "large":
        # Initialize vLLM engine
        if eval_pretrain:
            # Set checkpoint path
            checkpoint_path = "Original-Model"  # File name for the pre-trained model
        else:
            # Set checkpoint path
            checkpoint_path = "checkpoint-" + str(checkpoint_id)
            config['lora_path'] = "./save_folder/" + checkpoint_path

        # Load the tokenizer
        # Since there is no prompt template for Mixtral MoE, we will use `Mistral-7B-Instruct-v0.3` prompt template
        if "Mixtral-8x" in model_name:
            tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Evaluating the smaller models
    elif mode == "small":
        print("Start to download the original pre-trained model and tokenizer!")
        download_pretrained_model(config=config)

        if eval_pretrain:
            # Set checkpoint path
            checkpoint_path = "Original-Model"  # File name for the pre-trained model
        else:
            # Set checkpoint path
            checkpoint_path = "checkpoint-" + str(checkpoint_id)
            config['save_dir'] = "./save_folder/" + checkpoint_path

        # Assign LLM as None for now and load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Evaluating the ChatGPT
    elif mode == "chatgpt":
        # Set checkpoint path
        checkpoint_path = model_name
        tokenizer = None

    # Print the only three evaluation modes.
    else:
        checkpoint_path = None
        tokenizer = None
        print("The evaluation mode should be one of small, large, or chatgpt.")

    # English Benchmarks
    english_medicine_medqa = config.get("english_medicine_medqa")
    english_medicine_pubmedqa = config.get("english_medicine_pubmedqa")
    english_medicine_medmcqa = config.get("english_medicine_medmcqa")
    english_medicine_usmle_step1 = config.get("english_medicine_usmle_step1")
    english_medicine_usmle_step2 = config.get("english_medicine_usmle_step2")
    english_medicine_usmle_step3 = config.get("english_medicine_usmle_step3")
    english_medicine_usmle_ethics = config.get("english_medicine_usmle_ethics")
    english_medicine_medexpqa = config.get("english_medicine_medexpqa")
    # MMLU Medicine
    english_mmlu_medicine_anatomy = config.get("english_mmlu_medicine_anatomy")
    english_mmlu_medicine_clinical_knowledge = config.get("english_mmlu_medicine_clinical_knowledge")
    english_mmlu_medicine_college_biology = config.get("english_mmlu_medicine_college_biology")
    english_mmlu_medicine_college_medicine = config.get("english_mmlu_medicine_college_medicine")
    english_mmlu_medicine_medical_genetics = config.get("english_mmlu_medicine_medical_genetics")
    english_mmlu_medicine_professional_medicine = config.get("english_mmlu_medicine_professional_medicine")
    # MMLU Physics
    english_mmlu_physics_astronomy = config.get("english_mmlu_physics_astronomy")
    english_mmlu_physics_college_physics = config.get("english_mmlu_physics_college_physics")
    english_mmlu_physics_conceptual_physics = config.get("english_mmlu_physics_conceptual_physics")
    english_mmlu_physics_high_school_physics = config.get("english_mmlu_physics_high_school_physics")
    # MMLU Chemistry
    english_mmlu_chemistry_college_chemistry = config.get("english_mmlu_chemistry_college_chemistry")
    english_mmlu_chemistry_high_school_chemistry = config.get("english_mmlu_chemistry_high_school_chemistry")
    # MMLU Biology
    english_mmlu_biology_college_biology = config.get("english_mmlu_biology_college_biology")
    english_mmlu_biology_high_school_biology = config.get("english_mmlu_biology_high_school_biology")
    # MMLU Computer Science
    english_mmlu_cs_college_computer_science = config.get("english_mmlu_cs_college_computer_science")
    english_mmlu_cs_computer_security = config.get("english_mmlu_cs_computer_security")
    english_mmlu_cs_high_school_computer_science = config.get("english_mmlu_cs_high_school_computer_science")
    english_mmlu_cs_machine_learning = config.get("english_mmlu_cs_machine_learning")
    # MMLU Math
    english_mmlu_math_abstract_algebra = config.get("english_mmlu_math_abstract_algebra")
    english_mmlu_math_college_mathematics = config.get("english_mmlu_math_college_mathematics")
    english_mmlu_math_elementary_mathematics = config.get("english_mmlu_math_elementary_mathematics")
    english_mmlu_math_high_school_mathematics = config.get("english_mmlu_math_high_school_mathematics")
    english_mmlu_math_high_school_statistics = config.get("english_mmlu_math_high_school_statistics")
    # MMLU Engineering
    english_mmlu_engineering_electrical_engineering = config.get("english_mmlu_engineering_electrical_engineering")

    # Chinese Benchmarks
    chinese_medicine_mcmle = config.get("chinese_medicine_mcmle")
    # CMMLU Medicine
    chinese_cmmlu_medicine_anatomy = config.get("chinese_cmmlu_medicine_anatomy")
    chinese_cmmlu_medicine_clinical_knowledge = config.get("chinese_cmmlu_medicine_clinical_knowledge")
    chinese_cmmlu_medicine_college_medicine = config.get("chinese_cmmlu_medicine_college_medicine")
    chinese_cmmlu_medicine_genetics = config.get("chinese_cmmlu_medicine_genetics")
    chinese_cmmlu_medicine_nutrition = config.get("chinese_cmmlu_medicine_nutrition")
    chinese_cmmlu_medicine_tcm = config.get("chinese_cmmlu_medicine_tcm")
    chinese_cmmlu_medicine_virology = config.get("chinese_cmmlu_medicine_virology")
    # CMMLU Physics
    chinese_cmmlu_physics_astronomy = config.get("chinese_cmmlu_physics_astronomy")
    chinese_cmmlu_physics_conceptual_physics = config.get("chinese_cmmlu_physics_conceptual_physics")
    chinese_cmmlu_physics_high_school_physics = config.get("chinese_cmmlu_physics_high_school_physics")
    # CMMLU Chemistry
    chinese_cmmlu_chemistry_high_school_chemistry = config.get("chinese_cmmlu_chemistry_high_school_chemistry")
    # CMMLU Biology
    chinese_cmmlu_biology_high_school_biology = config.get("chinese_cmmlu_biology_high_school_biology")
    # CMMLU Computer Science
    chinese_cmmlu_cs_computer_science = config.get("chinese_cmmlu_cs_computer_science")
    chinese_cmmlu_cs_computer_security = config.get("chinese_cmmlu_cs_computer_security")
    chinese_cmmlu_cs_machine_learning = config.get("chinese_cmmlu_cs_machine_learning")
    # CMMLU Math
    chinese_cmmlu_math_college_actuarial_science = config.get("chinese_cmmlu_math_college_actuarial_science")
    chinese_cmmlu_math_college_college_mathematics = config.get("chinese_cmmlu_math_college_college_mathematics")
    chinese_cmmlu_math_elementary_mathematics = config.get("chinese_cmmlu_math_elementary_mathematics")
    chinese_cmmlu_math_high_school_mathematics = config.get("chinese_cmmlu_math_high_school_mathematics")
    # CMMLU Engineering
    chinese_cmmlu_engineering_college_engineering_hydrology = config.get(
        "chinese_cmmlu_engineering_college_engineering_hydrology"
    )
    chinese_cmmlu_engineering_electrical_engineering = config.get(
        "chinese_cmmlu_engineering_electrical_engineering"
    )
    # CMMLU Statistics
    chinese_cmmlu_statistics_college_medical_statistics = config.get(
        "chinese_cmmlu_statistics_college_medical_statistics"
    )

    # French Benchmarks
    french_medicine_medmcqa = config.get("french_medicine_medmcqa")
    french_medicine_medexpqa = config.get("french_medicine_medexpqa")
    # MMLU Medicine
    french_mmlu_medicine_anatomy = config.get("french_mmlu_medicine_anatomy")
    french_mmlu_medicine_clinical_knowledge = config.get("french_mmlu_medicine_clinical_knowledge")
    french_mmlu_medicine_college_biology = config.get("french_mmlu_medicine_college_biology")
    french_mmlu_medicine_college_medicine = config.get("french_mmlu_medicine_college_medicine")
    french_mmlu_medicine_medical_genetics = config.get("french_mmlu_medicine_medical_genetics")
    french_mmlu_medicine_professional_medicine = config.get("french_mmlu_medicine_professional_medicine")
    # MMLU Physics
    french_mmlu_physics_astronomy = config.get("french_mmlu_physics_astronomy")
    french_mmlu_physics_college_physics = config.get("french_mmlu_physics_college_physics")
    french_mmlu_physics_conceptual_physics = config.get("french_mmlu_physics_conceptual_physics")
    french_mmlu_physics_high_school_physics = config.get("french_mmlu_physics_high_school_physics")
    # MMLU Chemistry
    french_mmlu_chemistry_college_chemistry = config.get("french_mmlu_chemistry_college_chemistry")
    french_mmlu_chemistry_high_school_chemistry = config.get("french_mmlu_chemistry_high_school_chemistry")
    # MMLU Biology
    french_mmlu_biology_college_biology = config.get("french_mmlu_biology_college_biology")
    french_mmlu_biology_high_school_biology = config.get("french_mmlu_biology_high_school_biology")
    # MMLU Computer Science
    french_mmlu_cs_college_computer_science = config.get("french_mmlu_cs_college_computer_science")
    french_mmlu_cs_computer_security = config.get("french_mmlu_cs_computer_security")
    french_mmlu_cs_high_school_computer_science = config.get("french_mmlu_cs_high_school_computer_science")
    french_mmlu_cs_machine_learning = config.get("french_mmlu_cs_machine_learning")
    # MMLU Math
    french_mmlu_math_abstract_algebra = config.get("french_mmlu_math_abstract_algebra")
    french_mmlu_math_college_mathematics = config.get("french_mmlu_math_college_mathematics")
    french_mmlu_math_elementary_mathematics = config.get("french_mmlu_math_elementary_mathematics")
    french_mmlu_math_high_school_mathematics = config.get("french_mmlu_math_high_school_mathematics")
    french_mmlu_math_high_school_statistics = config.get("french_mmlu_math_high_school_statistics")
    # MMLU Engineering
    french_mmlu_engineering_electrical_engineering = config.get("french_mmlu_engineering_electrical_engineering")

    # Spanish Benchmarks
    spanish_medicine_headqa = config.get("spanish_medicine_headqa")
    spanish_medicine_medexpqa = config.get("spanish_medicine_medexpqa")
    # MMLU Medicine
    spanish_mmlu_medicine_anatomy = config.get("spanish_mmlu_medicine_anatomy")
    spanish_mmlu_medicine_clinical_knowledge = config.get("spanish_mmlu_medicine_clinical_knowledge")
    spanish_mmlu_medicine_college_biology = config.get("spanish_mmlu_medicine_college_biology")
    spanish_mmlu_medicine_college_medicine = config.get("spanish_mmlu_medicine_college_medicine")
    spanish_mmlu_medicine_medical_genetics = config.get("spanish_mmlu_medicine_medical_genetics")
    spanish_mmlu_medicine_professional_medicine = config.get("spanish_mmlu_medicine_professional_medicine")
    # MMLU Physics
    spanish_mmlu_physics_astronomy = config.get("spanish_mmlu_physics_astronomy")
    spanish_mmlu_physics_college_physics = config.get("spanish_mmlu_physics_college_physics")
    spanish_mmlu_physics_conceptual_physics = config.get("spanish_mmlu_physics_conceptual_physics")
    spanish_mmlu_physics_high_school_physics = config.get("spanish_mmlu_physics_high_school_physics")
    # MMLU Chemistry
    spanish_mmlu_chemistry_college_chemistry = config.get("spanish_mmlu_chemistry_college_chemistry")
    spanish_mmlu_chemistry_high_school_chemistry = config.get("spanish_mmlu_chemistry_high_school_chemistry")
    # MMLU Biology
    spanish_mmlu_biology_college_biology = config.get("spanish_mmlu_biology_college_biology")
    spanish_mmlu_biology_high_school_biology = config.get("spanish_mmlu_biology_high_school_biology")
    # MMLU Computer Science
    spanish_mmlu_cs_college_computer_science = config.get("spanish_mmlu_cs_college_computer_science")
    spanish_mmlu_cs_computer_security = config.get("spanish_mmlu_cs_computer_security")
    spanish_mmlu_cs_high_school_computer_science = config.get("spanish_mmlu_cs_high_school_computer_science")
    spanish_mmlu_cs_machine_learning = config.get("spanish_mmlu_cs_machine_learning")
    # MMLU Math
    spanish_mmlu_math_abstract_algebra = config.get("spanish_mmlu_math_abstract_algebra")
    spanish_mmlu_math_college_mathematics = config.get("spanish_mmlu_math_college_mathematics")
    spanish_mmlu_math_elementary_mathematics = config.get("spanish_mmlu_math_elementary_mathematics")
    spanish_mmlu_math_high_school_mathematics = config.get("spanish_mmlu_math_high_school_mathematics")
    spanish_mmlu_math_high_school_statistics = config.get("spanish_mmlu_math_high_school_statistics")
    # MMLU Engineering
    spanish_mmlu_engineering_electrical_engineering = config.get("spanish_mmlu_engineering_electrical_engineering")

    # Hindi Benchmarks
    # MMLU Medicine
    hindi_mmlu_medicine_anatomy = config.get("hindi_mmlu_medicine_anatomy")
    hindi_mmlu_medicine_clinical_knowledge = config.get("hindi_mmlu_medicine_clinical_knowledge")
    hindi_mmlu_medicine_college_biology = config.get("hindi_mmlu_medicine_college_biology")
    hindi_mmlu_medicine_college_medicine = config.get("hindi_mmlu_medicine_college_medicine")
    hindi_mmlu_medicine_medical_genetics = config.get("hindi_mmlu_medicine_medical_genetics")
    hindi_mmlu_medicine_professional_medicine = config.get("hindi_mmlu_medicine_professional_medicine")
    # MMLU Physics
    hindi_mmlu_physics_astronomy = config.get("hindi_mmlu_physics_astronomy")
    hindi_mmlu_physics_college_physics = config.get("hindi_mmlu_physics_college_physics")
    hindi_mmlu_physics_conceptual_physics = config.get("hindi_mmlu_physics_conceptual_physics")
    hindi_mmlu_physics_high_school_physics = config.get("hindi_mmlu_physics_high_school_physics")
    # MMLU Chemistry
    hindi_mmlu_chemistry_college_chemistry = config.get("hindi_mmlu_chemistry_college_chemistry")
    hindi_mmlu_chemistry_high_school_chemistry = config.get("hindi_mmlu_chemistry_high_school_chemistry")
    # MMLU Biology
    hindi_mmlu_biology_college_biology = config.get("hindi_mmlu_biology_college_biology")
    hindi_mmlu_biology_high_school_biology = config.get("hindi_mmlu_biology_high_school_biology")
    # MMLU Computer Science
    hindi_mmlu_cs_college_computer_science = config.get("hindi_mmlu_cs_college_computer_science")
    hindi_mmlu_cs_computer_security = config.get("hindi_mmlu_cs_computer_security")
    hindi_mmlu_cs_high_school_computer_science = config.get("hindi_mmlu_cs_high_school_computer_science")
    hindi_mmlu_cs_machine_learning = config.get("hindi_mmlu_cs_machine_learning")
    # MMLU Math
    hindi_mmlu_math_abstract_algebra = config.get("hindi_mmlu_math_abstract_algebra")
    hindi_mmlu_math_college_mathematics = config.get("hindi_mmlu_math_college_mathematics")
    hindi_mmlu_math_elementary_mathematics = config.get("hindi_mmlu_math_elementary_mathematics")
    hindi_mmlu_math_high_school_mathematics = config.get("hindi_mmlu_math_high_school_mathematics")
    hindi_mmlu_math_high_school_statistics = config.get("hindi_mmlu_math_high_school_statistics")
    # MMLU Engineering
    hindi_mmlu_engineering_electrical_engineering = config.get("hindi_mmlu_engineering_electrical_engineering")

    # English Benchmarks
    print("English Benchmark - Start performance evaluation on the MedQA Benchmark")
    dataset = "english_medicine_medqa"
    test_path = english_medicine_medqa
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    medqa_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("English Benchmark - Start performance evaluation on the PubMedQA Benchmark")
    dataset = "english_medicine_pubmedqa"
    test_path = english_medicine_pubmedqa
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    pubmedqa_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("English Benchmark - Start performance evaluation on the MedMCQA Benchmark")
    dataset = "english_medicine_medmcqa"
    test_path = english_medicine_medmcqa
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    medmcqa_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("English Benchmark - Start performance evaluation on the Internal USMLE STEP 1 Benchmark")
    dataset = "english_medicine_usmle_step1"
    test_path = english_medicine_usmle_step1
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    usmle_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("English Benchmark - Start performance evaluation on the Internal USMLE STEP 2 Benchmark")
    dataset = "english_medicine_usmle_step2"
    test_path = english_medicine_usmle_step2
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    usmle_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("English Benchmark - Start performance evaluation on the Internal USMLE STEP 3 Benchmark")
    dataset = "english_medicine_usmle_step3"
    test_path = english_medicine_usmle_step3
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    usmle_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("English Benchmark - Start performance evaluation on the Internal USMLE Ethics Benchmark")
    dataset = "english_medicine_usmle_ethics"
    test_path = english_medicine_usmle_ethics
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    usmle_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("English Benchmark - Start performance evaluation on the MedExpQA English Benchmark")
    dataset = "english_medicine_medexpqa"
    test_path = english_medicine_medexpqa
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    medexpqa_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("English Benchmark - Start performance evaluation on the MMLU Anatomy Benchmark")
    dataset = "english_mmlu_medicine_anatomy"
    test_path = english_mmlu_medicine_anatomy
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    mmlu_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("English Benchmark - Start performance evaluation on the MMLU Clinical Knowledge Benchmark")
    dataset = "english_mmlu_medicine_clinical_knowledge"
    test_path = english_mmlu_medicine_clinical_knowledge
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    mmlu_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("English Benchmark - Start performance evaluation on the MMLU College Biology Benchmark")
    dataset = "english_mmlu_medicine_college_biology"
    test_path = english_mmlu_medicine_college_biology
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    mmlu_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("English Benchmark - Start performance evaluation on the MMLU College Medicine Benchmark")
    dataset = "english_mmlu_medicine_college_medicine"
    test_path = english_mmlu_medicine_college_medicine
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    mmlu_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("English Benchmark - Start performance evaluation on the MMLU Medical Genetics Benchmark")
    dataset = "english_mmlu_medicine_medical_genetics"
    test_path = english_mmlu_medicine_medical_genetics
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    mmlu_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("English Benchmark - Start performance evaluation on the MMLU Professional Medicine Benchmark")
    dataset = "english_mmlu_medicine_professional_medicine"
    test_path = english_mmlu_medicine_professional_medicine
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    mmlu_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("English Benchmark - Start performance evaluation on the MMLU Astronomy Benchmark")
    dataset = "english_mmlu_physics_astronomy"
    test_path = english_mmlu_physics_astronomy
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    mmlu_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("English Benchmark - Start performance evaluation on the MMLU College Physics Benchmark")
    dataset = "english_mmlu_physics_college_physics"
    test_path = english_mmlu_physics_college_physics
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    mmlu_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("English Benchmark - Start performance evaluation on the MMLU Conceptual Physics Benchmark")
    dataset = "english_mmlu_physics_conceptual_physics"
    test_path = english_mmlu_physics_conceptual_physics
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    mmlu_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("English Benchmark - Start performance evaluation on the MMLU High School Physics Benchmark")
    dataset = "english_mmlu_physics_high_school_physics"
    test_path = english_mmlu_physics_high_school_physics
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    mmlu_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("English Benchmark - Start performance evaluation on the MMLU College Chemistry Benchmark")
    dataset = "english_mmlu_chemistry_college_chemistry"
    test_path = english_mmlu_chemistry_college_chemistry
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    mmlu_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("English Benchmark - Start performance evaluation on the MMLU High School Chemistry Benchmark")
    dataset = "english_mmlu_chemistry_high_school_chemistry"
    test_path = english_mmlu_chemistry_high_school_chemistry
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    mmlu_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("English Benchmark - Start performance evaluation on the MMLU College Biology Benchmark")
    dataset = "english_mmlu_biology_college_biology"
    test_path = english_mmlu_biology_college_biology
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    mmlu_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("English Benchmark - Start performance evaluation on the MMLU High School Biology Benchmark")
    dataset = "english_mmlu_biology_high_school_biology"
    test_path = english_mmlu_biology_high_school_biology
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    mmlu_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("English Benchmark - Start performance evaluation on the MMLU College Computer Science Benchmark")
    dataset = "english_mmlu_cs_college_computer_science"
    test_path = english_mmlu_cs_college_computer_science
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    mmlu_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("English Benchmark - Start performance evaluation on the MMLU Computer Security Benchmark")
    dataset = "english_mmlu_cs_computer_security"
    test_path = english_mmlu_cs_computer_security
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    mmlu_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("English Benchmark - Start performance evaluation on the MMLU High School Computer Science Benchmark")
    dataset = "english_mmlu_cs_high_school_computer_science"
    test_path = english_mmlu_cs_high_school_computer_science
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    mmlu_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("English Benchmark - Start performance evaluation on the MMLU Machine Learning Benchmark")
    dataset = "english_mmlu_cs_machine_learning"
    test_path = english_mmlu_cs_machine_learning
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    mmlu_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("English Benchmark - Start performance evaluation on the MMLU Abstract Algebra Benchmark")
    dataset = "english_mmlu_math_abstract_algebra"
    test_path = english_mmlu_math_abstract_algebra
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    mmlu_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("English Benchmark - Start performance evaluation on the MMLU College Mathematics Benchmark")
    dataset = "english_mmlu_math_college_mathematics"
    test_path = english_mmlu_math_college_mathematics
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    mmlu_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("English Benchmark - Start performance evaluation on the MMLU Elementary Mathematics Benchmark")
    dataset = "english_mmlu_math_elementary_mathematics"
    test_path = english_mmlu_math_elementary_mathematics
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    mmlu_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("English Benchmark - Start performance evaluation on the MMLU High School Mathematics Benchmark")
    dataset = "english_mmlu_math_high_school_mathematics"
    test_path = english_mmlu_math_high_school_mathematics
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    mmlu_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("English Benchmark - Start performance evaluation on the MMLU High School Statistics Benchmark")
    dataset = "english_mmlu_math_high_school_statistics"
    test_path = english_mmlu_math_high_school_statistics
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    mmlu_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("English Benchmark - Start performance evaluation on the MMLU Electrical Engineering Benchmark")
    dataset = "english_mmlu_engineering_electrical_engineering"
    test_path = english_mmlu_engineering_electrical_engineering
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    mmlu_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    # Mandarin Chinese Benchmarks
    print("Mandarin Chinese Benchmark - Start performance evaluation on the MedQA-MCMLE Benchmark")
    dataset = "chinese_medicine_mcmle"
    test_path = chinese_medicine_mcmle
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    mcmle_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("Mandarin Chinese Benchmark - Start performance evaluation on the CMMLU Anatomy Benchmark")
    dataset = "chinese_cmmlu_medicine_anatomy"
    test_path = chinese_cmmlu_medicine_anatomy
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    cmmlu_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("Mandarin Chinese Benchmark - Start performance evaluation on the CMMLU Clinical Knowledge Benchmark")
    dataset = "chinese_cmmlu_medicine_clinical_knowledge"
    test_path = chinese_cmmlu_medicine_clinical_knowledge
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    cmmlu_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("Mandarin Chinese Benchmark - Start performance evaluation on the CMMLU College Medicine Benchmark")
    dataset = "chinese_cmmlu_medicine_college_medicine"
    test_path = chinese_cmmlu_medicine_college_medicine
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    cmmlu_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("Mandarin Chinese Benchmark - Start performance evaluation on the CMMLU Genetics Benchmark")
    dataset = "chinese_cmmlu_medicine_genetics"
    test_path = chinese_cmmlu_medicine_genetics
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    cmmlu_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("Mandarin Chinese Benchmark - Start performance evaluation on the CMMLU Nutrition Benchmark")
    dataset = "chinese_cmmlu_medicine_nutrition"
    test_path = chinese_cmmlu_medicine_nutrition
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    cmmlu_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("Mandarin Chinese Benchmark - Start performance evaluation on the CMMLU TCM Benchmark")
    dataset = "chinese_cmmlu_medicine_tcm"
    test_path = chinese_cmmlu_medicine_tcm
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    cmmlu_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("Mandarin Chinese Benchmark - Start performance evaluation on the CMMLU Virology Benchmark")
    dataset = "chinese_cmmlu_medicine_virology"
    test_path = chinese_cmmlu_medicine_virology
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    cmmlu_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    # Physics (3)
    print("Mandarin Chinese Benchmark - Start performance evaluation on the CMMLU Astronomy Benchmark")
    dataset = "chinese_cmmlu_physics_astronomy"
    test_path = chinese_cmmlu_physics_astronomy
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    cmmlu_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("Mandarin Chinese Benchmark - Start performance evaluation on the CMMLU Conceptual Physics Benchmark")
    dataset = "chinese_cmmlu_physics_conceptual_physics"
    test_path = chinese_cmmlu_physics_conceptual_physics
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    cmmlu_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("Mandarin Chinese Benchmark - Start performance evaluation on the CMMLU High School Physics Benchmark")
    dataset = "chinese_cmmlu_physics_high_school_physics"
    test_path = chinese_cmmlu_physics_high_school_physics
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    cmmlu_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    # Chemistry (1)
    print("Mandarin Chinese Benchmark - Start performance evaluation on the CMMLU High School Chemistry Benchmark")
    dataset = "chinese_cmmlu_chemistry_high_school_chemistry"
    test_path = chinese_cmmlu_chemistry_high_school_chemistry
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    cmmlu_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    # Biology (1)
    print("Mandarin Chinese Benchmark - Start performance evaluation on the CMMLU High School Biology Benchmark")
    dataset = "chinese_cmmlu_biology_high_school_biology"
    test_path = chinese_cmmlu_biology_high_school_biology
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    cmmlu_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    # Computer Science (3)
    print("Mandarin Chinese Benchmark - Start performance evaluation on the CMMLU Computer Science Benchmark")
    dataset = "chinese_cmmlu_cs_computer_science"
    test_path = chinese_cmmlu_cs_computer_science
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    cmmlu_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("Mandarin Chinese Benchmark - Start performance evaluation on the CMMLU Computer Security Benchmark")
    dataset = "chinese_cmmlu_cs_computer_security"
    test_path = chinese_cmmlu_cs_computer_security
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    cmmlu_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("Mandarin Chinese Benchmark - Start performance evaluation on the CMMLU Machine Learning Benchmark")
    dataset = "chinese_cmmlu_cs_machine_learning"
    test_path = chinese_cmmlu_cs_machine_learning
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    cmmlu_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    # Math (4)
    print("Mandarin Chinese Benchmark - Start performance evaluation on the CMMLU College Actuarial Science Benchmark")
    dataset = "chinese_cmmlu_math_college_actuarial_science"
    test_path = chinese_cmmlu_math_college_actuarial_science
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    cmmlu_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("Mandarin Chinese Benchmark - Start performance evaluation on the CMMLU College Mathematics Benchmark")
    dataset = "chinese_cmmlu_math_college_college_mathematics"
    test_path = chinese_cmmlu_math_college_college_mathematics
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    cmmlu_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("Mandarin Chinese Benchmark - Start performance evaluation on the CMMLU Elementary Mathematics Benchmark")
    dataset = "chinese_cmmlu_math_elementary_mathematics"
    test_path = chinese_cmmlu_math_elementary_mathematics
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    cmmlu_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("Mandarin Chinese Benchmark - Start performance evaluation on the CMMLU High School Mathematics Benchmark")
    dataset = "chinese_cmmlu_math_high_school_mathematics"
    test_path = chinese_cmmlu_math_high_school_mathematics
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    cmmlu_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    # Engineering (2)
    print("Mandarin Chinese Benchmark - Start performance evaluation on the CMMLU "
          "College Engineering Hydrology Benchmark")
    dataset = "chinese_cmmlu_engineering_college_engineering_hydrology"
    test_path = chinese_cmmlu_engineering_college_engineering_hydrology
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    cmmlu_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("Mandarin Chinese Benchmark - Start performance evaluation on the CMMLU Electrical Engineering Benchmark")
    dataset = "chinese_cmmlu_engineering_electrical_engineering"
    test_path = chinese_cmmlu_engineering_electrical_engineering
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    cmmlu_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    # Statistics (1)
    print("Mandarin Chinese Benchmark - Start performance evaluation on the CMMLU College Medical Statistics Benchmark")
    dataset = "chinese_cmmlu_statistics_college_medical_statistics"
    test_path = chinese_cmmlu_statistics_college_medical_statistics
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    cmmlu_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    # French Benchmarks
    print("French Benchmark - Start performance evaluation on the FrenchMedMCQA Benchmark")
    dataset = "french_medicine_medmcqa"
    test_path = french_medicine_medmcqa
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    frenchmedmcqa_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("French Benchmark - Start performance evaluation on the MedExpQA French Benchmark")
    dataset = "french_medicine_medexpqa"
    test_path = french_medicine_medexpqa
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    medexpqa_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("French Benchmark - Start performance evaluation on the French MMLU Anatomy Benchmark")
    dataset = "french_mmlu_medicine_anatomy"
    test_path = french_mmlu_medicine_anatomy
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    mmlu_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("French Benchmark - Start performance evaluation on the French MMLU Clinical Knowledge Benchmark")
    dataset = "french_mmlu_medicine_clinical_knowledge"
    test_path = french_mmlu_medicine_clinical_knowledge
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    mmlu_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("French Benchmark - Start performance evaluation on the French MMLU College Biology Benchmark")
    dataset = "french_mmlu_medicine_college_biology"
    test_path = french_mmlu_medicine_college_biology
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    mmlu_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("French Benchmark - Start performance evaluation on the French MMLU College Medicine Benchmark")
    dataset = "french_mmlu_medicine_college_medicine"
    test_path = french_mmlu_medicine_college_medicine
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    mmlu_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("French Benchmark - Start performance evaluation on the French MMLU Medical Genetics Benchmark")
    dataset = "french_mmlu_medicine_medical_genetics"
    test_path = french_mmlu_medicine_medical_genetics
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    mmlu_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("French Benchmark - Start performance evaluation on the French MMLU Professional Medicine Benchmark")
    dataset = "french_mmlu_medicine_professional_medicine"
    test_path = french_mmlu_medicine_professional_medicine
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    mmlu_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("French Benchmark - Start performance evaluation on the MMLU Astronomy Benchmark")
    dataset = "french_mmlu_physics_astronomy"
    test_path = french_mmlu_physics_astronomy
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    mmlu_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("French Benchmark - Start performance evaluation on the MMLU College Physics Benchmark")
    dataset = "french_mmlu_physics_college_physics"
    test_path = french_mmlu_physics_college_physics
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    mmlu_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("French Benchmark - Start performance evaluation on the MMLU Conceptual Physics Benchmark")
    dataset = "french_mmlu_physics_conceptual_physics"
    test_path = french_mmlu_physics_conceptual_physics
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    mmlu_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("French Benchmark - Start performance evaluation on the MMLU High School Physics Benchmark")
    dataset = "french_mmlu_physics_high_school_physics"
    test_path = french_mmlu_physics_high_school_physics
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    mmlu_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("French Benchmark - Start performance evaluation on the MMLU College Chemistry Benchmark")
    dataset = "french_mmlu_chemistry_college_chemistry"
    test_path = french_mmlu_chemistry_college_chemistry
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    mmlu_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("French Benchmark - Start performance evaluation on the MMLU High School Chemistry Benchmark")
    dataset = "french_mmlu_chemistry_high_school_chemistry"
    test_path = french_mmlu_chemistry_high_school_chemistry
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    mmlu_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("French Benchmark - Start performance evaluation on the MMLU College Biology Benchmark")
    dataset = "french_mmlu_biology_college_biology"
    test_path = french_mmlu_biology_college_biology
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    mmlu_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("French Benchmark - Start performance evaluation on the MMLU High School Biology Benchmark")
    dataset = "french_mmlu_biology_high_school_biology"
    test_path = french_mmlu_biology_high_school_biology
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    mmlu_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("French Benchmark - Start performance evaluation on the MMLU College Computer Science Benchmark")
    dataset = "french_mmlu_cs_college_computer_science"
    test_path = french_mmlu_cs_college_computer_science
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    mmlu_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("French Benchmark - Start performance evaluation on the MMLU Computer Security Benchmark")
    dataset = "french_mmlu_cs_computer_security"
    test_path = french_mmlu_cs_computer_security
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    mmlu_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("French Benchmark - Start performance evaluation on the MMLU High School Computer Science Benchmark")
    dataset = "french_mmlu_cs_high_school_computer_science"
    test_path = french_mmlu_cs_high_school_computer_science
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    mmlu_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("French Benchmark - Start performance evaluation on the MMLU Machine Learning Benchmark")
    dataset = "french_mmlu_cs_machine_learning"
    test_path = french_mmlu_cs_machine_learning
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    mmlu_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("French Benchmark - Start performance evaluation on the MMLU Abstract Algebra Benchmark")
    dataset = "french_mmlu_math_abstract_algebra"
    test_path = french_mmlu_math_abstract_algebra
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    mmlu_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("French Benchmark - Start performance evaluation on the MMLU College Mathematics Benchmark")
    dataset = "french_mmlu_math_college_mathematics"
    test_path = french_mmlu_math_college_mathematics
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    mmlu_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("French Benchmark - Start performance evaluation on the MMLU Elementary Mathematics Benchmark")
    dataset = "french_mmlu_math_elementary_mathematics"
    test_path = french_mmlu_math_elementary_mathematics
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    mmlu_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("French Benchmark - Start performance evaluation on the MMLU High School Mathematics Benchmark")
    dataset = "french_mmlu_math_high_school_mathematics"
    test_path = french_mmlu_math_high_school_mathematics
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    mmlu_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("French Benchmark - Start performance evaluation on the MMLU High School Statistics Benchmark")
    dataset = "french_mmlu_math_high_school_statistics"
    test_path = french_mmlu_math_high_school_statistics
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    mmlu_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("French Benchmark - Start performance evaluation on the MMLU Electrical Engineering Benchmark")
    dataset = "french_mmlu_engineering_electrical_engineering"
    test_path = french_mmlu_engineering_electrical_engineering
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    mmlu_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    # Spanish Benchmarks
    print("Spanish Benchmark - Start performance evaluation on the HEAD-QA Benchmark")
    dataset = "spanish_medicine_headqa"
    test_path = spanish_medicine_headqa
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    headqa_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("Spanish Benchmark - Start performance evaluation on the MedExpQA Spanish Benchmark")
    dataset = "spanish_medicine_medexpqa"
    test_path = spanish_medicine_medexpqa
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    medexpqa_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("Spanish Benchmark - Start performance evaluation on the Spanish MMLU Anatomy Benchmark")
    dataset = "spanish_mmlu_medicine_anatomy"
    test_path = spanish_mmlu_medicine_anatomy
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    mmlu_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("Spanish Benchmark - Start performance evaluation on the Spanish MMLU Clinical Knowledge Benchmark")
    dataset = "spanish_mmlu_medicine_clinical_knowledge"
    test_path = spanish_mmlu_medicine_clinical_knowledge
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    mmlu_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("Spanish Benchmark - Start performance evaluation on the Spanish MMLU College Biology Benchmark")
    dataset = "spanish_mmlu_medicine_college_biology"
    test_path = spanish_mmlu_medicine_college_biology
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    mmlu_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("Spanish Benchmark - Start performance evaluation on the Spanish MMLU College Medicine Benchmark")
    dataset = "spanish_mmlu_medicine_college_medicine"
    test_path = spanish_mmlu_medicine_college_medicine
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    mmlu_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("Spanish Benchmark - Start performance evaluation on the Spanish MMLU Medical Genetics Benchmark")
    dataset = "spanish_mmlu_medicine_medical_genetics"
    test_path = spanish_mmlu_medicine_medical_genetics
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    mmlu_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("Spanish Benchmark - Start performance evaluation on the Spanish MMLU Professional Medicine Benchmark")
    dataset = "spanish_mmlu_medicine_professional_medicine"
    test_path = spanish_mmlu_medicine_professional_medicine
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    mmlu_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("Spanish Benchmark - Start performance evaluation on the MMLU Astronomy Benchmark")
    dataset = "spanish_mmlu_physics_astronomy"
    test_path = spanish_mmlu_physics_astronomy
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    mmlu_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("Spanish Benchmark - Start performance evaluation on the MMLU College Physics Benchmark")
    dataset = "spanish_mmlu_physics_college_physics"
    test_path = spanish_mmlu_physics_college_physics
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    mmlu_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("Spanish Benchmark - Start performance evaluation on the MMLU Conceptual Physics Benchmark")
    dataset = "spanish_mmlu_physics_conceptual_physics"
    test_path = spanish_mmlu_physics_conceptual_physics
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    mmlu_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("Spanish Benchmark - Start performance evaluation on the MMLU High School Physics Benchmark")
    dataset = "spanish_mmlu_physics_high_school_physics"
    test_path = spanish_mmlu_physics_high_school_physics
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    mmlu_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("Spanish Benchmark - Start performance evaluation on the MMLU College Chemistry Benchmark")
    dataset = "spanish_mmlu_chemistry_college_chemistry"
    test_path = spanish_mmlu_chemistry_college_chemistry
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    mmlu_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("Spanish Benchmark - Start performance evaluation on the MMLU High School Chemistry Benchmark")
    dataset = "spanish_mmlu_chemistry_high_school_chemistry"
    test_path = spanish_mmlu_chemistry_high_school_chemistry
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    mmlu_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("Spanish Benchmark - Start performance evaluation on the MMLU College Biology Benchmark")
    dataset = "spanish_mmlu_biology_college_biology"
    test_path = spanish_mmlu_biology_college_biology
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    mmlu_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("Spanish Benchmark - Start performance evaluation on the MMLU High School Biology Benchmark")
    dataset = "spanish_mmlu_biology_high_school_biology"
    test_path = spanish_mmlu_biology_high_school_biology
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    mmlu_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("Spanish Benchmark - Start performance evaluation on the MMLU College Computer Science Benchmark")
    dataset = "spanish_mmlu_cs_college_computer_science"
    test_path = spanish_mmlu_cs_college_computer_science
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    mmlu_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("Spanish Benchmark - Start performance evaluation on the MMLU Computer Security Benchmark")
    dataset = "spanish_mmlu_cs_computer_security"
    test_path = spanish_mmlu_cs_computer_security
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    mmlu_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("Spanish Benchmark - Start performance evaluation on the MMLU High School Computer Science Benchmark")
    dataset = "spanish_mmlu_cs_high_school_computer_science"
    test_path = spanish_mmlu_cs_high_school_computer_science
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    mmlu_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("Spanish Benchmark - Start performance evaluation on the MMLU Machine Learning Benchmark")
    dataset = "spanish_mmlu_cs_machine_learning"
    test_path = spanish_mmlu_cs_machine_learning
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    mmlu_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("Spanish Benchmark - Start performance evaluation on the MMLU Abstract Algebra Benchmark")
    dataset = "spanish_mmlu_math_abstract_algebra"
    test_path = spanish_mmlu_math_abstract_algebra
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    mmlu_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("Spanish Benchmark - Start performance evaluation on the MMLU College Mathematics Benchmark")
    dataset = "spanish_mmlu_math_college_mathematics"
    test_path = spanish_mmlu_math_college_mathematics
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    mmlu_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("Spanish Benchmark - Start performance evaluation on the MMLU Elementary Mathematics Benchmark")
    dataset = "spanish_mmlu_math_elementary_mathematics"
    test_path = spanish_mmlu_math_elementary_mathematics
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    mmlu_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("Spanish Benchmark - Start performance evaluation on the MMLU High School Mathematics Benchmark")
    dataset = "spanish_mmlu_math_high_school_mathematics"
    test_path = spanish_mmlu_math_high_school_mathematics
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    mmlu_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("Spanish Benchmark - Start performance evaluation on the MMLU High School Statistics Benchmark")
    dataset = "spanish_mmlu_math_high_school_statistics"
    test_path = spanish_mmlu_math_high_school_statistics
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    mmlu_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("Spanish Benchmark - Start performance evaluation on the MMLU Electrical Engineering Benchmark")
    dataset = "spanish_mmlu_engineering_electrical_engineering"
    test_path = spanish_mmlu_engineering_electrical_engineering
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    mmlu_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    # Hindi Benchmarks
    print("Hindi Benchmark - Start performance evaluation on the Hindi MMLU Anatomy Benchmark")
    dataset = "hindi_mmlu_medicine_anatomy"
    test_path = hindi_mmlu_medicine_anatomy
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    mmlu_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("Hindi Benchmark - Start performance evaluation on the Hindi MMLU Clinical Knowledge Benchmark")
    dataset = "hindi_mmlu_medicine_clinical_knowledge"
    test_path = hindi_mmlu_medicine_clinical_knowledge
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    mmlu_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("Hindi Benchmark - Start performance evaluation on the Hindi MMLU College Biology Benchmark")
    dataset = "hindi_mmlu_medicine_college_biology"
    test_path = hindi_mmlu_medicine_college_biology
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    mmlu_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("Hindi Benchmark - Start performance evaluation on the Hindi MMLU College Medicine Benchmark")
    dataset = "hindi_mmlu_medicine_college_medicine"
    test_path = hindi_mmlu_medicine_college_medicine
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    mmlu_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("Hindi Benchmark - Start performance evaluation on the Hindi MMLU Medical Genetics Benchmark")
    dataset = "hindi_mmlu_medicine_medical_genetics"
    test_path = hindi_mmlu_medicine_medical_genetics
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    mmlu_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("Hindi Benchmark - Start performance evaluation on the Hindi MMLU Professional Medicine Benchmark")
    dataset = "hindi_mmlu_medicine_professional_medicine"
    test_path = hindi_mmlu_medicine_professional_medicine
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    mmlu_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("Hindi Benchmark - Start performance evaluation on the MMLU Astronomy Benchmark")
    dataset = "hindi_mmlu_physics_astronomy"
    test_path = hindi_mmlu_physics_astronomy
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    mmlu_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("Hindi Benchmark - Start performance evaluation on the MMLU College Physics Benchmark")
    dataset = "hindi_mmlu_physics_college_physics"
    test_path = hindi_mmlu_physics_college_physics
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    mmlu_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("Hindi Benchmark - Start performance evaluation on the MMLU Conceptual Physics Benchmark")
    dataset = "hindi_mmlu_physics_conceptual_physics"
    test_path = hindi_mmlu_physics_conceptual_physics
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    mmlu_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("Hindi Benchmark - Start performance evaluation on the MMLU High School Physics Benchmark")
    dataset = "hindi_mmlu_physics_high_school_physics"
    test_path = hindi_mmlu_physics_high_school_physics
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    mmlu_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("Hindi Benchmark - Start performance evaluation on the MMLU College Chemistry Benchmark")
    dataset = "hindi_mmlu_chemistry_college_chemistry"
    test_path = hindi_mmlu_chemistry_college_chemistry
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    mmlu_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("Hindi Benchmark - Start performance evaluation on the MMLU High School Chemistry Benchmark")
    dataset = "hindi_mmlu_chemistry_high_school_chemistry"
    test_path = hindi_mmlu_chemistry_high_school_chemistry
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    mmlu_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("Hindi Benchmark - Start performance evaluation on the MMLU College Biology Benchmark")
    dataset = "hindi_mmlu_biology_college_biology"
    test_path = hindi_mmlu_biology_college_biology
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    mmlu_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("Hindi Benchmark - Start performance evaluation on the MMLU High School Biology Benchmark")
    dataset = "hindi_mmlu_biology_high_school_biology"
    test_path = hindi_mmlu_biology_high_school_biology
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    mmlu_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("Hindi Benchmark - Start performance evaluation on the MMLU College Computer Science Benchmark")
    dataset = "hindi_mmlu_cs_college_computer_science"
    test_path = hindi_mmlu_cs_college_computer_science
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    mmlu_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("Hindi Benchmark - Start performance evaluation on the MMLU Computer Security Benchmark")
    dataset = "hindi_mmlu_cs_computer_security"
    test_path = hindi_mmlu_cs_computer_security
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    mmlu_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("Hindi Benchmark - Start performance evaluation on the MMLU High School Computer Science Benchmark")
    dataset = "hindi_mmlu_cs_high_school_computer_science"
    test_path = hindi_mmlu_cs_high_school_computer_science
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    mmlu_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("Hindi Benchmark - Start performance evaluation on the MMLU Machine Learning Benchmark")
    dataset = "hindi_mmlu_cs_machine_learning"
    test_path = hindi_mmlu_cs_machine_learning
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    mmlu_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("Hindi Benchmark - Start performance evaluation on the MMLU Abstract Algebra Benchmark")
    dataset = "hindi_mmlu_math_abstract_algebra"
    test_path = hindi_mmlu_math_abstract_algebra
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    mmlu_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("Hindi Benchmark - Start performance evaluation on the MMLU College Mathematics Benchmark")
    dataset = "hindi_mmlu_math_college_mathematics"
    test_path = hindi_mmlu_math_college_mathematics
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    mmlu_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("Hindi Benchmark - Start performance evaluation on the MMLU Elementary Mathematics Benchmark")
    dataset = "hindi_mmlu_math_elementary_mathematics"
    test_path = hindi_mmlu_math_elementary_mathematics
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    mmlu_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("Hindi Benchmark - Start performance evaluation on the MMLU High School Mathematics Benchmark")
    dataset = "hindi_mmlu_math_high_school_mathematics"
    test_path = hindi_mmlu_math_high_school_mathematics
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    mmlu_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("Hindi Benchmark - Start performance evaluation on the MMLU High School Statistics Benchmark")
    dataset = "hindi_mmlu_math_high_school_statistics"
    test_path = hindi_mmlu_math_high_school_statistics
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    mmlu_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("Hindi Benchmark - Start performance evaluation on the MMLU Electrical Engineering Benchmark")
    dataset = "hindi_mmlu_engineering_electrical_engineering"
    test_path = hindi_mmlu_engineering_electrical_engineering
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    mmlu_eval(config=config, mode=mode, tokenizer=tokenizer, file_path=file_path, data_path=test_path)
