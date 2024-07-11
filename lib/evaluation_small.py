# coding=utf-8
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
# MMLU
# We conducted performance evaluation on medicine-related 6 databases
# anatomy_test, clinical_knowledge_test, college_biology_test
# college_medicine_test, medical_genetics_test, professional_medicine_test
# Paper: Measuring Massive Multitask Language Understanding
# Link: https://arxiv.org/abs/2009.03300
# Database and Codes: https://www.kaggle.com/datasets/lizhecheng/mmlu-dataset
#
# MedExpQA English
# We conducted performance evaluation on the MedExpQA (English) Benchmark
# Paper: MedExpQA: Multilingual Benchmarking of Large Language Models for Medical Question Answering
# Link: https://arxiv.org/abs/2404.05590
# Database and Codes: https://huggingface.co/datasets/HiTZ/MedExpQA
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
# Spanish Benchmarks
# HEAD-QA
# We conducted performance evaluation on the HEAD-QA Benchmark
# Paper: HEAD-QA: A Healthcare Dataset for Complex Reasoning
# Link: https://arxiv.org/abs/1906.04701
# Homepage: https://aghie.github.io/head-qa/
# Database and Codes: https://github.com/aghie/head-qa
#
# MMLU Spanish
# We conducted performance evaluation on medicine-related 6 databases
# Clinical knowledge, Medical genetics, Anatomy,
# Professional medicine, College biology, and College medicine.
# Paper: Apollo: An Lightweight Multilingual Medical LLM towards Democratizing Medical AI to 6B People
# Link: https://arxiv.org/abs/2403.03640
# Database and Codes: https://huggingface.co/datasets/FreedomIntelligence/MMLU_Spanish
#
# MedExpQA Spanish
# We conducted performance evaluation on the MedExpQA (Spanish) Benchmark
# Paper: MedExpQA: Multilingual Benchmarking of Large Language Models for Medical Question Answering
# Link: https://arxiv.org/abs/2404.05590
# Database and Codes: https://huggingface.co/datasets/HiTZ/MedExpQA
#
# Hindi Benchmarks
# MMLU Hindi
# We conducted performance evaluation on medicine-related 6 databases
# Clinical knowledge, Medical genetics, Anatomy,
# Professional medicine, College biology, and College medicine.
# Paper: Apollo: An Lightweight Multilingual Medical LLM towards Democratizing Medical AI to 6B People
# Link: https://arxiv.org/abs/2403.03640
# Database and Codes: https://huggingface.co/datasets/FreedomIntelligence/MMLU_Hindi

import os

from transformers import AutoTokenizer

from lib.model_loader_small import download_pretrained
from utils.eval_small_utils import *


def evaluation(config, eval_pretrain=False, checkpoint_id=None):
    """
    Conduct model inference, get the model's responses, and evaluate performance
    :param config: the configuration file
    :param eval_pretrain: is this evaluating the pre-trained model?
    :param checkpoint_id: checkpoint ID
    """
    # Retrieve the pathes of needed hyperparameters
    model_name = config.get("model_name")
    result_dir = config.get("result_dir")

    if eval_pretrain:
        # Initialize model and tokenizer
        print("Download the pretrained model and tokenizer!")
        download_pretrained(config)
        checkpoint_path = "Original-Model"  # File name for the pre-trained model
    else:
        # Load the checkpoint path
        checkpoint_path = "checkpoint-" + str(checkpoint_id)
        config['save_dir'] = "./save_folder/" + checkpoint_path

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # English Benchmarks
    english_medqa = config.get("english_medqa")
    english_pubmedqa = config.get("english_pubmedqa")
    english_medmcqa = config.get("english_medmcqa")
    english_usmle_step1 = config.get("english_usmle_step1")
    english_usmle_step2 = config.get("english_usmle_step2")
    english_usmle_step3 = config.get("english_usmle_step3")
    english_usmle_ethics = config.get("english_usmle_ethics")
    english_mmlu_anatomy = config.get("english_mmlu_anatomy")
    english_mmlu_clinical_knowledge = config.get("english_mmlu_clinical_knowledge")
    english_mmlu_college_biology = config.get("english_mmlu_college_biology")
    english_mmlu_college_medicine = config.get("english_mmlu_college_medicine")
    english_mmlu_medical_genetics = config.get("english_mmlu_medical_genetics")
    english_mmlu_professional_medicine = config.get("english_mmlu_professional_medicine")
    english_medexpqa = config.get("english_medexpqa")

    # Chinese Benchmarks
    chinese_mcmle = config.get("chinese_mcmle")
    chinese_cmmlu_anatomy = config.get("chinese_cmmlu_anatomy")
    chinese_cmmlu_clinical_knowledge = config.get("chinese_cmmlu_clinical_knowledge")
    chinese_cmmlu_college_medicine = config.get("chinese_cmmlu_college_medicine")
    chinese_cmmlu_genetics = config.get("chinese_cmmlu_genetics")
    chinese_cmmlu_nutrition = config.get("chinese_cmmlu_nutrition")
    chinese_cmmlu_tcm = config.get("chinese_cmmlu_tcm")
    chinese_cmmlu_virology = config.get("chinese_cmmlu_virology")

    # French Benchmarks
    french_medmcqa = config.get("french_medmcqa")
    french_mmlu_anatomy = config.get("french_mmlu_anatomy")
    french_mmlu_clinical_knowledge = config.get("french_mmlu_clinical_knowledge")
    french_mmlu_college_biology = config.get("french_mmlu_college_biology")
    french_mmlu_college_medicine = config.get("french_mmlu_college_medicine")
    french_mmlu_medical_genetics = config.get("french_mmlu_medical_genetics")
    french_mmlu_professional_medicine = config.get("french_mmlu_professional_medicine")
    french_medexpqa = config.get("french_medexpqa")

    # Spanish Benchmarks
    spanish_headqa = config.get("spanish_headqa")
    spanish_mmlu_anatomy = config.get("spanish_mmlu_anatomy")
    spanish_mmlu_clinical_knowledge = config.get("spanish_mmlu_clinical_knowledge")
    spanish_mmlu_college_biology = config.get("spanish_mmlu_college_biology")
    spanish_mmlu_college_medicine = config.get("spanish_mmlu_college_medicine")
    spanish_mmlu_medical_genetics = config.get("spanish_mmlu_medical_genetics")
    spanish_mmlu_professional_medicine = config.get("spanish_mmlu_professional_medicine")
    spanish_medexpqa = config.get("spanish_medexpqa")

    # Hindi Benchmarks
    hindi_mmlu_anatomy = config.get("hindi_mmlu_anatomy")
    hindi_mmlu_clinical_knowledge = config.get("hindi_mmlu_clinical_knowledge")
    hindi_mmlu_college_biology = config.get("hindi_mmlu_college_biology")
    hindi_mmlu_college_medicine = config.get("hindi_mmlu_college_medicine")
    hindi_mmlu_medical_genetics = config.get("hindi_mmlu_medical_genetics")
    hindi_mmlu_professional_medicine = config.get("hindi_mmlu_professional_medicine")

    # English Benchmarks
    print("English Benchmark - Start performance evaluation on the MedQA Benchmark")
    dataset = "english_medqa"
    test_path = english_medqa
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    medqa_eval(config=config, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("English Benchmark - Start performance evaluation on the PubMedQA Benchmark")
    dataset = "english_pubmedqa"
    test_path = english_pubmedqa
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    pubmedqa_eval(config=config, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("English Benchmark - Start performance evaluation on the MedMCQA Benchmark")
    dataset = "english_medmcqa"
    test_path = english_medmcqa
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    medmcqa_eval(config=config, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("English Benchmark - Start performance evaluation on the Internal USMLE STEP 1 Benchmark")
    dataset = "english_usmle_step1"
    test_path = english_usmle_step1
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    usmle_eval(config=config, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("English Benchmark - Start performance evaluation on the Internal USMLE STEP 2 Benchmark")
    dataset = "english_usmle_step2"
    test_path = english_usmle_step2
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    usmle_eval(config=config, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("English Benchmark - Start performance evaluation on the Internal USMLE STEP 3 Benchmark")
    dataset = "english_usmle_step3"
    test_path = english_usmle_step3
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    usmle_eval(config=config, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("English Benchmark - Start performance evaluation on the Internal USMLE Ethics Benchmark")
    dataset = "english_usmle_ethics"
    test_path = english_usmle_ethics
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    usmle_eval(config=config, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("English Benchmark - Start performance evaluation on the MMLU Anatomy Benchmark")
    dataset = "english_mmlu_anatomy"
    test_path = english_mmlu_anatomy
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    mmlu_eval(config=config, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("English Benchmark - Start performance evaluation on the MMLU Clinical Knowledge Benchmark")
    dataset = "english_mmlu_clinical_knowledge"
    test_path = english_mmlu_clinical_knowledge
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    mmlu_eval(config=config, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("English Benchmark - Start performance evaluation on the MMLU College Biology Benchmark")
    dataset = "english_mmlu_college_biology"
    test_path = english_mmlu_college_biology
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    mmlu_eval(config=config, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("English Benchmark - Start performance evaluation on the MMLU College Medicine Benchmark")
    dataset = "english_mmlu_college_medicine"
    test_path = english_mmlu_college_medicine
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    mmlu_eval(config=config, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("English Benchmark - Start performance evaluation on the MMLU Medical Genetics Benchmark")
    dataset = "english_mmlu_medical_genetics"
    test_path = english_mmlu_medical_genetics
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    mmlu_eval(config=config, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("English Benchmark - Start performance evaluation on the MMLU Professional Medicine Benchmark")
    dataset = "english_mmlu_professional_medicine"
    test_path = english_mmlu_professional_medicine
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    mmlu_eval(config=config, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("English Benchmark - Start performance evaluation on the MedExpQA English Benchmark")
    dataset = "english_medexpqa"
    test_path = english_medexpqa
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    medexpqa_eval(config=config, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    # Mandarin Chinese Benchmarks
    print("Mandarin Chinese Benchmark - Start performance evaluation on the MedQA-MCMLE Benchmark")
    dataset = "chinese_mcmle"
    test_path = chinese_mcmle
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    mcmle_eval(config=config, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("Mandarin Chinese Benchmark - Start performance evaluation on the CMMLU Anatomy Benchmark")
    dataset = "chinese_cmmlu_anatomy"
    test_path = chinese_cmmlu_anatomy
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    cmmlu_eval(config=config, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("Mandarin Chinese Benchmark - Start performance evaluation on the CMMLU Clinical Knowledge Benchmark")
    dataset = "chinese_cmmlu_clinical_knowledge"
    test_path = chinese_cmmlu_clinical_knowledge
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    cmmlu_eval(config=config, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("Mandarin Chinese Benchmark - Start performance evaluation on the CMMLU College Medicine Benchmark")
    dataset = "chinese_cmmlu_college_medicine"
    test_path = chinese_cmmlu_college_medicine
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    cmmlu_eval(config=config, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("Mandarin Chinese Benchmark - Start performance evaluation on the CMMLU Genetics Benchmark")
    dataset = "chinese_cmmlu_genetics"
    test_path = chinese_cmmlu_genetics
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    cmmlu_eval(config=config, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("Mandarin Chinese Benchmark - Start performance evaluation on the CMMLU Nutrition Benchmark")
    dataset = "chinese_cmmlu_nutrition"
    test_path = chinese_cmmlu_nutrition
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    cmmlu_eval(config=config, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("Mandarin Chinese Benchmark - Start performance evaluation on the CMMLU TCM Benchmark")
    dataset = "chinese_cmmlu_tcm"
    test_path = chinese_cmmlu_tcm
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    cmmlu_eval(config=config, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("Mandarin Chinese Benchmark - Start performance evaluation on the CMMLU Virology Benchmark")
    dataset = "chinese_cmmlu_virology"
    test_path = chinese_cmmlu_virology
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    cmmlu_eval(config=config, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    # French Benchmarks
    print("French Benchmark - Start performance evaluation on the FrenchMedMCQA Benchmark")
    dataset = "french_frenchmedmcqa"
    test_path = french_medmcqa
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    frenchmedmcqa_eval(config=config, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("French Benchmark - Start performance evaluation on the French MMLU Anatomy Benchmark")
    dataset = "french_mmlu_anatomy"
    test_path = french_mmlu_anatomy
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    mmlu_eval(config=config, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("French Benchmark - Start performance evaluation on the French MMLU Clinical Knowledge Benchmark")
    dataset = "french_mmlu_clinical_knowledge"
    test_path = french_mmlu_clinical_knowledge
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    mmlu_eval(config=config, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("French Benchmark - Start performance evaluation on the French MMLU College Biology Benchmark")
    dataset = "french_mmlu_college_biology"
    test_path = french_mmlu_college_biology
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    mmlu_eval(config=config, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("French Benchmark - Start performance evaluation on the French MMLU College Medicine Benchmark")
    dataset = "french_mmlu_college_medicine"
    test_path = french_mmlu_college_medicine
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    mmlu_eval(config=config, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("French Benchmark - Start performance evaluation on the French MMLU Medical Genetics Benchmark")
    dataset = "french_mmlu_medical_genetics"
    test_path = french_mmlu_medical_genetics
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    mmlu_eval(config=config, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("French Benchmark - Start performance evaluation on the French MMLU Professional Medicine Benchmark")
    dataset = "french_mmlu_professional_medicine"
    test_path = french_mmlu_professional_medicine
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    mmlu_eval(config=config, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("French Benchmark - Start performance evaluation on the MedExpQA French Benchmark")
    dataset = "french_medexpqa"
    test_path = french_medexpqa
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    medexpqa_eval(config=config, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    # Spanish Benchmarks
    print("Spanish Benchmark - Start performance evaluation on the HEAD-QA Benchmark")
    dataset = "spanish_headqa"
    test_path = spanish_headqa
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    headqa_eval(config=config, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("Spanish Benchmark - Start performance evaluation on the Spanish MMLU Anatomy Benchmark")
    dataset = "spanish_mmlu_anatomy"
    test_path = spanish_mmlu_anatomy
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    mmlu_eval(config=config, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("Spanish Benchmark - Start performance evaluation on the Spanish MMLU Clinical Knowledge Benchmark")
    dataset = "spanish_mmlu_clinical_knowledge"
    test_path = spanish_mmlu_clinical_knowledge
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    mmlu_eval(config=config, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("Spanish Benchmark - Start performance evaluation on the Spanish MMLU College Biology Benchmark")
    dataset = "spanish_mmlu_college_biology"
    test_path = spanish_mmlu_college_biology
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    mmlu_eval(config=config, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("Spanish Benchmark - Start performance evaluation on the Spanish MMLU College Medicine Benchmark")
    dataset = "spanish_mmlu_college_medicine"
    test_path = spanish_mmlu_college_medicine
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    mmlu_eval(config=config, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("Spanish Benchmark - Start performance evaluation on the Spanish MMLU Medical Genetics Benchmark")
    dataset = "spanish_mmlu_medical_genetics"
    test_path = spanish_mmlu_medical_genetics
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    mmlu_eval(config=config, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("Spanish Benchmark - Start performance evaluation on the Spanish MMLU Professional Medicine Benchmark")
    dataset = "spanish_mmlu_professional_medicine"
    test_path = spanish_mmlu_professional_medicine
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    mmlu_eval(config=config, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("Spanish Benchmark - Start performance evaluation on the MedExpQA Spanish Benchmark")
    dataset = "spanish_medexpqa"
    test_path = spanish_medexpqa
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    medexpqa_eval(config=config, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    # Hindi Benchmarks
    print("Hindi Benchmark - Start performance evaluation on the Hindi MMLU Anatomy Benchmark")
    dataset = "hindi_mmlu_anatomy"
    test_path = hindi_mmlu_anatomy
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    mmlu_eval(config=config, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("Hindi Benchmark - Start performance evaluation on the Hindi MMLU Clinical Knowledge Benchmark")
    dataset = "hindi_mmlu_clinical_knowledge"
    test_path = hindi_mmlu_clinical_knowledge
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    mmlu_eval(config=config, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("Hindi Benchmark - Start performance evaluation on the Hindi MMLU College Biology Benchmark")
    dataset = "hindi_mmlu_college_biology"
    test_path = hindi_mmlu_college_biology
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    mmlu_eval(config=config, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("Hindi Benchmark - Start performance evaluation on the Hindi MMLU College Medicine Benchmark")
    dataset = "hindi_mmlu_college_medicine"
    test_path = hindi_mmlu_college_medicine
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    mmlu_eval(config=config, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("Hindi Benchmark - Start performance evaluation on the Hindi MMLU Medical Genetics Benchmark")
    dataset = "hindi_mmlu_medical_genetics"
    test_path = hindi_mmlu_medical_genetics
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    mmlu_eval(config=config, tokenizer=tokenizer, file_path=file_path, data_path=test_path)

    print("Hindi Benchmark - Start performance evaluation on the Hindi MMLU Professional Medicine Benchmark")
    dataset = "hindi_mmlu_professional_medicine"
    test_path = hindi_mmlu_professional_medicine
    os.makedirs(result_dir + "/" + dataset, exist_ok=True)
    file_path = result_dir + "/" + dataset + "/" + checkpoint_path + ".json"
    mmlu_eval(config=config, tokenizer=tokenizer, file_path=file_path, data_path=test_path)
