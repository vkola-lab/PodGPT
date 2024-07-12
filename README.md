<h1 align="center">MedPodGPT</h1>
<h4 align="center">Benchmarking Multilingual Medical Large Language Models (LLMs)</h4>
<p align="center">
  <a href="https://github.com/vkola-lab/medpodgpt"> <img width="250px" src="MedPodGPT.png"></a> 
  <br />
  <br />
  <a href="https://img.shields.io/badge/Code%20License-MIT-green.svg"><img alt="CODE_LICENSE" src="https://img.shields.io/badge/Code%20License-MIT-green.svg" /></a>
  <a href="https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red.svg"><img alt="DATA_LICENSE" src="https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red.svg" /></a>
  <a href="https://www.apache.org/licenses/LICENSE-2.0"><img alt="Model Weight License" src="https://img.shields.io/badge/Model%20Weights%20License-Apache-yellow" /></a>
  <a href="https://www.python.org/downloads/release/python-3100/"><img alt="Python 3.10" src="https://img.shields.io/badge/python-3.10-blue.svg" /></a>
</p>

# 🎉 Announcements
[2024.7.12] ![](news.gif) Our AI Platform [MedPodGPT](https://medpodgpt.org/) is available to authorized users. 
It is an online platform for deploying our latest multimodal foundation models for medical and clinical applications.
Please try it out if you are interested!

# 📚 Table of Contents
- [Installation](#Installation)
- [Quick Start](#Quick-Start)
  - [Train Lightweight Models](#Train-Lightweight-Models)
  - [Train Heavy Models](#Train-Heavy-Models)
  - [Train Quantized Models](#Train-Quantized-Models)
- [Performance Evaluation](#Performance-Evaluation)
  - [Single GPU For Lightweight Models](#Single-GPU-For-Lightweight-Models)
  - [Distributed GPUs For Heavy Models](#Distributed-GPUs-For-Heavy-Models)
- [Automatic Speech Recognition](#Automatic-Speech-Recognition)
- [Dataset Builder](#Dataset-Builder)
- [Upload and Download Models](#Upload-and-Download-Models)
- [Structure of the Code](#Structure-of-the-Code)
- [Citation](#Citation)
- [Contact](#Contact)
- [Contribution](#Contribution)
- [Acknowledgement](#Acknowledgement)

# Installation
```shell
pip install -r requirements.txt
```

# Quick Start
## Train Lightweight Models
For lightweight models (2B, 7B, and 8B), we optimize the entire model. 
Please check and setup hyper-parameters in [config_small.yml](https://github.com/vkola-lab/medpodgpt/blob/main/config_small.yml).
```shell
python main_small.py
```

## Train Heavy Models
For lager and heavy models (>8B), we optimize the Low-rank Adapter (LoRA).
Please check and setup hyper-parameters in [config_small.yml](https://github.com/vkola-lab/medpodgpt/blob/main/config_large.yml).
```shell
python main_large.py
```

## Train Quantized Models
We also provide support for quantizing larger models using the [GPTQ](https://arxiv.org/abs/2210.17323) algorithm and then optimizing the LoRA.
The large models can be deployed on consumer GPUs after quantization.
```shell
python quantization.py "./save_folder" "./gptq_model" "medical" --bits 4 --group_size 128 --desc_act 1 --dtype float16
```
Then, we need to upload the model to Hugging Face,
```shell
python upload_quantized_model.py --repo "shuyuej/MedLLaMA3-70B-BASE-MODEL-QUANT" --folder_path "./gptq_model"
```
Lastly, we optimize the LoRA module,
```shell
python main_quantization.py
```

# Performance Evaluation
we use `inference_pretrain.py` and `inference_single_model.py` for larger models (> 8B) 
and `inference_sequential.py` for smaller models (2B/7B/8B).

First, in the project home directory, please copy and paste the inference files,
```shell
cp -r ./inference/inference_sequential.py ./
cp -r ./inference/inference_large.sh ./
cp -r ./inference/inference_pretrain.py ./
cp -r ./inference/inference_single_model.py ./
```

### Single GPU For Lightweight Models
#### inference_sequential.py
**Sequentially** evaluate the performance of multiple checkpoints (models).<br>
Please note that we use `--eval_pretrain` to indicate whether to evaluate the original pre-trained model.
```shell
python inference_sequential.py --eval_pretrain True --id 35166 52749 70332 87915
```

### Distributed GPUs For Heavy Models
**Sequentially** evaluate the performance of the original pre-trained model and all the checkpoints.<br>
Special Notice: Please change the checkpoint IDs and CUDA_VISIBLE_DEVICES in the `inference_large.sh` file.
```shell
sh inference_large.sh
```

#### inference_pretrain.py
**Only** evaluate the performance of the original pre-trained model.
```shell
python inference_pretrain.py
```

#### inference_single_model.py
**Only** evaluate the performance of a single checkpoint (model).<br>
Please note that `--id` is the checkpoint id.
```shell
python inference_single_model.py --id 35166
```

# Automatic Speech Recognition
In the [scripts folder](https://github.com/vkola-lab/medpodgpt/tree/main/scripts), 
we provide Automatic Speech Recognition (ASR) support.
```shell
python audio2text.py
```

# Dataset Builder
We used the following codes to pre-process our transcripts and generate training dataset.
Please check [these lines](https://github.com/vkola-lab/medpodgpt/blob/main/scripts/database_builder.py#L236-L242) for different languages support. 
```shell
python database_builder.py
```
```shell
python merge_database.py
```

# Upload and Download Models
In the [scripts folder](https://github.com/vkola-lab/medpodgpt/tree/main/scripts), 
we offer support for both uploading and downloading models.

To upload your checkpoints to your Hugging Face model repo,
```shell
python upload_model.py --repo "shuyuej/DrGemma2B" --id 35166 52749 70332 87915
```

To download your model or files from Hugging Face,
```shell
python download_model.py --repo "shuyuej/DrGemma2B" --repo_type "model" --save_dir "./save_folder"
```

# Structure of the Code
At the root of the project, you will see:

```text
├── config_chatgpt.yml
├── config_large.yml
├── config_quantization.yml
├── config_small.yml
├── main_large.py
├── main_quantization.py
├── main_small.py
├── lib
│   ├── README.md
│   ├── data_manager.py
│   ├── evaluation_chatgpt.py
│   ├── evaluation_large.py
│   ├── evaluation_small.py
│   ├── model_loader_large.py
│   ├── model_loader_quantization.py
│   └── model_loader_small.py
├── inference
│   ├── README.md
│   ├── inference_chatgpt.py
│   ├── inference_large.sh
│   ├── inference_pretrain.py
│   ├── inference_sequential.py
│   └── inference_single_model.py
├── download_files
│   ├── README.md
│   ├── download_model_from_hf.py
│   └── download_model_to_local.py
├── quantization
│   ├── README.md
│   ├── quantization.py
│   └── upload_quantization_model.py
├── scripts
│   ├── README.md
│   ├── audio2text.py
│   ├── database_builder.py
│   ├── download_model.py
│   └── merge_database.py
├── setup.cfg
├── upload_model.py
├── requirements.txt
├── benchmark
│   ├── README.md
│   ├── chinese_cmmlu
│   │   ├── anatomy.csv
│   │   ├── clinical_knowledge.csv
│   │   ├── college_medicine.csv
│   │   ├── genetics.csv
│   │   ├── nutrition.csv
│   │   ├── traditional_chinese_medicine.csv
│   │   └── virology.csv
│   ├── chinese_mcmle
│   │   └── MedQA-MCMLE.jsonl
│   ├── english_medexpqa
│   │   └── test.en.casimedicos.rag.jsonl
│   ├── english_medmcqa
│   │   └── MedMCQA_test.json
│   ├── english_medqa
│   │   └── MedQA_USMLE_test.jsonl
│   ├── english_mmlu
│   │   ├── anatomy_test.csv
│   │   ├── clinical_knowledge_test.csv
│   │   ├── college_biology_test.csv
│   │   ├── college_medicine_test.csv
│   │   ├── medical_genetics_test.csv
│   │   └── professional_medicine_test.csv
│   ├── english_pubmedqa
│   │   └── PubMedQA_test.json
│   ├── english_usmle
│   │   ├── USMLE_STEP_1.json
│   │   ├── USMLE_STEP_2.json
│   │   ├── USMLE_STEP_3.json
│   │   └── USMLE_ethics.json
│   ├── french_medexpqa
│   │   └── test.fr.casimedicos.rag.jsonl
│   ├── french_medmcqa
│   │   └── FrenchMedMCQA-test.json
│   ├── french_mmlu
│   │   ├── mmlu_French_test_anatomy_test.csv
│   │   ├── mmlu_French_test_clinical_knowledge_test.csv
│   │   ├── mmlu_French_test_college_biology_test.csv
│   │   ├── mmlu_French_test_college_medicine_test.csv
│   │   ├── mmlu_French_test_medical_genetics_test.csv
│   │   └── mmlu_French_test_professional_medicine_test.csv
│   ├── hindi_mmlu
│   │   ├── mmlu_Hindi_test_anatomy_test.csv
│   │   ├── mmlu_Hindi_test_clinical_knowledge_test.csv
│   │   ├── mmlu_Hindi_test_college_biology_test.csv
│   │   ├── mmlu_Hindi_test_college_medicine_test.csv
│   │   ├── mmlu_Hindi_test_medical_genetics_test.csv
│   │   └── mmlu_Hindi_test_professional_medicine_test.csv
│   ├── spanish_headqa
│   │   └── HEAD-QA-test.json
│   ├── spanish_medexpqa
│   │   └── test.es.casimedicos.rag.jsonl
│   └── spanish_mmlu
│       ├── mmlu_Spanish_test_anatomy_test.csv
│       ├── mmlu_Spanish_test_clinical_knowledge_test.csv
│       ├── mmlu_Spanish_test_college_biology_test.csv
│       ├── mmlu_Spanish_test_college_medicine_test.csv
│       ├── mmlu_Spanish_test_medical_genetics_test.csv
│       └── mmlu_Spanish_test_professional_medicine_test.csv
└── utils
    ├── README.md
    ├── answer_utils.py
    ├── benchmark_utils.py
    ├── eval_chatgpt_utils.py
    ├── eval_large_utils.py
    ├── eval_small_utils.py
    ├── test_extraction_chinese.py
    ├── test_extraction_english.py
    ├── test_extraction_french.py
    ├── test_extraction_hindi.py
    ├── test_extraction_spanish.py
    └── utils.py
```

# Citation
If you find our work useful in your research, please consider citing it in your publications. We provide a BibTeX entry below.

```bibtex
@article{jia2024medpodgpt,
  title   = {{MedPodGPT}: A Multilingual Audio-augmented Large Language Model for Medical Research and Education},
  author  = {Shuyue Jia, Subhrangshu Bit, Edward Searls, Lindsey A. Claus, Pengrui Fan, Varuna H. Jasodanand, Meagan V. Lauber, Divya Veerapaneni, William M. Wang, Rhoda Au, Vijaya B. Kolachalama},
  journal = {medRxiv}
  year    = {2024},
}
```

# Contact
**Core Contributor and Maintainer**: <br>
- [Shuyue Jia](https://github.com/SuperBruceJia)
- [Subhrangshu Bit](https://github.com/SubhrangshuBit)
- [Edward Searls](https://github.com/nsearls-bu)
- [Pengrui Fan](https://github.com/pengrui26)
- [William M. Wang](https://github.com/bomas7)

If you have any questions, please drop us an email at [brucejia@bu.edu](brucejia@bu.edu), [sbit@bu.edu](sbit@bu.edu), and [nsearls@bu.edu](nsearls@bu.edu).

# Contribution
We always welcome contributions to help make **MedPodGPT** Library better. 
If you would like to contribute, please submit a [pull request](https://github.com/vkola-lab/medpodgpt/pulls).

# Acknowledgement
The **MedPodGPT** Library is created and maintained by the Kolachalama Lab at Boston University.

<a href="https://www.bu.edu/"> <img width="250" src="https://raw.githubusercontent.com/SuperBruceJia/promptcraft/main/bu.png"></a>
