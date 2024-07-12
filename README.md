<h1 align="center">MedPodGPT</h1>
<h4 align="center">Benchmarking Multilingual Medical Large Language Models (LLMs)</h4>
<p align="center">
  <a href="https://github.com/vkola-lab/medpodgpt"> <img width="250px" src="figures/MedPodGPT.png"></a> 
  <br />
  <br />
  <a href="https://img.shields.io/badge/Code%20License-MIT-green.svg"><img alt="CODE_LICENSE" src="https://img.shields.io/badge/Code%20License-MIT-green.svg" /></a>
  <a href="https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red.svg"><img alt="DATA_LICENSE" src="https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red.svg" /></a>
  <a href="https://www.apache.org/licenses/LICENSE-2.0"><img alt="Model Weight License" src="https://img.shields.io/badge/Model%20Weights%20License-Apache-yellow" /></a>
  <a href="https://www.python.org/downloads/release/python-3100/"><img alt="Python 3.10" src="https://img.shields.io/badge/python-3.10-blue.svg" /></a>
</p>

# 🎉 Announcements
[2024.7.12] ![](figures/news.gif) Our AI Platform [MedPodGPT](https://medpodgpt.org/) is available to authorized users. 
It is an online platform for deploying our latest multimodal foundation models for medical and clinical applications.
Please try it out if you are interested!

[2024.7.12] We are releasing a new benchmark encompassing the latest USMLE Step 1, Step 2, Step 3, and Ethics 
to further advance the filed.
Check our database [here](https://github.com/vkola-lab/medpodgpt/tree/main/benchmark/english_usmle).

# 📚 Table of Contents
- [Installation](#Installation)
- [Quick Start](#Quick-Start)
  - [Train Lightweight Models](#Train-Lightweight-Models)
  - [Train Heavy Models](#Train-Heavy-Models)
  - [Train Quantized Large Models](#Train-Quantized-Large-Models)
- [Performance Evaluation](#Performance-Evaluation)
  - [Single GPU For Lightweight Models](#Single-GPU-For-Lightweight-Models)
  - [Distributed GPUs For Heavy Models](#Distributed-GPUs-For-Heavy-Models)
  - [OpenAI ChatGPT Support](#OpenAI-ChatGPT-Support)
- [Benchmarks and Results](#Benchmarks-and-Results)
  - [Benchmarks Description](#Benchmarks-Description)
  - [Performance on In-domain Benchmarks](#Performance-on-In-domain-Benchmarks)
  - [Zero-shot Cross-lingual Performance](#Zero-shot-Cross-lingual-Performance)
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
Please check and setup hyper-parameters in [config_large.yml](https://github.com/vkola-lab/medpodgpt/blob/main/config_large.yml).
```shell
python main_large.py
```

## Train Quantized Large Models
We also provide support for quantizing larger models, _e.g._, LLaMA 3 70B model, using the [GPTQ](https://arxiv.org/abs/2210.17323) algorithm and then optimizing the LoRA.
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
we use [inference_pretrain.py](https://github.com/vkola-lab/medpodgpt/blob/main/inference/inference_pretrain.py) 
and [inference_single_model.py](https://github.com/vkola-lab/medpodgpt/blob/main/inference/inference_single_model.py)
for larger models (>8B) 
and [inference_sequential.py](https://github.com/vkola-lab/medpodgpt/blob/main/inference/inference_sequential.py) 
for smaller models (2B/7B/8B). 
Please check [here](https://github.com/vkola-lab/medpodgpt/tree/main/inference) for more information.

### Single GPU For Lightweight Models
#### inference_sequential.py
**Sequentially** evaluate the performance of multiple checkpoints (models).<br>
Please note that we use `--eval_pretrain` to indicate whether to evaluate the original pre-trained model.
```shell
python inference_sequential.py --eval_pretrain True --id 35166 52749 70332 87915
```

### Distributed GPUs For Heavy Models
**Sequentially** evaluate the performance of the original pre-trained model and all the checkpoints.<br>
Special Notice: Please change the `checkpoint IDs` and `CUDA_VISIBLE_DEVICES` 
in the [inference_large.sh](https://github.com/vkola-lab/medpodgpt/blob/main/inference/inference_large.sh) file.
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

### OpenAI ChatGPT Support
We also offer support for running OpenAI ChatGPT inference using API.
Please enter your OpenAI API Key [here](https://github.com/vkola-lab/medpodgpt/blob/main/config_chatgpt.yml#L18).
```shell
python inference_chatgpt.py
```

# Benchmarks and Results
## Benchmarks Description

| **Language**  |     **Benchmark Datasets**     | **Description** |
|:-------------:|:------------------------------:|:---------------:|
| **English** 	 |            MedExpQA            |                 |
|       	       |        MedMCQA        	        |                 |  
|       	       |        MedQA         	         |                 |           
|       	       |        PubMedQA       	        |                 |             
|       	       |        Anatomy        	        |                 |            
|       	       |      College Biology    	      |        	        |             
|       	       |      College Medicine   	      |        	        |             
|       	       |      Medical Genetics   	      |        	        |             
|       	       |    Professional Medicine 	     |        	        |           
| **French**  	 |      FrenchMedMCQA     	       |        	        |           
|       	       |        MedExpQA       	        |        	        |            
|       	       |        Anatomy        	        |        	        |           
|       	       |     Clinical Knowledge  	      |        	        |           
|       	       |      College Biology    	      |        	        |           
|       	       |      College Medicine   	      |        	        |            
|       	       |      Medical Genetics   	      |        	        |            
|       	       |    Professional Medicine 	     |        	        |            
| **Spanish** 	 |        HeadQA        	         |        	        |            
|       	       |        MedExpQA       	        |        	        |            
|       	       |        Anatomy        	        |        	        |            
|       	       |     Clinical Knowledge  	      |        	        |            
|       	       |      College Biology    	      |        	        |           
|       	       |      College Medicine   	      |        	        |           
|       	       |      Medical Genetics   	      |        	        |           
|       	       |    Professional Medicine 	     |        	        |            
| **Chinese** 	 |     MedQA-MCMLE         	      |                 |           
|       	       |      Anatomy           	       |        	        |           
|       	       |   Clinical Knowledge      	    |        	        |            
|       	       |    College Medicine       	    |        	        |           
|       	       |    Medical Genetics       	    |        	        |           
|       	       |    Medical Nutrition      	    |        	        |           
|       	       | Traditional Chinese Medicine 	 |        	        |           
|       	       |      Virology           	      |        	        |           
| **Hindi**  	  |      Anatomy           	       |        	        |           
|       	       |   Clinical Knowledge      	    |        	        |           
|       	       |    College Biology       	     |        	        |           
|       	       |    College Medicine       	    |        	        |           
|       	       |    Medical Genetics       	    |        	        |           
|       	       |   Professional Medicine    	   |        	        |           

## Performance on In-domain Benchmarks
<p align="center">
  <a href="https://github.com/vkola-lab/medpodgpt"> <img src="figures/Table-2.png"></a> 
</p>

## Zero-shot Cross-lingual Performance
<p align="center">
  <a href="https://github.com/vkola-lab/medpodgpt"> <img src="figures/Table-3.png"></a> 
</p>

# Automatic Speech Recognition
In the [scripts folder](https://github.com/vkola-lab/medpodgpt/tree/main/scripts), 
we provide Automatic Speech Recognition (ASR) support.
```shell
python audio2text.py
```

# Dataset Builder
We used the following codes to pre-process our transcripts and generate training dataset.
Please check [these lines](https://github.com/vkola-lab/medpodgpt/blob/main/scripts/database_builder.py#L236-L242) 
for different languages support.
```shell
python database_builder.py
```
```shell
python merge_database.py
```

# Upload and Download Models
In the [scripts folder](https://github.com/vkola-lab/medpodgpt/tree/main/scripts), 
we offer support for both uploading and downloading models.

To upload your checkpoints to Hugging Face model repo,
```shell
python upload_model.py --repo "shuyuej/DrGemma2B" --id 35166 52749 70332 87915
```

To download your model or files from Hugging Face repo,
```shell
python download_model.py --repo "shuyuej/DrGemma2B" --repo_type "model" --save_dir "./save_folder"
```

# Structure of the Code
At the root of the project, you will see:

```text
├── requirements.txt
├── main_small.py
├── main_large.py
├── main_quantization.py
├── config_small.yml
├── config_large.yml
├── config_quantization.yml
├── config_chatgpt.yml
├── lib
│   ├── data_manager.py
│   ├── model_loader_small.py
│   ├── model_loader_large.py
│   ├── model_loader_quantization.py
│   ├── evaluation_small.py
│   ├── evaluation_large.py
│   └── evaluation_chatgpt.py
├── inference
│   ├── inference_large.sh
│   ├── inference_chatgpt.py
│   ├── inference_pretrain.py
│   ├── inference_sequential.py
│   └── inference_single_model.py
├── download_files
│   ├── download_model_from_hf.py
│   └── download_model_to_local.py
├── quantization
│   ├── quantization.py
│   └── upload_quantized_model.py
├── scripts
│   ├── audio2text.py
│   ├── download_model.py
│   ├── upload_model.py
│   ├── database_builder.py
│   └── merge_database.py
├── benchmark
│   ├── chinese_cmmlu
│   ├── chinese_mcmle
│   ├── english_medexpqa
│   ├── english_medmcqa
│   ├── english_medqa
│   ├── english_mmlu
│   ├── english_pubmedqa
│   ├── english_usmle
│   ├── french_medexpqa
│   ├── french_medmcqa
│   ├── french_mmlu
│   ├── hindi_mmlu
│   ├── spanish_headqa
│   ├── spanish_medexpqa
│   └── spanish_mmlu
└── utils
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

**Database Contributor and Maintainer**: <br>
- [Lindsey A. Claus](https://scholar.google.com/citations?user=bENmp-UAAAAJ&hl=en)
- [Divya Veerapaneni](https://sites.google.com/view/divyav/research?authuser=0)
- [Meagan V. Lauber](https://scholar.google.com/citations?user=t_QKUhEAAAAJ&hl=en)

If you have any questions, please drop us an email at [brucejia@bu.edu](brucejia@bu.edu), [sbit@bu.edu](sbit@bu.edu), and [nsearls@bu.edu](nsearls@bu.edu).

# Contribution
We always welcome contributions to help make **MedPodGPT** Library better. 
If you would like to contribute, please submit a [pull request](https://github.com/vkola-lab/medpodgpt/pulls).

# Acknowledgement
The **MedPodGPT** Library is created and maintained by the Kolachalama Lab at Boston University.

<a href="https://www.bu.edu/"> <img width="250" src="https://raw.githubusercontent.com/SuperBruceJia/promptcraft/main/bu.png"></a>
