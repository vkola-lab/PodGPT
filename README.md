<h1 align="center">PodGPT</h1>
<h4 align="center">An audio-augmented large language model for research and education</h4>
<p align="center">
  <a href="https://www.medrxiv.org/content/10.1101/2024.07.11.24310304v2"> <img width="250px" src="figures/PodGPT.png"></a> 
  <br />
  <br />
  <a href="https://img.shields.io/badge/Code%20License-AGPL3.0-green.svg"><img alt="CODE_LICENSE" src="https://img.shields.io/badge/Code%20License-AGPL3.0-green.svg" /></a>
  <a href="https://img.shields.io/badge/Data%20License-CC%20By%204.0-red.svg"><img alt="DATA_LICENSE" src="https://img.shields.io/badge/Data%20License-CC%20BY%204.0-red.svg" /></a>
  <a href="https://www.apache.org/licenses/LICENSE-2.0"><img alt="Model Weight License" src="https://img.shields.io/badge/Model%20Weights%20License-Apache-yellow" /></a>
  <a href="https://www.python.org/downloads/release/python-3100/"><img alt="Python 3.10" src="https://img.shields.io/badge/python-3.10-blue.svg" /></a>
</p>

# 🎉 Announcements
[2024.12.04] ![](figures/news.gif) Our [**PodGPT preprint**](https://www.medrxiv.org/content/10.1101/2024.07.11.24310304v2) is available online! Please check it!

[2024.7.14] ![](figures/news.gif) Our AI Platform [**PodGPT**](https://podgpt.org/) is publicly available. 
It is an online platform for deploying our latest multimodal foundation models for STEMM education and research.
Please try it out if you are interested!

[2024.7.12] We are releasing a new benchmark encompassing the latest USMLE Step 1, Step 2, Step 3, and Ethics to further advance the filed.
Check our database [here](https://github.com/vkola-lab/PodGPT/tree/main/benchmark/english_usmle).

[2024.7.11] We open-sourced the [source codes](https://github.com/vkola-lab/PodGPT) of our **PodGPT**: **STEMM LLMs in 
your pocket** and **benchmarking multilingual STEMM LLMs**.

# 📚 Table of contents
- [PodGPT](#-PodGPT)
- [Installation](#-Installation)
- [Quick Start](#-Quick-Start)
  - [Train Lightweight Models](#-Train-Lightweight-Models)
  - [Train Heavy Models](#-Train-Heavy-Models)
  - [Train Quantized Large Models](#-Train-Quantized-Large-Models)
- [Performance Evaluation](#-Performance-Evaluation)
  - [Prompt Format](#-Prompt-Format)
  - [Model Inference](#-Model-Inference)
  - [OpenAI ChatGPT Support](#-OpenAI-ChatGPT-Support)
- [Dataset Description](#-Dataset-Description)
- [Benchmarks and Results](#-Benchmarks-and-Results)
  - [Benchmarks Description](#Multilingual-Benchmarks-Description)
  - [Performance on In-domain Benchmarks](#Performance-on-In-domain-Benchmarks)
  - [Performance of Retrieval-augmented Generation](#Performance-of-Retrieval-augmented-Generation)
  - [Zero-shot Cross-lingual Performance](#Zero-shot-Cross-lingual-Performance)
- [Real-world Deployment](#-Real-world-Deployment)
- [Automatic Speech Recognition](#-Automatic-Speech-Recognition)
- [Dataset Builder](#-Dataset-Builder)
- [Upload and Download Models](#-Upload-and-Download-Models)
- [Structure of the Code](#-Structure-of-the-Code)
- [Citation](#-Citation)
- [Contact](#-Contact)
- [Contribution](#-Contribution)
- [Acknowledgement](#-Acknowledgement)

# ❤️ PodGPT
Our proposed **PodGPT** computational framework for research and education
<p align="center">
  <a href="https://www.medrxiv.org/content/10.1101/2024.07.11.24310304v2"> <img src="figures/PodGPT-Framework.png"></a> 
</p>

# 💻 Installation
```shell
pip install -r requirements.txt
```

# 🚀 Quick start
## 🐣 Train lightweight models
For lightweight models (2B, 7B, 8B, and 9B), we optimize the entire model. 
Please check and setup hyperparameters and Hugging Face READ/WRITE Tokens in [config_small.yml](https://github.com/vkola-lab/PodGPT/blob/main/config_small.yml).
```shell
python main_small.py
```

## 🐥 Train heavy models
For lager and heavy models (>9B), we optimize the Low-rank Adapter (LoRA).
Please check and setup hyperparameters and Hugging Face READ/WRITE Token in [config_large.yml](https://github.com/vkola-lab/PodGPT/blob/main/config_large.yml).
```shell
python main_large.py
```

## 🐤 Train quantized large models
We also provide support for quantizing larger models, _e.g._, LLaMA 3.1 70B model, using the [GPTQ](https://arxiv.org/abs/2210.17323) algorithm and then optimizing the LoRA.
***The large models can be deployed on consumer GPUs after quantization.***

We can directly use the Hugging Face [transformers](https://github.com/huggingface/transformers) package to conduct quantization.
```shell
python quantization_HF.py --repo "meta-llama/Meta-Llama-3.1-70B-Instruct" --bits 4 --group_size 128
```

Or, we enable the Python [GPTQModel](https://github.com/ModelCloud/GPTQModel) package to conduct quantization.
```shell
pip install -v gptqmodel --no-build-isolation
```

Then,
```shell
python quantization_GPTQModel.py "meta-llama/Llama-3.3-70B-Instruct" "./gptq_model" --bits 4 --group_size 128 --seqlen 2048 --damp 0.01 --desc_act 1 --dtype bfloat16
```

Alternatively, we also provide a quantization script using the Python [AutoGPTQ](https://github.com/AutoGPTQ/AutoGPTQ) package.
```shell
python quantization.py "meta-llama/Meta-Llama-3.1-70B-Instruct" "./gptq_model" --bits 4 --group_size 128 --desc_act 1 --dtype bfloat16 --seqlen 2048 --damp 0.01
```

Then, we need to upload the model to Hugging Face, for example,
```shell
python upload_quantized_model.py --repo "shuyuej/MedLLaMA3-70B-BASE-MODEL-QUANT" --folder_path "./gptq_model"
```

Lastly, we optimize the LoRA module,
```shell
python main_quantization.py
```
_Special Notice_: <br>
1. Check [this solution](https://github.com/AutoGPTQ/AutoGPTQ/issues/509#issuecomment-2274011154) if you cannot successfully start the model training.
2. Check [this solution](https://github.com/oobabooga/text-generation-webui/issues/4074#issuecomment-1790059935) if your adapters cannot be saved due to PEFT.
3. There are many unexpected issues for model quantization as well as model training, checkpoint saving, and vLLM 
   inference. Please submit a GitHub issue if you cannot solve it. **We should meet all the problems before**
   in terms of single-GPU and distributed-GPU, _e.g._, 4 A100 80G GPUs, settings.

# 📊 Performance evaluation
All inferences are conducted using the [vLLM engine](https://github.com/vllm-project/vllm).
We use [inference.py](https://github.com/vkola-lab/PodGPT/blob/main/inference/inference.py) to sequentially evaluate the performance of multiple checkpoints (models).<br>
Please check [here](https://github.com/vkola-lab/PodGPT/tree/main/inference) for more information.

## 📜 Prompt format
We simply use `Directly answer the best option:` instead of `Answer:` to better guide LLMs to generate the best option 
and to easier extract the best option from the responses.<br>
Please modify [these lines](https://github.com/vkola-lab/PodGPT/blob/main/utils/benchmark_utils.py#L5-L21) if you wanna try other prompts.

```python
english_prompt = "Directly answer the best option:"
english_prompt_pubmedqa = "Directly answer yes/no/maybe:"
hindi_prompt = "सीधे सबसे अच्छे विकल्प के साथ जवाब दें:"
french_prompt = "Répondez directement avec la meilleure option:"
spanish_prompt = "Responde directamente con la mejor opción:"
chinese_prompt = "直接回答最优选项:"
```

## 🛠 Model inference
**Sequentially** evaluate the performance of multiple checkpoints (models).<br>
Please note that we use `--eval_pretrain` to indicate whether to evaluate the original pre-trained model.
```shell
python inference.py --mode small --eval_pretrain True --id 35166 52749 70332 87915
```

## 🤖 OpenAI ChatGPT support
We also offer support for running OpenAI ChatGPT inference using API.
Please enter your OpenAI API Key [here](https://github.com/vkola-lab/PodGPT/blob/main/config_chatgpt.yml#L18).
> [!WARNING]  
> Please note that OpenAI ChatGPT API is extremely expensive.<br>
> Please only use it if you have a budget for it!

```shell
python inference.py --mode chatgpt
```

# 📚 Dataset description
Please follow our instructions to [transcribe your own podcasts](https://github.com/vkola-lab/PodGPT/blob/main/scripts/audio2text.py) 
and [build your own dataset](https://github.com/vkola-lab/PodGPT/blob/main/scripts/database_builder.py).

The podcasts data used for the continual pre-training of **PodGPT**:
<p align="center">
  <a href="https://www.medrxiv.org/content/10.1101/2024.07.11.24310304v2"> <img src="figures/Table-1.png"></a> 
</p>

# 🏆 Benchmarks and results
## Multilingual benchmarks
We utilized a comprehensive set of medical benchmarks from the most widely spoken languages in the world, 
including **_English_**, **_Mandarin_**, **_French_**, **_Spanish_**, and **_Hindi_**.

|  *Language*   |               *Dataset*                | *# test examples* | *# of choices* |                                          *Link*                                          |                    *Ref*                    |
|:-------------:|:--------------------------------------:|:-----------------:|:--------------:|:----------------------------------------------------------------------------------------:|:-------------------------------------------:|
| ***English*** |                MedExpQA                |        125        |       5        |           [Link](https://huggingface.co/datasets/HiTZ/MedExpQA/viewer/en/test)           |  [Paper](https://arxiv.org/abs/2404.05590)  |
|               |                 MedQA                  |       1273        |       4        |      [Link](https://drive.google.com/file/d/1ImYUSLk9JbgHXOemfvyiDiirluZHPeQw/view)      |  [Paper](https://arxiv.org/abs/2009.13081)  |
|               |                MedMCQA                 |       4183        |       4        | [Link](https://drive.google.com/uc?export=download&id=15VkJdq5eyWIkfb_aoD3oS8i4tScbHYky) |  [Paper](https://arxiv.org/abs/2203.14371)  |
|               |                PubMedQA                |       500        |       3        |             [Link](https://github.com/pubmedqa/pubmedqa?tab=readme-ov-file)              |  [Paper](https://arxiv.org/abs/1909.06146)  |
|               |            *MMLU* - Anatomy            |        135        |       4        |             [Link](https://www.kaggle.com/datasets/lizhecheng/mmlu-dataset)              |  [Paper](https://arxiv.org/abs/2009.03300)  |
|               |      *MMLU* - Clinical Knowledge       |        265        |       4        |             [Link](https://www.kaggle.com/datasets/lizhecheng/mmlu-dataset)              |  [Paper](https://arxiv.org/abs/2009.03300)  |
|               |        *MMLU* - College Biology        |        144        |       4        |             [Link](https://www.kaggle.com/datasets/lizhecheng/mmlu-dataset)              |  [Paper](https://arxiv.org/abs/2009.03300)  |
|               |       *MMLU* - College Medicine        |        173        |       4        |             [Link](https://www.kaggle.com/datasets/lizhecheng/mmlu-dataset)              |  [Paper](https://arxiv.org/abs/2009.03300)  |
|               |       *MMLU* - Medical Genetics        |        100        |       4        |             [Link](https://www.kaggle.com/datasets/lizhecheng/mmlu-dataset)              |  [Paper](https://arxiv.org/abs/2009.03300)  |
|               |     *MMLU* - Professional Medicine     |        272        |       4        |             [Link](https://www.kaggle.com/datasets/lizhecheng/mmlu-dataset)              |  [Paper](https://arxiv.org/abs/2009.03300)  |
| ***French***  |                MedExpQA                |        125        |       5        |           [Link](https://huggingface.co/datasets/HiTZ/MedExpQA/viewer/fr/test)           |  [Paper](https://arxiv.org/abs/2404.05590)  |
|               |                MedMCQA                 |        622        |       5        |                    [Link](https://github.com/qanastek/FrenchMedMCQA)                     | [Paper](https://hal.science/hal-03824241v1) |
|               |            *MMLU* - Anatomy            |        135        |       4        |         [Link](https://huggingface.co/datasets/FreedomIntelligence/MMLU_French)          |  [Paper](https://arxiv.org/abs/2403.03640)  |
|               |      *MMLU* - Clinical Knowledge       |        265        |       4        |         [Link](https://huggingface.co/datasets/FreedomIntelligence/MMLU_French)          |  [Paper](https://arxiv.org/abs/2403.03640)  |
|               |        *MMLU* - College Biology        |        144        |       4        |         [Link](https://huggingface.co/datasets/FreedomIntelligence/MMLU_French)          |  [Paper](https://arxiv.org/abs/2403.03640)  |
|               |       *MMLU* - College Medicine        |        173        |       4        |         [Link](https://huggingface.co/datasets/FreedomIntelligence/MMLU_French)          |  [Paper](https://arxiv.org/abs/2403.03640)  |
|               |       *MMLU* - Medical Genetics        |        100        |       4        |         [Link](https://huggingface.co/datasets/FreedomIntelligence/MMLU_French)          |  [Paper](https://arxiv.org/abs/2403.03640)  |
|               |     *MMLU* - Professional Medicine     |        272        |       4        |         [Link](https://huggingface.co/datasets/FreedomIntelligence/MMLU_French)          |  [Paper](https://arxiv.org/abs/2403.03640)  |
| ***Spanish*** |                HEAD-QA                 |       2742        |       4        |                 [Link](https://huggingface.co/datasets/dvilares/head_qa)                 | [Paper](https://aclanthology.org/P19-1092/) |
|               |                MedExpQA                |        125        |       5        |                  [Link](https://huggingface.co/datasets/HiTZ/MedExpQA)                   |  [Paper](https://arxiv.org/abs/2404.05590)  |
|               |            *MMLU* - Anatomy            |        135        |       4        |         [Link](https://huggingface.co/datasets/FreedomIntelligence/MMLU_Spanish)         |  [Paper](https://arxiv.org/abs/2403.03640)  |
|               |      *MMLU* - Clinical Knowledge       |        265        |       4        |         [Link](https://huggingface.co/datasets/FreedomIntelligence/MMLU_Spanish)         |  [Paper](https://arxiv.org/abs/2403.03640)  |
|               |        *MMLU* - College Biology        |        144        |       4        |         [Link](https://huggingface.co/datasets/FreedomIntelligence/MMLU_Spanish)         |  [Paper](https://arxiv.org/abs/2403.03640)  |
|               |       *MMLU* - College Medicine        |        173        |       4        |         [Link](https://huggingface.co/datasets/FreedomIntelligence/MMLU_Spanish)         |  [Paper](https://arxiv.org/abs/2403.03640)  |
|               |       *MMLU* - Medical Genetics        |        100        |       4        |         [Link](https://huggingface.co/datasets/FreedomIntelligence/MMLU_Spanish)         |  [Paper](https://arxiv.org/abs/2403.03640)  |
|               |     *MMLU* - Professional Medicine     |        272        |       4        |         [Link](https://huggingface.co/datasets/FreedomIntelligence/MMLU_Spanish)         |  [Paper](https://arxiv.org/abs/2403.03640)  |
| ***Chinese*** |              MedQA-MCMLE               |       3426        |       4        |      [Link](https://drive.google.com/file/d/1ImYUSLk9JbgHXOemfvyiDiirluZHPeQw/view)      |  [Paper](https://arxiv.org/abs/2009.13081)  |
|               |           *CMMLU* - Anatomy            |        148        |       4        |                 [Link](https://huggingface.co/datasets/haonan-li/cmmlu)                  |  [Paper](https://arxiv.org/abs/2306.09212)  |
|               |      *CMMLU* - Clinical Knowledge      |        237        |       4        |                 [Link](https://huggingface.co/datasets/haonan-li/cmmlu)                  |  [Paper](https://arxiv.org/abs/2306.09212)  |
|               |       *CMMLU* - College Medicine       |        273        |       4        |                 [Link](https://huggingface.co/datasets/haonan-li/cmmlu)                  |  [Paper](https://arxiv.org/abs/2306.09212)  |
|               |       *CMMLU* - Medical Genetics       |        176        |       4        |                 [Link](https://huggingface.co/datasets/haonan-li/cmmlu)                  |  [Paper](https://arxiv.org/abs/2306.09212)  |
|               | *CMMLU* - Traditional Chinese Medicine |        185        |       4        |                 [Link](https://huggingface.co/datasets/haonan-li/cmmlu)                  |  [Paper](https://arxiv.org/abs/2306.09212)  |
|               |           *CMMLU* - Virology           |        169        |       4        |                 [Link](https://huggingface.co/datasets/haonan-li/cmmlu)                  |  [Paper](https://arxiv.org/abs/2306.09212)  |
|  ***Hindi***  |            *MMLU* - Anatomy            |        135        |       4        |          [Link](https://huggingface.co/datasets/FreedomIntelligence/MMLU_Hindi)          |  [Paper](https://arxiv.org/abs/2403.03640)  |
|               |      *MMLU* - Clinical Knowledge       |        265        |       4        |          [Link](https://huggingface.co/datasets/FreedomIntelligence/MMLU_Hindi)          |  [Paper](https://arxiv.org/abs/2403.03640)  |
|               |        *MMLU* - College Biology        |        144        |       4        |          [Link](https://huggingface.co/datasets/FreedomIntelligence/MMLU_Hindi)          |  [Paper](https://arxiv.org/abs/2403.03640)  |
|               |       *MMLU* - College Medicine        |        173        |       4        |          [Link](https://huggingface.co/datasets/FreedomIntelligence/MMLU_Hindi)          |  [Paper](https://arxiv.org/abs/2403.03640)  |
|               |       *MMLU* - Medical Genetics        |        100        |       4        |          [Link](https://huggingface.co/datasets/FreedomIntelligence/MMLU_Hindi)          |  [Paper](https://arxiv.org/abs/2403.03640)  |
|               |     *MMLU* - Professional Medicine     |        272        |       4        |          [Link](https://huggingface.co/datasets/FreedomIntelligence/MMLU_Hindi)          |  [Paper](https://arxiv.org/abs/2403.03640)  |

## Performance on in-domain benchmarks
<p align="center">
  <a href="https://www.medrxiv.org/content/10.1101/2024.07.11.24310304v2"> <img src="figures/Table-2.png"></a> 
</p>

## Performance of retrieval-augmented generation
<p align="center">
  <a href="https://www.medrxiv.org/content/10.1101/2024.07.11.24310304v2"> <img src="figures/Table-3.png"></a>
</p>

## Zero-shot cross-lingual performance
<p align="center">
  <a href="https://www.medrxiv.org/content/10.1101/2024.07.11.24310304v2"> <img src="figures/Table-4.png"></a> 
</p>

# 🔥 Real-world deployment
For real-world deployment, please refer to 
the [vLLM Distributed Inference and Serving](https://docs.vllm.ai/en/latest/serving/distributed_serving.html) 
and [OpenAI Compatible Server](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html).

# 🎯 Automatic speech recognition
In [this file](https://github.com/vkola-lab/PodGPT/blob/main/scripts/audio2text.py), we provide Automatic Speech Recognition (ASR) service.
```shell
python audio2text.py
```

# ⚒️ Dataset builder
We used the following codes to pre-process our transcripts and generate the training dataset.
```shell
python database_builder.py
```

# 🛠️ Upload and download models
In the [scripts folder](https://github.com/vkola-lab/PodGPT/tree/main/scripts), 
we offer support for both uploading and downloading models.

To upload your checkpoints to Hugging Face model repo,
```shell
python upload_model.py --repo "shuyuej/DrGemma2B" --id 35166 52749 70332 87915
```

To download your model or files from Hugging Face repo,
```shell
python download_model.py --repo "shuyuej/DrGemma2B" --repo_type "model" --save_dir "./save_folder"
```

# 🖼️ Structure of the code
At the root of the project, you will see:

```text
├── config_benchmark.yml
├── config_chatgpt.yml
├── config_large.yml
├── config_quantization.yml
├── config_small.yml
├── main_large.py
├── main_quantization.py
├── main_small.py
├── lib
│   ├── data_manager.py
│   ├── evaluation.py
│   ├── model_loader_large.py
│   ├── model_loader_quantization.py
│   └── model_loader_small.py
├── inference
│   └── inference.py
├── quantization
│   ├── model_split.py
│   ├── quantization.py
│   ├── quantization_HF.py
│   ├── quantization_GPTQModel.py
│   └── upload_quantized_model.py
├── download_files
│   ├── download_model_from_hf.py
│   └── download_model_to_local.py
├── requirements.txt
├── benchmark
├── results
├── save_folder
├── scripts
│   ├── audio2text.py
│   ├── database_builder.py
│   ├── download_model.py
│   └── upload_model.py
└── utils
    ├── answer_utils.py
    ├── benchmark_utils.py
    ├── eval_utils.py
    └── utils.py
```

# 🙏 Citation
If you find our work useful in your research, please consider citing it in your publications. We provide a BibTeX entry below.

```bibtex
@article {Jia2024podgpt,
	author = {Jia, Shuyue and Bit, Subhrangshu and Searls, Edward and Lauber, Meagan V. and Claus, Lindsey A. and Fan, Pengrui and Jasodanand, Varuna H. and Veerapaneni, Divya and Wang, William M. and Au, Rhoda and Kolachalama, Vijaya B.},
	title = {{PodGPT}: An audio-augmented large language model for research and education},
	elocation-id = {2024.07.11.24310304},
	year = {2024},
	doi = {10.1101/2024.07.11.24310304},
	publisher = {Cold Spring Harbor Laboratory Press},
	URL = {https://www.medrxiv.org/content/early/2024/11/27/2024.07.11.24310304},
	eprint = {https://www.medrxiv.org/content/early/2024/11/27/2024.07.11.24310304.full.pdf},
	journal = {medRxiv}
}
```

# 📧 Contact
**Core Contributor and Maintainer** (Equal Contributions): <br>
- [Shuyue Jia](https://github.com/SuperBruceJia)
- [Subhrangshu Bit](https://github.com/SubhrangshuBit)
- [Edward Searls](https://github.com/nsearls-bu)
- [Meagan V. Lauber](https://scholar.google.com/citations?user=t_QKUhEAAAAJ&hl=en)
- [Pengrui Fan](https://github.com/pengrui26)
- [William M. Wang](https://github.com/bomas7)

**Database Contributor and Maintainer**: <br>
- [Lindsey A. Claus](https://scholar.google.com/citations?user=bENmp-UAAAAJ&hl=en)
- [Divya Veerapaneni](https://sites.google.com/view/divyav/research?authuser=0)
- [Meagan V. Lauber](https://scholar.google.com/citations?user=t_QKUhEAAAAJ&hl=en)
- [Varuna H. Jasodanand](https://scholar.google.com/citations?user=VZwxdNQAAAAJ&hl=en)

If you have any questions, please drop us an email at [brucejia@bu.edu](brucejia@bu.edu), [sbit@bu.edu](sbit@bu.edu), and [nsearls@bu.edu](nsearls@bu.edu).

# 🔨 Contributions
We always welcome contributions to help make **PodGPT** better. If you would like to contribute, please submit a [pull request](https://github.com/vkola-lab/PodGPT/pulls).

# 🙌 Acknowledgements
This repository is maintained by members of the [Kolachalama Laboratory](https://vkola-lab.github.io/).
