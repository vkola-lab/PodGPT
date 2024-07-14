<h1 align="center">MedPodGPT</h1>
<h4 align="center">Benchmarking Multilingual Medical Large Language Models (LLMs)</h4>
<p align="center">
  <a href="https://www.medrxiv.org/content/10.1101/2024.07.11.24310304v1"> <img width="250px" src="figures/MedPodGPT.png"></a> 
  <br />
  <br />
  <a href="https://img.shields.io/badge/Code%20License-AGPL3.0-green.svg"><img alt="CODE_LICENSE" src="https://img.shields.io/badge/Code%20License-AGPL3.0-green.svg" /></a>
  <a href="https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red.svg"><img alt="DATA_LICENSE" src="https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red.svg" /></a>
  <a href="https://www.apache.org/licenses/LICENSE-2.0"><img alt="Model Weight License" src="https://img.shields.io/badge/Model%20Weights%20License-Apache-yellow" /></a>
  <a href="https://www.python.org/downloads/release/python-3100/"><img alt="Python 3.10" src="https://img.shields.io/badge/python-3.10-blue.svg" /></a>
</p>

# 🎉 Announcements
[2024.7.14] ![](figures/news.gif) Our AI Platform [**MedPodGPT**](https://medpodgpt.org/) is publicly available. 
It is an online platform for deploying our latest multimodal foundation models for medical and clinical applications.
Please try it out if you are interested!

[2024.7.12] ![](figures/news.gif) Our [**preprint**](https://www.medrxiv.org/content/10.1101/2024.07.11.24310304v1) is available online! Please check it!

[2024.7.12] We are releasing a new benchmark encompassing the latest USMLE Step 1, Step 2, Step 3, and Ethics to further advance the filed.
Check our database [here](https://github.com/vkola-lab/medpodgpt/tree/main/benchmark/english_usmle).

[2024.7.11] We open-sourced the [source codes](https://github.com/vkola-lab/medpodgpt) of our **MedPodGPT**: **medical LLMs in your pocket** and **benchmarking multilingual medical LLMs**.

# 📚 Table of Contents
- [Installation](#-Installation)
- [Quick Start](#-Quick-Start)
  - [Train Lightweight Models](#-Train-Lightweight-Models)
  - [Train Heavy Models](#-Train-Heavy-Models)
  - [Train Quantized Large Models](#-Train-Quantized-Large-Models)
- [Performance Evaluation](#-Performance-Evaluation)
  - [Prompt Format](#-Prompt-Format)
  - [Single GPU For Lightweight Models](#-Single-GPU-For-Lightweight-Models)
  - [Distributed GPUs For Heavy Models](#-Distributed-GPUs-For-Heavy-Models)
  - [OpenAI ChatGPT Support](#-OpenAI-ChatGPT-Support)
- [Dataset Description](#-Dataset-Description)
- [Benchmarks and Results](#-Benchmarks-and-Results)
  - [Benchmarks Description](#Benchmarks-Description)
  - [Performance on In-domain Benchmarks](#Performance-on-In-domain-Benchmarks)
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

# 💻 Installation
```shell
pip install -r requirements.txt
```

# 🚀 Quick Start
## 🐣 Train Lightweight Models
For lightweight models (2B, 7B, and 8B), we optimize the entire model. 
Please check and setup hyper-parameters in [config_small.yml](https://github.com/vkola-lab/medpodgpt/blob/main/config_small.yml).
```shell
python main_small.py
```

## 🐥 Train Heavy Models
For lager and heavy models (>8B), we optimize the Low-rank Adapter (LoRA).
Please check and setup hyper-parameters in [config_large.yml](https://github.com/vkola-lab/medpodgpt/blob/main/config_large.yml).
```shell
python main_large.py
```

## 🐤 Train Quantized Large Models
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

# 📊 Performance Evaluation
All inferences are conducted using the [vLLM engine](https://github.com/vllm-project/vllm).
We use [inference_pretrain.py](https://github.com/vkola-lab/medpodgpt/blob/main/inference/inference_pretrain.py) 
and [inference_single_model.py](https://github.com/vkola-lab/medpodgpt/blob/main/inference/inference_single_model.py)
for larger models (>8B) 
and [inference_sequential.py](https://github.com/vkola-lab/medpodgpt/blob/main/inference/inference_sequential.py) 
for smaller models (2B/7B/8B). 
Please check [here](https://github.com/vkola-lab/medpodgpt/tree/main/inference) for more information.
> [!NOTE]
> **Mistral 7B on Hindi MMLU Benchmarks**:<br>
    Please un-comment [this line](https://github.com/vkola-lab/medpodgpt/blob/main/utils/eval_small_utils.py#L65).<br>
    To address the issue of repeated content in some responses, we applied a repetition_penalty during inference.

## 📜 Prompt Format
We simply use `Directly answer the best option:` instead of `Answer:` to better guide LLMs to generate the best option 
and to easier extract the best option from the responses.<br>
Please modify [these lines](https://github.com/vkola-lab/medpodgpt/blob/main/utils/benchmark_utils.py#L5-L21) 
if you wanna try other prompts.

> [!NOTE]
> **LLaMA 3 8B on Hindi MMLU Benchmarks**:<br>
    Please modify [these lines](https://github.com/vkola-lab/medpodgpt/blob/main/utils/benchmark_utils.py#L15-L18).<br>
    Because most responses are in mixed English-Hindi or English, we used `कृपया प्रश्न का उत्तर हिंदी में दें और सीधे सबसे अच्छे विकल्प के साथ जवाब दें:` (Please answer the question in Hindi and directly answer the best option:) to guide the model.<br><br>

```python
english_prompt = "Directly answer the best option:"
english_prompt_pubmedqa = "Directly answer yes/no/maybe:"
hindi_prompt = "सीधे सबसे अच्छे विकल्प के साथ जवाब दें:"
french_prompt = "Répondez directement avec la meilleure option:"
spanish_prompt = "Responde directamente con la mejor opción:"
chinese_prompt = "直接回答最优选项:"
```

## 🔧 Single GPU For Lightweight Models
> [!IMPORTANT]
> Please note that if you wanna conduct model inference using multiple GPUs, the GPUs' memory cannot be successfully released. 
> Please modify [these lines](https://github.com/vkola-lab/medpodgpt/blob/main/utils/eval_small_utils.py#L84-L85)
> and make use of [this `sh` file](https://github.com/vkola-lab/medpodgpt/blob/main/inference/inference_large.sh).

### inference_sequential.py
**Sequentially** evaluate the performance of multiple checkpoints (models).<br>
Please note that we use `--eval_pretrain` to indicate whether to evaluate the original pre-trained model.
```shell
python inference_sequential.py --eval_pretrain True --id 35166 52749 70332 87915
```

## 🛠️ Distributed GPUs For Heavy Models
**Sequentially** evaluate the performance of the original pre-trained model and all the checkpoints.<br>
Special Notice: Please change the `checkpoint IDs` and `CUDA_VISIBLE_DEVICES` 
in the [inference_large.sh](https://github.com/vkola-lab/medpodgpt/blob/main/inference/inference_large.sh) file.
```shell
sh inference_large.sh
```

### inference_pretrain.py
**Only** evaluate the performance of the original pre-trained model.
```shell
python inference_pretrain.py
```

### inference_single_model.py
**Only** evaluate the performance of a single checkpoint (model).<br>
Please note that `--id` is the checkpoint id.
```shell
python inference_single_model.py --id 35166
```

## 🤖 OpenAI ChatGPT Support
We also offer support for running OpenAI ChatGPT inference using API.
Please enter your OpenAI API Key [here](https://github.com/vkola-lab/medpodgpt/blob/main/config_chatgpt.yml#L18).
> [!WARNING]  
> Please note that OpenAI ChatGPT API is extremely expensive.<br>
> Please only use it if you have a budget for it!

```shell
python inference_chatgpt.py
```

# 📚 Dataset Description
For now, we released a [demo dataset](https://huggingface.co/datasets/shuyuej/MedPodGPT-Demo-Data) 
for you to run the codes. Please follow our instructions 
to [transcribe your own podcasts](https://github.com/vkola-lab/medpodgpt/blob/main/scripts/audio2text.py) 
and [build your own dataset](https://github.com/vkola-lab/medpodgpt/blob/main/scripts/database_builder.py).

The podcasts data used for the continual pre-training of **MedPodGPT**:
<p align="center">
  <a href="https://www.medrxiv.org/content/10.1101/2024.07.11.24310304v1"> <img src="figures/Table-1.png"></a> 
</p>

# 🏆 Benchmarks and Results

## Benchmarks Description

|  *Language*   |               *Dataset*                | *# test examples* | *# of choices* |                                          *Link*                                          |                    *Ref*                    |
|:-------------:|:--------------------------------------:|:-----------------:|:--------------:|:----------------------------------------------------------------------------------------:|:-------------------------------------------:|
| ***English*** |                MedExpQA                |        125        |       5        |           [Link](https://huggingface.co/datasets/HiTZ/MedExpQA/viewer/en/test)           |  [Paper](https://arxiv.org/abs/2404.05590)  |
|               |                 MedQA                  |       1273        |       4        |      [Link](https://drive.google.com/file/d/1ImYUSLk9JbgHXOemfvyiDiirluZHPeQw/view)      |  [Paper](https://arxiv.org/abs/2009.13081)  |
|               |                MedMCQA                 |       4183        |       4        | [Link](https://drive.google.com/uc?export=download&id=15VkJdq5eyWIkfb_aoD3oS8i4tScbHYky) |  [Paper](https://arxiv.org/abs/2203.14371)  |
|               |                PubMedQA                |       1000        |       3        |             [Link](https://github.com/pubmedqa/pubmedqa?tab=readme-ov-file)              |  [Paper](https://arxiv.org/abs/1909.06146)  |
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

## Performance on In-domain Benchmarks
<p align="center">
  <a href="https://www.medrxiv.org/content/10.1101/2024.07.11.24310304v1"> <img src="figures/Table-2.png"></a> 
</p>

## Zero-shot Cross-lingual Performance
<p align="center">
  <a href="https://www.medrxiv.org/content/10.1101/2024.07.11.24310304v1"> <img src="figures/Table-3.png"></a> 
</p>

# 🔥 Real-world Deployment
For real-world deployment, please refer to 
the [vLLM Distributed Inference and Serving](https://docs.vllm.ai/en/latest/serving/distributed_serving.html) 
and [OpenAI Compatible Server](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html).

# 🎯 Automatic Speech Recognition
In the [scripts folder](https://github.com/vkola-lab/medpodgpt/tree/main/scripts), 
we provide Automatic Speech Recognition (ASR) service.
```shell
python audio2text.py
```

# ⚒️ Dataset Builder
We used the following codes to pre-process our transcripts and generate training dataset.
Please check [these lines](https://github.com/vkola-lab/medpodgpt/blob/main/scripts/database_builder.py#L236-L242) 
for different languages support.
```shell
python database_builder.py
```
```shell
python merge_database.py
```

# 🛠️ Upload and Download Models
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

# 🖼️ Structure of the Code
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

# 🙏 Citation
If you find our work useful in your research, please consider citing it in your publications. We provide a BibTeX entry below.

```bibtex
@article {Jia2024medpodgpt,
	author       = {Jia, Shuyue and Bit, Subhrangshu and Searls, Edward and Claus, Lindsey and Fan, Pengrui and Jasodanand, Varuna H. and Lauber, Meagan V. and Veerapaneni, Divya and Wang, William M. and Au, Rhoda and Kolachalama, Vijaya B},
	title        = {{MedPodGPT}: A multilingual audio-augmented large language model for medical research and education},
	elocation-id = {2024.07.11.24310304},
	year         = {2024},
	doi          = {10.1101/2024.07.11.24310304},
	publisher    = {Cold Spring Harbor Laboratory Press},
	abstract     = {The proliferation of medical podcasts has generated an extensive repository of audio content, rich in specialized terminology, diverse medical topics, and expert dialogues. Here we introduce a computational framework designed to enhance large language models (LLMs) by leveraging the informational content of publicly accessible medical podcast data. This dataset, comprising over 4,300 hours of audio content, was transcribed to generate over 39 million text tokens. Our model, MedPodGPT, integrates the varied dialogue found in medical podcasts to improve understanding of natural language nuances, cultural contexts, and medical knowledge. Evaluated across multiple benchmarks, MedPodGPT demonstrated an average improvement of 2.31\% over standard open-source benchmarks and showcased an improvement of 2.58\% in its zero-shot multilingual transfer ability, effectively generalizing to different linguistic contexts. By harnessing the untapped potential of podcast content, MedPodGPT advances natural language processing, offering enhanced capabilities for various applications in medical research and education.Competing Interest StatementV.B.K. is on the scientific advisory board for Altoida Inc. and serves as a consultant to AstraZeneca. R.A. is a scientific advisor to Signant Health and NovoNordisk. The remaining authors declare no competing interests.Funding StatementNational Institutes of HealthAuthor DeclarationsI confirm all relevant ethical guidelines have been followed, and any necessary IRB and/or ethics committee approvals have been obtained.YesI confirm that all necessary patient/participant consent has been obtained and the appropriate institutional forms have been archived, and that any patient/participant/sample identifiers included were not known to anyone (e.g., hospital staff, patients or participants themselves) outside the research group so cannot be used to identify individuals.YesI understand that all clinical trials and any other prospective interventional studies must be registered with an ICMJE-approved registry, such as ClinicalTrials.gov. I confirm that any such study reported in the manuscript has been registered and the trial registration ID is provided (note: if posting a prospective study registered retrospectively, please provide a statement in the trial ID field explaining why the study was not registered in advance).Yes I have followed all appropriate research reporting guidelines, such as any relevant EQUATOR Network research reporting checklist(s) and other pertinent material, if applicable.YesAll data produced are available online at https://github.com/vkola-lab/MedPodGPT.https://github.com/vkola-lab/MedPodGPT},
	URL          = {https://www.medrxiv.org/content/early/2024/07/12/2024.07.11.24310304},
	eprint       = {https://www.medrxiv.org/content/early/2024/07/12/2024.07.11.24310304.full.pdf},
	journal      = {medRxiv}
}
```

# 📧 Contact
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

# 🔨 Contribution
We always welcome contributions to help make **MedPodGPT** Library better. 
If you would like to contribute, please submit a [pull request](https://github.com/vkola-lab/medpodgpt/pulls).

# 🙌 Acknowledgement
The **MedPodGPT** Library is created and maintained by the Kolachalama Lab at Boston University.

<a href="https://www.bu.edu/"> <img width="250" src="https://raw.githubusercontent.com/SuperBruceJia/promptcraft/main/bu.png"></a>
