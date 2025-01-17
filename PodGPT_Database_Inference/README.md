# Database evaluation
Evaluate PodGPT, PodGPT with Retrieval-augmented Generation (RAG), or ChatGPT on your own database!

## 💻 Installation
In addition to the [**PodGPT dependencies**](https://github.com/vkola-lab/PodGPT/blob/main/requirements.txt), please make sure to install the following packages:
```bash
pip install pgvector
pip install flask-sqlalchemy
pip install psycopg2
```

## 📖 Prepare your own database
We used the MMLU Professional medicine database as an example here:
<p align="center">
  <a href="https://www.medrxiv.org/content/10.1101/2024.07.11.24310304v2"> <img src="figures/database.png"></a> 
</p>
To prepare your own database, please make sure you have:

1. `question`: the question is in the first column (`column A`) of our demo database.
2. `options`: we have four options (A, B, C, and D) in our demo database, from `column B` to `column E`.
3. `answer`: The ground truth answer is located in the fifth column (`column F`) of our demo database.

## 🚀 Inference and benchmarking
```bash
python main.py --mode podgpt --rag True --eval_pretrain True
```
1. `--mode`: Evaluate PodGPT or ChatGPT: podgpt/chatgpt. The default is "podgpt".
2. `--rag`: Whether to use RAG database and pipeline: True/False. The default is True.
2. `--eval_pretrain`: Evaluate the original pre-trained model: True/False. The default is True.

## 🏞️ Structure of the code
At the root of this folder, you will see:
```text
├── main.py
├── config_podgpt.yml
├── config_benchmark.yml
├── config_chatgpt.yml
├── lib
│   ├── config.py
│   ├── database.py
│   ├── evaluation.py
│   ├── model_loader.py
│   └── pipeline.py
├── benchmark
│   └── database.csv
└── utils
    ├── answer_utils.py
    ├── benchmark_utils.py
    ├── eval_utils.py
    ├── utils.py
    └── vllm_utils.py
```
