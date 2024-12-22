# PodRAG
PodGPT with Retrieval-augmented Generation (RAG)

## 💻 Installation
In addition to the [**PodGPT dependencies**](https://github.com/vkola-lab/PodGPT/blob/main/requirements.txt), please make sure to install the following packages:
```bash
pip install pgvector
pip install flask-sqlalchemy
pip install psycopg2
```

## 📖 Download your fine-tuned checkpoints
All your checkpoints will be saved to `./save_folder` by default.
```bash
python download_model.py --repo "shuyuej/gemma-2b-it-2048"
```
1. `--repo`: Your Hugging Face Repo ID that contains all your fine-tuned checkpoints. 
    The default is "shuyuej/gemma-2b-it-2048".
2. `--save_dir`: Your checkpoints will be saved into this `save_dir`. The default is "./save_folder".

## 🚀 Inference and benchmarking
```bash
python main.py --mode small --eval_pretrain True
```
1. `--mode`: Evaluate the smaller model or larger model: small or large. The default is "small".
2. `--eval_pretrain`: Evaluate the original pre-trained model: True/False. The default is True.

## 🏞️ Structure of the code
At the root of this folder, you will see:
```text
├── benchmark
├── config_small.yml
├── config_large.yml
├── config_benchmark.yml
├── main.py
├── download_model.py
├── lib
│   ├── config.py
│   ├── database.py
│   ├── evaluation.py
│   ├── model_loader.py
│   └── pipeline.py
└── utils
    ├── answer_utils.py
    ├── benchmark_utils.py
    ├── eval_utils.py
    ├── vllm_utils.py
    └── utils.py
```
