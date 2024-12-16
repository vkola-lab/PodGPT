# PodGPT with retrieval-augmented generation (RAG)

## 💻 Installation
In addition to the [**PodGPT dependencies**](https://github.com/vkola-lab/PodGPT/blob/main/requirements.txt), please make sure to install the following packages:
```bash
pip install pgvector
pip install flask-sqlalchemy
```

## 📖 Download your fine-tuned checkpoints
All your checkpints will be saved to `./save_folder` by default.
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
2. `--eval_pretrain`: Evaluate the original pretrained model: True/False. The default is True.
