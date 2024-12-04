# 🚀 Model Inference Usage
Here, we use a unified `inference.py` file for model inference.

First, in the project home directory, please copy and paste the inference.py file,
```shell
cp -r ./inference/inference.py ./
```

## 🐣 Inference
**Sequentially** evaluate the performance of multiple checkpoints (models).<br>

```shell
python inference.py --mode small --eval_pretrain True --id 35166 52749 70332 87915
```
- `--mode`: whether to evaluate smaller models (2B/7B/8B), larger models (with LoRA), or ChatGPT (small, large, or 
  chatgpt), the default is small
- `--eval_pretrain`: whether to evaluate the original pre-trained model (True or False), the default value is True
- `--id`: a checkpoint id or a list of checkpoint ids that need to be evaluated sequentially 

## 🙋 Special Notice
1. Please modify the vLLM hyperparameter configurations in the `config_small.yml`, `config_large.yml`, and 
   `config_quantization.yml` according to your own GPU settings.
2. Please change the OpenAI API KEY in the `config_chatgpt.yml` file if you wanna evaluate ChatGPT.
