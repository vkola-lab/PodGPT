# 🚩 Download HF Repo and Models to A Local Folder
First, in the project home directory, please copy and paste these files,
```shell
cp -r ./download_files/download_model_from_hf.py ./
cp -r ./download_files/download_model_to_local.py ./
```

## 💻 Download a repo to a local folder
We support downloading `model` and `dataset` from Hugging Face.

```shell
python download_model_from_hf.py --repo "shuyuej/MedGemma7B-Multilingual" --repo_type "model" --save_dir "./save_folder"
```

## 🤖 Download models to a local folder
The model and tokenizer (`model_name`) in the `config_large.yml` will be downloaded and saved in the `save_dir`.

```shell
python download_model_to_local.py
```
