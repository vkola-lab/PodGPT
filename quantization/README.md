# Quantize a Large Language Model (LLM) Using GPTQ
First, in the project home directory, please copy and paste these files,
```shell
cp -r ./quantization/quantization.py ./
cp -r ./quantization/quantization_HF.py ./
cp -r ./quantization/upload_quantized_model.py ./
```
Meanwhile, please use your own Hugging Face READ and WRITE tokens in the `config_quantization.yml` file.

## Conduct quantization based on GPTQ algorithm
For `quantization.py`, we are using Python [AutoGPTQ](https://github.com/AutoGPTQ/AutoGPTQ) package to conduct quantization.
```shell
python quantization.py "meta-llama/Meta-Llama-3-70B-Instruct" "./gptq_model" --bits 4 --group_size 128 --desc_act 1 --dtype bfloat16 --seqlen 2048 --damp 0.01
```

For `quantization_GPTQModel.py`, we are using Python [GPTQModel](https://github.com/ModelCloud/GPTQModel) package to conduct quantization.
```shell
pip install -v gptqmodel --no-build-isolation
```

Then,
```shell
python quantization_GPTQModel.py "meta-llama/Llama-3.3-70B-Instruct" "./gptq_model" --bits 4 --group_size 128 --seqlen 2048 --damp 0.01 --desc_act 1 --dtype bfloat16
```

For `quantization_HF.py`, we are using Hugging Face [transformers](https://github.com/huggingface/transformers) package to conduct quantization.
```shell
python quantization_HF.py --repo "meta-llama/Meta-Llama-3-70B-Instruct" --bits 4 --group_size 128
```

## Upload the quantized model to Hugging Face
```shell
python upload_quantized_model.py --repo "shuyuej/MedLLaMA3-70B-BASE-MODEL-QUANT" --folder_path "./gptq_model"
```

## Change the model config files
In the `config.json` file, please change the "architectures" to `LLaMAForCausalLM` if it is a LLaMA model.<br>
We don't specifically automatically upload the tokenizer files.<br>
Please manually download them from Hugging Face official repo and upload them to your repo.

## Model Split
We also provide a script to split a large SafeTensor file into smaller shards.<be>
The large file will be saved into 5GB shards and a `model.safetensors.index.json` will also be saved.
```shell
python model_split.py --large_file "gptq_model/model.safetensors" --output_dir "split_model" --max_size_gb 5
```
