# Quantize a Large Language Model (LLM) Using GPTQ
First, in the project home directory, please copy and paste these files,
```shell
cp -r ./quantization/quantization.py ./
cp -r ./quantization/upload_quantization_model.py ./
```

## Conduct quantization based on GPTQ algorithm
```shell
python quantization.py "./save_folder" "./gptq_model" "medical" --bits 4 --group_size 128 --desc_act 1 --dtype float16
```

## Upload the quantized model to Hugging Face
```shell
python upload_quantization_model.py --repo "shuyuej/MedLLaMA3-70B-BASE-MODEL-QUANT" --folder_path "./gptq_model"
```

## Change the model config files
In the `config.json` file, please change the "architectures" to `LLaMAForCausalLM`.<br>
We don't specifically automatically upload the tokenizer files.<br>
Please manually download them from Hugging Face official repo and upload them to your repo.
