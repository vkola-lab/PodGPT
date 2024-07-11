# 🚀 Model Inference Usage
Due to the internal NCCL communication issues on Boston University Shared Computing Cluster (SCC),
we cannot [release the memory of the distributed GPUs](https://github.com/vllm-project/vllm/issues/1908).
Thus, we use `inference_pretrain.py` and `inference_single_model.py` for larger models (> 8B) 
and `inference_sequential.py` for smaller models (2B/7B/8B).

First, in the project home directory, please copy and paste the inference files,
```shell
cp -r ./inference/inference_sequential.py ./
cp -r ./inference/inference_large.sh ./
cp -r ./inference/inference_pretrain.py ./
cp -r ./inference/inference_single_model.py ./
```

## 🐣 Single GPU For Smaller Models (2B/7B/8B)
### inference_sequential.py
**Sequentially** evaluate the performance of multiple checkpoints (models).<br>
Please note that we use `--eval_pretrain` to indicate whether to evaluate the original pre-trained model.
```shell
python inference_sequential.py --eval_pretrain True --id 35166 52749 70332 87915
```

## 🐥 Distributed GPUs For Larger Models (> 8B)
**Sequentially** evaluate the performance of the original pre-trained model and all the checkpoints.<br>
Special Notice: Please change the checkpoint IDs and CUDA_VISIBLE_DEVICES in the `inference_large.sh` file.
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
