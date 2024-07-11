#! /bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3 python inference_pretrain.py
CUDA_VISIBLE_DEVICES=0,1,2,3 python inference_single_model.py --id 14996
CUDA_VISIBLE_DEVICES=0,1,2,3 python inference_single_model.py --id 29992
CUDA_VISIBLE_DEVICES=0,1,2,3 python inference_single_model.py --id 44988
CUDA_VISIBLE_DEVICES=0,1,2,3 python inference_single_model.py --id 59984
CUDA_VISIBLE_DEVICES=0,1,2,3 python inference_single_model.py --id 82870
