#!/bin/sh

CUDA_VISIBLE_DEVICES=0 python main.py --vlm_model llava-mixtral --enhance_method threshold_region --mix_factor 0.2
CUDA_VISIBLE_DEVICES=0 python main.py --vlm_model llava-mixtral --enhance_method threshold_region --mix_factor 0.4
CUDA_VISIBLE_DEVICES=0 python main.py --vlm_model llava-mixtral --enhance_method threshold_region --mix_factor 0.6
CUDA_VISIBLE_DEVICES=0 python main.py --vlm_model llava-mixtral --enhance_method threshold_region --mix_factor 0.8
CUDA_VISIBLE_DEVICES=0 python main.py --vlm_model llava-mixtral --enhance_method threshold_region --mix_factor 1

