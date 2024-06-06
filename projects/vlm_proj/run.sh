#!/bin/sh

# LLaVA-LLaMa-7b
CUDA_VISIBLE_DEVICES=0 python main.py --vlm_model llava --enhance_method original
CUDA_VISIBLE_DEVICES=0 python main.py --vlm_model llava --enhance_method sobel
CUDA_VISIBLE_DEVICES=0 python main.py --vlm_model llava --enhance_method canny
CUDA_VISIBLE_DEVICES=0 python main.py --vlm_model llava --enhance_method marr_hildreth
CUDA_VISIBLE_DEVICES=0 python main.py --vlm_model llava --enhance_method threshold_region
CUDA_VISIBLE_DEVICES=0 python main.py --vlm_model llava --enhance_method active_contour

# LLaVA-LLaMa-13b
CUDA_VISIBLE_DEVICES=0 python main.py --vlm_model llava-13b --enhance_method sobel
CUDA_VISIBLE_DEVICES=0 python main.py --vlm_model llava-13b --enhance_method sobel
CUDA_VISIBLE_DEVICES=0 python main.py --vlm_model llava-13b --enhance_method canny
CUDA_VISIBLE_DEVICES=0 python main.py --vlm_model llava-13b --enhance_method marr_hildreth
CUDA_VISIBLE_DEVICES=0 python main.py --vlm_model llava-13b --enhance_method threshold_region
CUDA_VISIBLE_DEVICES=0 python main.py --vlm_model llava-13b --enhance_method active_contour

# LLaVANeX-Mixtral
CUDA_VISIBLE_DEVICES=0 python main.py --vlm_model llava-mixtral --enhance_method sobel
CUDA_VISIBLE_DEVICES=0 python main.py --vlm_model llava-mixtral --enhance_method sobel
CUDA_VISIBLE_DEVICES=0 python main.py --vlm_model llava-mixtral --enhance_method canny
CUDA_VISIBLE_DEVICES=0 python main.py --vlm_model llava-mixtral --enhance_method marr_hildreth
CUDA_VISIBLE_DEVICES=0 python main.py --vlm_model llava-mixtral --enhance_method threshold_region
CUDA_VISIBLE_DEVICES=0 python main.py --vlm_model llava-mixtral --enhance_method active_contour





