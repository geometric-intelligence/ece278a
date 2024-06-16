#!/bin/sh

# 'original','sobel','canny','marr_hildreth','threshold_region','active_contour'
python main_gpt.py --enhance_method original
python main_gpt.py --enhance_method sobel
python main_gpt.py --enhance_method canny
python main_gpt.py --enhance_method marr_hildreth
python main_gpt.py --enhance_method active_contour

