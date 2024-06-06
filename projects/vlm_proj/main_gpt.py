OPANAI_KEY = "<Your Own OpenAI Key>"
META_KEY = ""
CSV_DATA_PATH = "/home/zhiyuxue/dataset/mmvp_data_questions/Answers-SDXL.csv"

from openai import OpenAI
import os
import pandas as pd
import numpy as np
import pdb
import tqdm
import base64
import requests
import argparse
from PIL import Image
from vlm_loader import load_vlm, prompt_llava, prompt_llava_mixtral
from utils import enhance_image

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
max_new_tokens = 10

parser = argparse.ArgumentParser()
parser.add_argument('--enhance_method',type=str, default='threshold_region',choices=['original','sobel','canny','marr_hildreth','threshold_region','active_contour'])
parser.add_argument("--mix_method", type=str, default='blend',choices=['blend'])
parser.add_argument("--mix_factor", type=float, default=0.5)
# parser.add_argument()
args = parser.parse_args()

# load data
df = pd.read_csv(CSV_DATA_PATH)
df = df.dropna()

# load model
headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPANAI_KEY}"
    }

ans_lst = []
for i in range(len(df)):
    row = df.iloc[i]
    img_index = row['Index']
    question = row['Question']
    choices = row['Options']
    correct_answer = row['Correct Answer']
    add_context = row['Additional Context']

    img_path = f'/home/zhiyuxue/dataset/mmvp_data_questions/MMVP Images/{int(img_index)}.jpg'
    pil_image = Image.open(img_path)

    # Image Enhancement
    method = args.enhance_method
    if method == 'active_contour':
        # Example initial points for active contour (a circular contour)
        s = np.linspace(0, 2*np.pi, 500)
        r = 110 + 140*np.sin(s)
        c = 180 + 120*np.cos(s)
        init_points = np.array([r, c]).T
        output_image = enhance_image(pil_image, method=method, init_points=init_points)
    elif method == 'hough':
        output_image = enhance_image(pil_image, method=method, canny_low_threshold=50, canny_high_threshold=150, hough_threshold=100, max_lines=50)
    elif method == 'threshold_region':
        output_image = enhance_image(pil_image, method=method, threshold_value=110, tolerance=128)
    elif method == 'sobel':
        output_image = enhance_image(pil_image, method=method, ksize=3)
    elif method == 'canny':
        output_image = enhance_image(pil_image, method=method, low_threshold=50, high_threshold=150)
    elif method == 'marr_hildreth':
        output_image = enhance_image(pil_image, method=method, sigma=3.0)
    elif method == 'original':
        output_image = enhance_image(pil_image, method=method, sigma=3.0)
    else:
        output_image = enhance_image(pil_image, method=method)
    pil_image = Image.blend(pil_image,output_image.convert('RGB'),alpha=args.mix_factor)
    pil_image.save('/home/zhiyuxue/vlm_proj/temp.png')

    base64_image = encode_image('/home/zhiyuxue/vlm_proj/temp.png')
    prompt = f'{question}\n{choices}\n{add_context}'
    payload = {
            "model": "gpt-4-turbo",
            "messages": [
                {
                "role": "user",
                "content": [
                    {
                    "type": "text",
                    "text": f'{prompt}'
                    },
                    {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpg;base64,{base64_image}"
                    }
                    }
                ]
                }
            ],
            "max_tokens": 10,
            "temperature": 0,
            "seed": 1024
        }
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    ans = response.json()['choices'][0]['message']['content']
    print(ans)
    ans_lst.append(ans[:3])

correct = 0
total = 0
for i in range(0, 300, 2):
    print(ans_lst[i])
    if ('A' in ans_lst[i] or 'a' in ans_lst[i]) and ('B' in ans_lst[i+1] or 'b' in ans_lst[i+1]):
        correct += 1
    total += 1
print('--------------')
print(correct)
print(total)
print(correct/total)
print('--------------')

# pdb.set_trace()