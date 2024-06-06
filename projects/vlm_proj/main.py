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
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
max_new_tokens = 10

parser = argparse.ArgumentParser()
parser.add_argument('--vlm_model',type=str, default='llava-mixtral',choices=['llava','ins-blip','llava-13b','llava-mixtral','mini-cpm'])
parser.add_argument('--enhance_method',type=str, default='threshold_region',choices=['original','sobel','canny','marr_hildreth','threshold_region','active_contour'])
parser.add_argument("--mix_method", type=str, default='blend',choices=['blend'])
parser.add_argument("--mix_factor", type=float, default=0.5)
# parser.add_argument()
args = parser.parse_args()

# load data
df = pd.read_csv(CSV_DATA_PATH)
df = df.dropna()

# load model
model, processor = load_vlm(model_name=args.vlm_model)

ans_lst = []
for i in range(len(df)):
    row = df.iloc[i]
    img_index = row['Index']
    question = row['Question']
    choices = row['Options']
    correct_answer = row['Correct Answer']
    add_context = row['Additional Context']

    img_path = f'/home/zhiyuxue/dataset/mmvp_data_questions/MMVP Images/{int(img_index)}.jpg'
    base64_image = encode_image(img_path)
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
    # mix
    # pil_image.save('/home/zhiyuxue/vlm_proj/test.png')
    # for k in [0,0.2,0.4,0.6,0.8,1.0]:
    #     test_image = Image.blend(pil_image,output_image.convert('RGB'),alpha=k)
    #     test_image.save(f'/home/zhiyuxue/vlm_proj/test_{k}.png')
    # pdb.set_trace()
    pil_image = Image.blend(pil_image,output_image.convert('RGB'),alpha=args.mix_factor)
    prompt = f'{question}\n{choices}\n{add_context}'
    if args.vlm_model == 'llava-mixtral':
        prompt = prompt_llava_mixtral(prompt)
    elif 'llava' in args.vlm_model:
        prompt = prompt_llava(prompt)
    else:
        prompt = prompt    
    

    if 'cpm' in args.vlm_model:
        msgs = [{'role': 'user', 'content': prompt}]
        ans = model.chat(
                image=pil_image,
                msgs=msgs,
                context=None,
                tokenizer=processor,
                temperature=1
            )
        print(ans)
        ans_lst.append(ans)
        # pdb.set_trace()
    else:
        inputs = processor(text=prompt, images=pil_image, return_tensors="pt").to('cuda')
        input_ids = inputs['input_ids'].reshape(1, -1)
        input_length = input_ids.shape[1]
        if 'blip' in args.vlm_model:
            generate_ids = model.generate(
                **inputs,
                do_sample=False,
                num_beams=5,
                max_length=256,
                min_length=1,
                top_p=0.9,
                repetition_penalty=1.5,
                length_penalty=1.0,
                temperature=1,
            )
        else:
            generate_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
        # pdb.set_trace()
        # processor.batch_decode(generate_ids, skip_special_tokens=True)[0]
        ans = processor.batch_decode(generate_ids[:,input_ids.shape[-1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        # pdb.set_trace()
        print(ans)
        ans_lst.append(ans)

correct = 0
total = 0
correct_answer = df['Correct Answer'].tolist()
for i in range(0, 300, 2):
    print(ans_lst[i])
    if 'cpm' in args.vlm_model:
        if '(a)' in ans_lst[i] and '(b)' in ans_lst[i+1]:
            correct += 1
    else:
        if ('A' in ans_lst[i] or 'a' in ans_lst[i] or ans_lst[i] in correct_answer[i]) and ('B' in ans_lst[i+1] or 'b' in ans_lst[i+1] or ans_lst[i+1] in correct_answer[i+1]):
            correct += 1
    total += 1
print('--------------')
print(correct)
print(total)
print(correct/total)
print('--------------')
