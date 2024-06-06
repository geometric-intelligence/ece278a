from transformers import AutoProcessor, LlavaForConditionalGeneration
import torch
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from transformers import BitsAndBytesConfig
from transformers import AutoModel, AutoTokenizer
from accelerate import infer_auto_device_map

"""
Prompt for VG
"""
def prompt_llava(prompt):
    return "USER: <image>\n<prompt> ASSISTANT:".replace('<prompt>',prompt)

def prompt_llava_mixtral(prompt):
    return "[INST] <image>\n<prompt> [/INST]".replace('<prompt>',prompt)

# def 




"""
VLM for VQA
"""

def load_vlm(model_name, cache_dir='/home/zhiyuxue/models'):
    """
    Load a vision-language model from Hugging Face and optionally specify a directory to cache the model.

    Parameters:
    - model_name (str): The name of the model to load.
    - cache_dir (str, optional): The directory to cache the model parameters.

    Returns:
    - model: The loaded model.
    - tokenizer: The corresponding tokenizer for the model.
    """
    # quantization_config = BitsAndBytesConfig(
    #         load_in_4bit=True,
    #         bnb_4bit_compute_dtype=torch.float16)
    if model_name == 'llava':
        model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf",cache_dir=cache_dir,torch_dtype=torch.float16, device_map="auto")
        processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf",cache_dir=cache_dir)
    elif model_name == 'ins-blip':
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
        # model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-vicuna-7b",cache_dir=cache_dir,torch_dtype=torch.float16, device_map="auto")
        model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-vicuna-7b",cache_dir=cache_dir,torch_dtype=torch.float16).to("cuda")
        processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b",cache_dir=cache_dir)
    elif model_name == 'llava-13b':
        model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-13b-hf",cache_dir=cache_dir,torch_dtype=torch.float16, device_map="auto")
        processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-13b-hf",cache_dir=cache_dir)
    elif model_name == 'llava-mixtral':
        model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", cache_dir=cache_dir, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="auto") 
        processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", cache_dir=cache_dir)
    elif model_name == 'mini-cpm':
        model = AutoModel.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5', trust_remote_code=True, torch_dtype=torch.float16, cache_dir=cache_dir, device_map="auto")
        processor = AutoTokenizer.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5', trust_remote_code=True)
        model = model.to(device='cuda')
    return model, processor