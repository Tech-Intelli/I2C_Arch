"""
Author: Dipanjan Das
Disclaimer: This code and concept is entirely written by Dipanjan Das,
ChatGPT is not yet updated about BLIP2 and BLIP2 Quantization
"""

"""
import all needed packages
"""

import os
import torch
import pickle
import contextlib
from PIL import Image
from lavis.models import load_model_and_preprocess
from lavis.models.blip2_models.blip2_t5 import Blip2T5
from torch.cuda.amp import autocast, GradScaler
from torchvision import transforms
from accelerate import Accelerator


def new_maybe_autocast(self, dtype=torch.float16):
    enable_autocast = self.device != torch.device("cpu")
    
    if enable_autocast:
        return torch.cuda.amp.autocast(dtype=torch.float16)
    else:
        return contextlib.nullcontext()

Blip2T5.maybe_autocast = new_maybe_autocast
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
accelerator = Accelerator(cpu=True)

def clear_memory():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

def clean_mem_and_set_gradscaler():
    torch.cuda.empty_cache()
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    accelerator = Accelerator(cpu=True)
    clear_memory()

def load_blip2_model():
    clean_mem_and_set_gradscaler()
    try:
        with autocast():
            model, _, _ = load_model_and_preprocess(
                name="blip2_t5", model_type="caption_coco_flant5xl", is_eval=True, device=device
            )
        
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
        model = accelerator.prepare(model)

    except RuntimeError as e:
        print(f"Error during model loading: {e}")
        clear_memory()
    return model

def print_model_keys(model):
    return model.state_dict().keys()

def save_model(model, vis_processors, model_path, components_path):
    torch.save(model.state_dict(), model_path)
    components = {
        'vis_processors': vis_processors
    }
    with open(components_path, "wb") as f:
        pickle.dump(components, f)

def quantize_model():
    clean_mem_and_set_gradscaler()
    print("Cleaning done.")
    base_model = load_blip2_model()
    print("Base Model Loading done.\nQuantization will start now")

    quantized_model = torch.quantization.quantize_dynamic(
        base_model, {torch.nn.Linear}, dtype=torch.qint8
    )
    quantized_model.to(device)
    quantized_model.eval()
    print("Quantization done.")

    return quantized_model
accelerator = Accelerator(cpu=True)
clean_mem_and_set_gradscaler()
quantized_state_dict = quantize_model()

torch.save(quantized_state_dict, 'blip2_caption_coco_flant5xl_8Bit_quantized_model.pth')
