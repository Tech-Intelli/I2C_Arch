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
accelerator = Accelerator(cpu=False)
def clear_memory():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

def clean_mem_and_set_gradscaler():
    torch.cuda.empty_cache()
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    accelerator = Accelerator(cpu=False)
    clear_memory()
    #scaler = GradScaler()

def load_blip2_model():
    clean_mem_and_set_gradscaler()
    try:
        with autocast():
            model, vis_processors, _ = load_model_and_preprocess(
                name="blip2_t5", model_type="caption_coco_flant5xl", is_eval=True, device=device
            )
        
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
        model = accelerator.prepare(model)

    except RuntimeError as e:
        print(f"Error during model loading: {e}")
        clear_memory()
    return model, vis_processors

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
    base_model, vis_processors = load_blip2_model()
    base_model.load_state_dict(torch.load("model_state_dict_caption_coco_flant5xl.pth", map_location=device))
    print("Base Model Loding done.\nQuantization will start now")
    base_model_keys = print_model_keys(base_model)
    print("Quantization started.")
    quantized_model = torch.quantization.quantize_dynamic(
        base_model, {torch.nn.Linear}, dtype=torch.qint8
    )
    quantized_model.to(device)
    quantized_model.eval()
    print("Quantization done.")
    quantized_model_keys = print_model_keys(quantized_model)
    return quantized_model, vis_processors, base_model_keys, quantized_model_keys

"""
Try loading the saved .pth as the model, this should always work.
No architectural changes have been done up until this point
"""
#model.load_state_dict(torch.load("model_state_dict_caption_coco_flant5xl.pth", map_location=device))
accelerator = Accelerator(cpu=False)
clean_mem_and_set_gradscaler()
'''quantized_model, vis_processors, base_model_keys, quantized_model_keys = quantize_model()
#save_model(quantized_model, vis_processors, "model_state_dict_caption_coco_flant5xl_quantized.pth", "components_caption_coco_flant5xl_quantized.pkl")

with open('base_model_keys.txt', 'w') as file:
    for bkeys in base_model_keys:
        file.write(bkeys + '\n')
with open('quantized_model_keys.txt', 'w') as file:
    for qkeys in quantized_model_keys:
        file.write(qkeys + '\n')
print("Keys logging done.")'''

def initialize_quantized_blip2_model():
    model = Blip2T5(
        vit_model="eva_clip_g", 
        img_size=364,           
        drop_path_rate=0,       
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        num_query_token=32,
        t5_model="google/flan-t5-xl",
        prompt="",
        max_txt_len=32,
        apply_lemmatizer=False
    )
    model.eval()
    return model
def load_state_dict(base_state_dict, quantized_state_dict):
    filtered_state_dict = {k: v for k, v in quantized_state_dict.items() if k in base_state_dict}
    base_state_dict.update(filtered_state_dict)
    return base_state_dict
def load_quantized_model(base_model_path, quantized_model_path):
    model = initialize_quantized_blip2_model()
    base_model = torch.load(base_model_path, map_location=device)
    quantized_model = torch.load(quantized_model_path, map_location=device)
    filtered_dict  = load_state_dict(base_model, quantized_model)
    model.load_state_dict(filtered_dict)
    torch.cuda.empty_cache()
    return model

def initialize_vis_processors():
    return {
        'eval': transforms.Compose([
            transforms.Resize((364, 364)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }

torch.cuda.empty_cache()
try:
    with autocast(dtype=torch.float16):
        quantized_model = load_quantized_model("base_model_state_dict.pth", "model_state_dict_caption_coco_flant5xl_quantized.pth")
        quantized_model.to(device).half()
        quantized_model = accelerator.prepare(quantized_model)
        print(f"Model dtype after loading and before inference: {next(quantized_model.parameters()).dtype}")  # Check the dtype of model parameters
except RuntimeError as e:
        print(f"Error during model loading: {e}")
        clear_memory()

vis_processors = initialize_vis_processors()
torch.cuda.empty_cache()
print(device)

def prepare_and_process_image(image_path, vis_processors, device):
    raw_image = Image.open(image_path).convert('RGB')
    if 'eval' in vis_processors:
        preprocessed_image = vis_processors['eval'](raw_image)
        print("Device before operation:", device)
        image_tensor = preprocessed_image.unsqueeze(0).to(device).half()
        return image_tensor
    else:
        raise KeyError("'eval' key not found in vis_processors. Available keys: {}".format(vis_processors.keys()))

image_tensor = prepare_and_process_image("C:\\Users\\dasdi\\Desktop\\5.jpeg", vis_processors, device)
print(f"Tensor dtype before model inference: {image_tensor.dtype}")
with autocast(dtype=torch.float16):
    for param in quantized_model.parameters():
        param.data = param.data.half()
    quantized_model.eval()
    output = quantized_model.generate({"image": image_tensor})
    print(output)