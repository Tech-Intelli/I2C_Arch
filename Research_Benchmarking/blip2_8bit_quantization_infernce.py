import torch
import contextlib
from lavis.models.blip2_models.blip2_t5 import Blip2T5
from lavis.models.blip2_models import blip2
from PIL import Image
from torch.cuda.amp import autocast
from accelerate import Accelerator
from torchvision import transforms
import time
import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

def new_maybe_autocast(self, dtype=torch.float16):
    enable_autocast = self.device != torch.device("cpu")
    
    if enable_autocast:
        return torch.cuda.amp.autocast(dtype=torch.float16)
    else:
        return contextlib.nullcontext()

Blip2T5.maybe_autocast = new_maybe_autocast
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
accelerator = Accelerator(cpu=True)

def initialize_quantized_blip2_model(device):
    model = Blip2T5(
        vit_model="eva_clip_g", 
        img_size=364,           
        drop_path_rate=0,       
        use_grad_checkpoint=False,
        vit_precision="fp16" if device == "cuda" else "fp32",
        freeze_vit=True,
        num_query_token=32,
        t5_model="google/flan-t5-xl",
        prompt="",
        max_txt_len=32,
        apply_lemmatizer=False
    ).to(device)
    model.eval()
    return model

def initialize_vis_processors():
    return {
        'eval': transforms.Compose([
            transforms.Resize((364, 364)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }

def prepare_and_process_image(image_path, vis_processors, device):
    raw_image = Image.open(image_path).convert('RGB')
    if 'eval' in vis_processors:
        preprocessed_image = vis_processors['eval'](raw_image)
        print("Device before operation:", device)
        image_tensor = preprocessed_image.unsqueeze(0).to(device)
        if device == 'cuda':
            image_tensor = image_tensor.half()
        else:
            image_tensor = image_tensor.float()
        return image_tensor
    else:
        raise KeyError("'eval' key not found in vis_processors. Available keys: {}".format(vis_processors.keys()))

start_time_blip_loading = time.time()
print("BLIP2 model initialization started")
model = initialize_quantized_blip2_model(device)
print("BLIP2 model initialization finished.")
end_time_blip_loading = time.time() - start_time_blip_loading
print(f"Total time taken to initialize the BLIP2 model is {end_time_blip_loading} seconds.")

quantized_model_loading_start_time = time.time()
print("Loading the quantized model will start now.")
quantized_state_dict = model.load_state_dict(torch.load('blip2_caption_coco_flant5xl_8Bit_quantized_model.pth', map_location='cuda'))
quantized_state_dict.to(device).half()
print("Loading of quantized model is now finished.")
quantized_model_loading_time = time.time() - quantized_model_loading_start_time
print(f"Total time taken to load the quantized model is {quantized_model_loading_time} seconds.")

vis_processors = initialize_vis_processors()
start_time_inference = time.time()

print("Start of Inference....")
print("Creating IMage tensor.")
image_tensor = prepare_and_process_image("C:\\Users\\dasdi\\Desktop\\5.jpeg", vis_processors, device)
print("Image tensor creation is now done.")

print("Inferencing ....")
print(f"Tensor dtype before model inference: {image_tensor.dtype}")
with autocast(dtype=torch.float16):
    quantized_state_dict.eval()
    output = quantized_state_dict.generate({"image": image_tensor})
    print(output)
print("Inferencing finished.")
end_time_inference = time.time() - start_time_inference
print(f"Total time taken {end_time_inference} seconds.")