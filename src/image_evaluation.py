import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

def load_clip_model(model_name: str, device: str):
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    return model, processor

def compute_image_text_similarity(image: Image.Image, text: str, model, processor) -> float:
    # ... (Implementation of compute_similarity)