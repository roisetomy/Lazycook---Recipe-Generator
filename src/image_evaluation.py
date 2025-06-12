import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

def load_clip_model(model_name: str, device: str):
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    return model, processor

def compute_image_text_similarity(image: Image.Image, text: str, model, processor) -> float:
    inputs = processor(
        text=[text], 
        images=image, 
        return_tensors="pt", 
        padding=True, 
        truncation=True,
        max_length=77     
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        image_embeds = outputs.image_embeds
        text_embeds = outputs.text_embeds
    image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
    text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
    similarity = torch.matmul(text_embeds, image_embeds.T).item()
    return similarity