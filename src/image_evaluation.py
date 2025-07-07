"""This module provides utilities for computing semantic similarity between images and
text using OpenAI's CLIP model via the Hugging Face Transformers library."""

import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

def load_clip_model(model_name: str, device: str):
    """
    Load a pre-trained CLIP model and its processor.

    Args:
        model_name (str): The Hugging Face model identifier (e.g., "openai/clip-vit-base-patch32").
        device (str): The target device to load the model on ("cuda" or "cpu").

    Returns:
        Tuple[CLIPModel, CLIPProcessor]: The loaded model and processor.
    """
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    return model, processor

def compute_image_text_similarity(image: Image.Image, text: str, model, processor) -> float:
    """
    Compute the cosine similarity between an image and a text description using CLIP.

    Args:
        image (PIL.Image.Image): The image to evaluate.
        text (str): The textual description or prompt.
        model (CLIPModel): A pre-loaded CLIP model.
        processor (CLIPProcessor): The corresponding processor for the CLIP model.

    Returns:
        float: A similarity score between the text and image embeddings.
               Higher values indicate greater similarity.
    """
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
    # Normalize the embeddings to unit vectors for cosine similarity
    image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
    text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
    similarity = torch.matmul(text_embeds, image_embeds.T).item()
    return similarity
