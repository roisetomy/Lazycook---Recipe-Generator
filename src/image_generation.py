import requests
import base64
from PIL import Image
from io import BytesIO

def get_image_prompt_from_llm(question: str, url: str, model: str) -> str:
    # ... (Implementation of get_prompt)

def create_image_from_prompt(prompt: str, url: str) -> Image.Image:
    # ... (Implementation of create_image)