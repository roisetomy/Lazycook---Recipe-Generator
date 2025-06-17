import requests
import base64
from PIL import Image
from io import BytesIO

def get_image_prompt_from_llm(recipe: str, url: str, model: str) -> str:
    headers = {"Content-Type": "application/json"}

    system_prompt = """You are a helpful AI Assistant.
You write prompts for Stable Diffusion image generation, focused exclusively on food as the main subject.

Rules to follow:

Do NOT include kitchens, cooking tools, utensils, tables, people, or any detailed background elements.

Use only simple, neutral, or minimal backgrounds (e.g. plain color, subtle texture).

The food should be the centerpiece, placed clearly on a single plate or dish. No multiple plates or extra items.

Style must be a drawing or painting, NOT photorealistic. Emphasize illustration quality (e.g., watercolor, gouache, pencil sketch, digital painting, etc.).

Format:

Positive prompt: a [style] drawing/painting of [dish/food item], with [visual details: shape, texture, colors], on a single plate, on a simple background, [mood or lighting if relevant]

Example:

Positive prompt: a watercolor painting of a slice of strawberry cheesecake, creamy texture with bright red strawberries on top, on a white ceramic plate, placed on a soft beige background, warm and inviting"""

    data = {
        "model": "qwen3-0.6b",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": recipe}
        ],
        "temperature": 0.1,
        "max_tokens": 1024,
        "stream": False
    }

    response = requests.post(url, headers=headers, json=data)
    answer = response.json()["choices"][0]["message"]["content"].strip()

    if "</think>" in answer:
        clean_answer = answer.split("</think>")[-1].strip()
    else:
        clean_answer = answer.strip()

    print(clean_answer)

    prompt = clean_answer.replace("Positive prompt:", "").strip()
    return prompt


def create_image_from_prompt(prompt: str, url: str) -> Image.Image:
    payload = {
        "prompt": prompt,
        "negative_prompt": "blurry, low resolution, watermarks, text, logo, signature, bad anatomy, bad hands, bad proportions, ugly, duplicate, morbid, mutilated, out of frame, extra digit, fewer digits, cropped, worst quality, low quality",
        "steps": 30,
        "cfg_scale": 7,
        "width": 1024,
        "height": 512,
        "sampler_name": "Euler a",  # "DPM++ 2M Karras"
        "seed": -1 
    }

    response = requests.post(url, json=payload)
    result = response.json()

    # Decode the image
    image_base64 = result['images'][0]
    image_bytes = base64.b64decode(image_base64)
    image = Image.open(BytesIO(image_bytes))

    # Display inline in Jupyter
    # display(image)
    return image


    