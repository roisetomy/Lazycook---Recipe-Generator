import requests
import base64
from PIL import Image
from io import BytesIO

def get_image_prompt_from_llm(recipe: str, url: str, model: str) -> str:
    headers = {"Content-Type": "application/json"}

    system_prompt = """You are a helpful AI Assistant.
    You write prompts for Stable Diffusion image generation focused exclusively on food as the main subject. Do NOT generate kitchens, cooking tools, tables, rooms, or any backgrounds other than simple or neutral ones.
    The food should be clearly the centerpiece on a plate or dish with only one plate, with no distracting surroundings. Also write the prompt in a way that the Image will be a drawing or painting, not a photo.

    Always follow this format for Stable Diffusion prompting:

    Positive prompt: Describe clearly what to generate, including style, colors, objects, environment (simple or neutral backgrounds only), mood, etc.

    Format example:

    Positive prompt: a detailed painting of a futuristic city at sunset, vibrant colors, ultra-realistic

    For prompting:"""

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
        "sampler_name": "Euler a",  # or "DPM++ 2M Karras", etc.
        "seed": -1  # -1 means random
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


    