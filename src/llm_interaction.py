import requests
import json
import re
from pydantic import BaseModel, ValidationError
from typing import List
from src import config

class Recipe(BaseModel):
    title: str
    ingredients: List[str]
    directions: List[str]

class ReviewResult(BaseModel):
    approved: bool
    missing_ingredients: List[str]

def get_keywords_from_llm(question: str, url: str, model: str) -> str:
    # ... (Implementation of get_keywords)
    url = config.LLM_API_URL
    headers = {"Content-Type": "application/json"}

    data = {
        "model": config.LLM_MODEL,
        "messages": [
            {"role": "system", "content": """""You are an intelligent recipe query enrichment assistant. Your task is not to answer the user's question, but to think out loud and then output a list of highly relevant keywords related to food, cooking, ingredients, cuisines, or dish types.

    Begin your answer with a <think> block where you reason about what the user might want, and how to expand their query in a food-related context.

    End your answer with a comma-separated list of keywords. Do not include full sentences, explanations, or unrelated topics.

    For example:

    User: I want to eat something Italian.
    <think>
    They're probably looking for Italian food — maybe pasta, pizza, or other dishes typical of that cuisine. I will expand with some core ingredients and dish types.
    </think>
    Italian, pasta, pizza, mozzarella, tomato, olive oil, herbs, risotto

    User: {question}"

    """},
            {"role": "user", "content": f"{question}"}
        ],
        "temperature": 0.1,
        "max_tokens": 1024,
        "stream": False
    }

    response = requests.post(url, headers=headers, json=data)
    print(response.json()["choices"][0]["message"]["content"])
    raw_query = response.json()["choices"][0]["message"]["content"]
    _, q_ext = raw_query.split('</think>\n\n')
    return q_ext

def generate_recipe_from_llm(question: str, ingredients: str, recipes: List[dict], url: str, model: str) -> Recipe:
    # ... (Implementation of recipe creation)
    # Set up API call
    url = config.LLM_API_URL
    headers = {"Content-Type": "application/json"}

    data = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": """You are a helpful recipe assistant. Your task is to provide a concise and relevant response based on the user's question and the ingredients they have at home.
    You should return a new recipe based on the user's question and the ingredients they have, using the top recipes from a dataset.
    Do not include any explanations or additional information, just the recipe details in valid JSON format.

    Start with <think> for reasoning. After </think>, return ONLY a JSON object in this format:
    {
    "title": "...",
    "ingredients": ["..."],
    "directions": ["..."]
    }
    """
            },
            {
                "role": "user",
                "content": f"question: {question}, ingredients: {ingredients}, top recipes: {recipes}"
            }
        ],
        "temperature": 0.1,
        "max_tokens": 2048,
        "stream": False
    }

    # Call model
    response = requests.post(url, headers=headers, json=data)
    content = response.json()["choices"][0]["message"]["content"]
    print("🔍 Raw model output:\n", content)

    # Extract JSON after </think>
    match = re.search(r"</think>\s*(\{.*\})", content, re.DOTALL)
    if match:
        raw_json = match.group(1)
        try:
            parsed = json.loads(raw_json)
            recipe = Recipe(**parsed)
            print("\n✅ Structured recipe:")
        except (json.JSONDecodeError, ValidationError) as e:
            print("❌ Error parsing or validating the recipe:\n", e)
    else:
        print("❌ Could not find JSON block after </think>.")
    return recipe


def review_generated_recipe(question: str, ingredients: str, recipe: Recipe, url: str, model: str) -> ReviewResult:
    # ... (Implementation of recipe review)
    url = config.LLM_API_URL
    headers = {"Content-Type": "application/json"}

    data = {
        "model": model,
    "messages": [
        {
            "role": "system",
            "content": """You are a helpful recipe reviewer assistant.

    Your task is to review the newly generated recipe against the user's original question and the ingredients they have at home.

    Based on the recipe ingredients, check if all ingredients are available in the user's list.

    Return a JSON object ONLY with the following fields:

    {
    "approved": true or false,
    "missing_ingredients": [list of missing ingredient names, empty if none]
    }

    - "approved" is true if the recipe makes sense and kind of matches the users question.
    - "missing_ingredients" lists any ingredients required by the recipe that the user does not have.
    - Do NOT include any explanations or extra text, only the JSON.

    Example input:
    User question: I want to cook something Italian.
    User ingredients: ["pasta", "garlic", "olive oil"]
    Recipe: ["title": "Pasta with Garlic and Olive Oil", "ingredients": ["pasta", "garlic", "olive oil", "parsley"], "directions": ["Cook pasta", "Sauté garlic", "Mix with olive oil and parsley"]}

    Expected output:
    {
    "approved": false,
    "missing_ingredients": ["tomato sauce", "basil"]
    }
    """
            },
            {
                "role": "user",
                "content": f"question: {question}, ingredients: {ingredients}, recipe: {recipe}"
            }
        ],
        "temperature": 0.1,
        "max_tokens": 2048,
        "stream": False
    }

    # Call model
    response = requests.post(url, headers=headers, json=data)
    content = response.json()["choices"][0]["message"]["content"]

    print("🔍 Raw model output:\n", content)

    # Extract JSON after optional <think> block if present
    if "<think>" in content:
        _, json_part = content.split("</think>", 1)
    else:
        json_part = content

    json_part = json_part.strip()

    try:
        review = ReviewResult.parse_raw(json_part)
        print("✅ Parsed review result:")
    except ValidationError as e:
        print("❌ Failed to parse review JSON:", e)
        print("Raw JSON content was:", json_part)
    return review.approved, review.missing_ingredients