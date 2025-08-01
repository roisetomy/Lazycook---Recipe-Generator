"""This file provides functions to interact with an LLM for generating and reviewing recipes
 based on user input, including seasonal context and ingredient availability."""
import json
import re
from datetime import datetime
from typing import List
import requests
from pydantic import BaseModel, ValidationError
from src import config
from google import genai

class Recipe(BaseModel):
    """
    Pydantic class that represents a cooking recipe.

    Attributes:
        title (str): The name of the recipe.
        ingredients (List[str]): A list of ingredients required for the recipe.
        directions (List[str]): Step-by-step cooking instructions.
    """
    title: str
    ingredients: List[str]
    directions: List[str]

class ReviewResult(BaseModel):
    """
    Pydantic class that represents the result of reviewing a generated recipe.

    Attributes:
        approved (bool): Whether the recipe meets the user's requirements.
        ingredients_to_buy (List[str]): Ingredients the user does not have and needs to purchase.
        explanation (str): An explanation of the decision, including suggested improvements.
    """
    approved: bool
    ingredients_to_buy: List[str]
    explanation: str

def get_season(date):
    """
    Determine the season based on a given date.
    
    Args:
        date (datetime): Date to check
        
    Returns:
        str: Season name ('Spring', 'Summer', 'Autumn', or 'Winter')
    """
    year = 2000  # dummy leap year to handle dates like Feb 29
    seasons = {
        'Spring': (datetime(year, 3, 20), datetime(year, 6, 20)),
        'Summer': (datetime(year, 6, 21), datetime(year, 9, 22)),
        'Autumn': (datetime(year, 9, 23), datetime(year, 12, 20)),
        'Winter': (datetime(year, 12, 21), datetime(year + 1, 3, 19))
    }

    # Replace the year in the current date with year
    current = datetime(year, date.month, date.day)
    for season, (start, end) in seasons.items():
        if start <= current <= end:
            return season
    return 'Winter'  # covers Jan 1–Mar 19

def get_keywords_from_llm(question: str, url: str, model: str) -> str:
    """
    Get expanded keywords from LLM for a given question, including seasonal context.
    
    Args:
        question (str): The user's question about what they want to cook
        url (str): The LLM API endpoint URL
        model (str): The model identifier to use
        
    Returns:
        str: Comma-separated list of relevant keywords
    """
    headers = {"Content-Type": "application/json"}

    # Add seasonal context to the user's question
    current_season = get_season(datetime.now())
    question_with_context = f"{question}, season {current_season}"

    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": """You are an intelligent recipe query enrichment assistant. Your task is not to answer the user's question, but to think out loud and then output a list of highly relevant keywords related to food, cooking, ingredients, cuisines, or dish types.

    Begin your answer with a <think> block where you reason about what the user might want, and how to expand their query in a food-related context. Also take into account the current season, which is provided as a hint.

    End your answer with a comma-separated list of keywords. Do not include full sentences, explanations, or unrelated topics.

    For example:

    User: I want to eat something Italian.
    <think>
    They're probably looking for Italian food — maybe pasta, pizza, or other dishes typical of that cuisine. I will expand with some core ingredients and dish types.
    </think>
    Italian, pasta, pizza, mozzarella, tomato, olive oil, herbs, risotto

    User: {question_with_context}"

    """},
            {"role": "user", "content": question_with_context}
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

def generate_recipe_from_llm(question: str, ingredients: str, recipes: List[dict], url: str, model: str, model_big: str, feedback: str = "") -> Recipe:
    """
    Generate a new recipe using a language model based on a question, ingredients, and top recipes.

    Args:
        question (str): The user's cooking request.
        ingredients (str): Ingredients the user has.
        recipes (List[dict]): Top candidate recipes from the vector database.
        url (str): API endpoint for the LLM.
        model (str): Default model identifier.
        model_big (str): Bigger/more powerful model for use when feedback is provided.
        feedback (str, optional): Feedback from previous review to guide improvement.

    Returns:
        Recipe: Parsed and validated recipe object.
    """

    headers = {"Content-Type": "application/json"}   
    system_prompt = """You are a helpful recipe assistant. Your task is to provide a concise and relevant response based on the user's question and the ingredients they have at home.
    You should return a new recipe based on the user's question and the ingredients they have, using the top recipes from a dataset.
    Do not include any explanations or additional information, just the recipe details in valid JSON format.
    If the user specifies that he doesn't like a certain ingredient, or is allergic to it, DO NOT INCLUDE IT and replace it with something similar.

    Return ONLY a JSON object in this format:
    {
    "title": "...",
    "ingredients": ["..."],
    "directions": ["..."]
    }
    """
    model_to_use = model_big if feedback else model

    if feedback:
        system_prompt += f"\nThe last recipe was rejected for the following reason: {feedback}\nMake sure to correct this in your new recipe."

    data = {
        "model": model_to_use,
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": f"question: {question}, ingredients: {ingredients}, top recipes: {recipes}"
            }
        ],
        "temperature": 0.6,
        "max_tokens": 2048,
        "stream": False
    }

    # Call model
    response = requests.post(url, headers=headers, json=data)
    content = response.json()["choices"][0]["message"]["content"]
    print("🔍 Raw model output:\n", content)

    # Extract JSON after </think>, or fallback to parsing from full content
    match = re.search(r"</think>\s*(\{.*\})", content, re.DOTALL)
    json_block = match.group(1) if match else content.strip()

    try:
        parsed = json.loads(json_block)
        recipe = Recipe(**parsed)
        print("\n✅ Structured recipe:")
        print(recipe)
        return recipe
    except (json.JSONDecodeError, ValidationError) as e:
        print("❌ Error parsing or validating the recipe:\n", e)
        print("🔍 Raw model output:\n", content)
        raise ValueError("Invalid recipe format")


def review_generated_recipe(question: str, ingredients: str, recipe: Recipe, model: str = "gemini-1.5-flash") -> ReviewResult:
    """
    Review a generated recipe for suitability and provide feedback.

    Args:
        question (str): Original user question.
        ingredients (str): Ingredients the user has.
        recipe (Recipe): The generated recipe to review.
        model (str, optional): The model to use for review. Defaults to "gemini-1.5-flash".

    Returns:
        ReviewResult: Structured result indicating approval, missing ingredients, and explanation.
    """
    client = genai.Client(api_key=config.GOOGLE_API_KEY)

    prompt = f"""
You are a helpful recipe reviewer assistant.

Your task is to critically assess a newly generated recipe based on the user's original cooking request and the ingredients they currently have at home.

Your responsibilities are:

Determine whether the recipe logically and sensibly satisfies the user's request.

Check for any violations of dietary preferences, allergies, or other user-stated constraints.

Identify which ingredients the user needs to buy to make the recipe, based on the ingredients they already have.

Provide a clear and constructive explanation that will help a recipe-generation assistant revise the recipe in the next step.

Important:

Do not reject a recipe just because it uses ingredients the user doesn’t currently have. New ingredients are acceptable as long as they make sense and respect the user’s request.

Your goal is not to limit the recipe to only the user's current ingredients, but to help them understand what additional ingredients are needed.

Only reject a recipe if it fails to fulfill the user’s request, includes inappropriate ingredients, or violates their stated constraints.

Return a JSON object ONLY with the following structure:

{{
"approved": true or false,
"ingredients_to_buy": [list of missing ingredients, empty if none],
"explanation": "A detailed and actionable explanation for improving the recipe."
}}

Explanation Guidelines:

If the recipe violates user constraints, clearly state what those violations are and how to fix them.

Offer suggestions such as: "remove ingredient X", "substitute ingredient Y", or "adjust cooking method Z".

If the recipe is suitable but could be improved (e.g. it's bland, too complex, or inconsistent), note that too.

This explanation is meant to guide another assistant model that will revise the recipe accordingly.

Inputs:
User question: {question}
User ingredients: {ingredients}
Recipe: {recipe}
"""


    # Call the Gemini model with structured response
    response = client.models.generate_content(
        model= model,
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": ReviewResult,
        },
    )

    # Get parsed response directly as a typed Pydantic object
    review_result: ReviewResult = response.parsed
    return review_result
