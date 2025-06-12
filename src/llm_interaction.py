import requests
import json
import re
from pydantic import BaseModel, ValidationError
from typing import List

class Recipe(BaseModel):
    title: str
    ingredients: List[str]
    directions: List[str]

class ReviewResult(BaseModel):
    approved: bool
    missing_ingredients: List[str]

def get_keywords_from_llm(question: str, url: str, model: str) -> str:
    # ... (Implementation of get_keywords)

def generate_recipe_from_llm(question: str, ingredients: str, recipes: List[dict], url: str, model: str) -> Recipe:
    # ... (Implementation of recipe creation)

def review_generated_recipe(question: str, ingredients: str, recipe: Recipe, url: str, model: str) -> ReviewResult:
    # ... (Implementation of recipe review)