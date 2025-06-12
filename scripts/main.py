# scripts/main.py
from src.rag import search_recipes
from src import config
from src.llm_interaction import generate_recipe_from_llm, review_generated_recipe
from src.image_generation import get_image_prompt_from_llm, create_image_from_prompt
from .pipelines import image_pipeline, generate_validated_recipe
from src.image_evaluation import load_clip_model

def main():
    question = input("Enter your question: ")
    ingredients = input("Enter ingredients: ")


    combined_query = f"{question} {ingredients}"
    recipes = search_recipes(combined_query, top_k=3)
    recipe = generate_validated_recipe(question, ingredients, recipes, config)

    print("\nFinal Recipe:")
    print(recipe.title)
    print(recipe.ingredients)
    print(recipe.directions)

    model, processor = load_clip_model(config.CLIP_MODEL, config.DEVICE)

    # Run image generation pipeline
    best_image = image_pipeline(f"{recipe.title} with {', '.join(recipe.ingredients)}", config, model, processor)

if __name__ == "__main__":
    main()