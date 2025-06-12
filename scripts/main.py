# scripts/main.py
from src.rag_pipeline import search_recipes
from src import config
from src.llm_interaction import generate_recipe_from_llm, review_generated_recipe

def main():
    # question = input("Enter your question: ")
    # ingredients = input("Enter ingredients: ")
    question = "something german"
    ingredients = "potatoes, onions, butter"


    combined_query = f"{question} {ingredients}"
    recipes = search_recipes(combined_query, top_k=3)
    
    for i, recipe in enumerate(recipes, 1):
        print(f"\n{i}. {recipe['title']}")

    recipe = generate_recipe_from_llm(question, ingredients, recipes, config.LLM_API_URL, config.LLM_MODEL)
    print(recipe.title)
    print(recipe.ingredients)
    print(recipe.directions)
    print("Recipe generated successfully!")
    max_attempts = 3
    attempt = 0

    while attempt < max_attempts:
        approved, missing_ingredients = review_generated_recipe(
            question, ingredients, recipe, config.LLM_API_URL, config.LLM_MODEL
        )
        if approved:
            print("Recipe approved!")
            break
        else:
            print("Recipe not approved.")
            attempt += 1
            if attempt < max_attempts:
                recipe = generate_recipe_from_llm(
                    question, ingredients, recipes, config.LLM_API_URL, config.LLM_MODEL
                )
            else:
                print("Reached max attempts. Proceeding with the last generated recipe.")


if __name__ == "__main__":
    main()