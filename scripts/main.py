# scripts/main.py
from src.rag import search_recipes
from src import config
from src.llm_interaction import generate_recipe_from_llm, review_generated_recipe
from src.image_generation import get_image_prompt_from_llm, create_image_from_prompt
from .pipelines import image_pipeline, generate_validated_recipe
from src.image_evaluation import load_clip_model
from pinecone import Pinecone
# In your main.py file, you can now import and use the shopping agent like this:

from src.shopping_agent import create_shopping_agent

def main():
    """
    Command-line entry point for generating and reviewing recipes with LazyCook.
    
    Steps:
    1. Prompts the user for a cooking question and available ingredients.
    2. Searches for similar recipes using Pinecone and embeddings.
    3. Generates a new recipe using an LLM, with review and improvement loop.
    4. Prints the final recipe and shopping list.
    5. Loads the CLIP model and generates an image for the recipe, displaying the best match.
    6. Intelligently manages shopping list with ingredients to buy.
    """

    question = input("Enter your question: ")
    ingredients = input("Enter ingredients: ")

    # Initialize Pinecone
    pc = Pinecone(api_key=config.PINECONE_API_KEY)
    index = pc.Index("lazycook")

    recipes = search_recipes(question, ingredients, index=index, top_k=3)
    recipe, ingredients_to_buy = generate_validated_recipe(question, ingredients, recipes, config)

    print("\nFinal Recipe:")
    print(recipe.title)
    print(recipe.ingredients)
    print(recipe.directions)
    print(f"Ingredients to buy: {ingredients_to_buy}")

    # Initialize shopping list agent
    shopping_agent = create_shopping_agent()  # Will use shopping_list.txt in parent directory
    
    # Process ingredients to buy with the intelligent agent
    if ingredients_to_buy:
        print("\n🛒 Processing shopping list...")
        agent_response, _ = shopping_agent.process_ingredients(
            ingredients_to_buy, 
            f"I'm making {recipe.title} and need to buy these ingredients: {', '.join(ingredients_to_buy)}. Please check what's already on my shopping list and add what's missing."
        )
        print(f"Shopping agent: {agent_response}")
        
        # Show updated shopping list
        current_list = shopping_agent.get_current_list()
        print(f"\n📋 Updated shopping list: {current_list}")
    else:
        print("\n✅ No ingredients to buy - you have everything!")

    model, processor = load_clip_model(config.CLIP_MODEL, config.DEVICE)

    # Run image generation pipeline
    best_image = image_pipeline(f"{recipe.title} with {', '.join(recipe.ingredients)}", config, model, processor)

if __name__ == "__main__":
    main()