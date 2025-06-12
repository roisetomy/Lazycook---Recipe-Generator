from src.image_generation import create_image_from_prompt, get_image_prompt_from_llm
from src.image_evaluation import compute_image_text_similarity
from IPython.display import display
from src.llm_interaction import generate_recipe_from_llm, review_generated_recipe

def generate_validated_recipe(question, ingredients, recipes, config, max_attempts=3):
    attempt = 0
    while attempt < max_attempts:
        recipe = generate_recipe_from_llm(
            question, ingredients, recipes, config.LLM_API_URL, config.LLM_MODEL
        )
        approved, missing_ingredients = review_generated_recipe(
            question, ingredients, recipe, config.LLM_API_URL, config.LLM_MODEL
        )
        if approved:
            print("Recipe approved!")
            return recipe
        else:
            print(f"Recipe not approved (missing: {missing_ingredients})")
            attempt += 1

    print("Reached max attempts. Proceeding with the last generated recipe.")
    return recipe


def image_pipeline(recipe: str, config, model, processor):
    prompt = get_image_prompt_from_llm(recipe, config.LLM_API_URL, config.LLM_MODEL)
    similarity_scores = []
    images = []

    for i in range(3):
        image = create_image_from_prompt(prompt, config.IMAGE_API_URL)
        similarity_score = compute_image_text_similarity(image, recipe, model, processor)
        print(f"Iteration {i+1} - Similarity: {similarity_score:.4f}")
        similarity_scores.append(similarity_score)
        images.append(image)

    best_image = images[similarity_scores.index(max(similarity_scores))]
    print("Best image based on similarity score:")
    display(best_image)
    return best_image
