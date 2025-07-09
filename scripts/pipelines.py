from src.image_generation import create_image_from_prompt, get_image_prompt_from_llm
from src.image_evaluation import compute_image_text_similarity
from IPython.display import display
from src.llm_interaction import generate_recipe_from_llm, review_generated_recipe

def generate_validated_recipe(question, ingredients, recipes, config, max_attempts=3):
    """
    Generate and validate a recipe using an LLM and a review loop.
    Attempts to generate a recipe that fits the user's question and available ingredients.
    If the recipe is not approved by the reviewer, it will try to improve it up to max_attempts times.
    
    Args:
        question (str): User's cooking request or description.
        ingredients (str): Ingredients the user has at home.
        recipes (list): List of top similar recipes for context.
        config: Configuration object with model/API details.
        max_attempts (int): Maximum number of review/generation attempts.
    
    Returns:
        tuple: (Recipe object, list of ingredients to buy)
    """
    attempt = 0
    last_explanation = ""

    while attempt < max_attempts:
        try:
            # Pass the previous explanation (if any) as feedback to improve the recipe
            recipe = generate_recipe_from_llm(
                question, ingredients, recipes, config.LLM_API_URL, model = config.LLM_MODEL, model_big= config.LLM_MODEL_BIG,
                feedback=last_explanation
            )
            
            review_result = review_generated_recipe(
                question, ingredients, recipe, config.LLM_MODEL_Goog
            )
            
            if review_result.approved:
                print("Recipe approved!")
                return recipe, review_result.ingredients_to_buy
            else:
                print(f" Recipe not approved (ingredients to buy: {review_result.ingredients_to_buy})")
                print(f" Explanation: {review_result.explanation}")
                last_explanation = review_result.explanation
                attempt += 1
        except Exception as e:
            print(f" Error generating recipe: {e}")
            attempt += 1

    print("Reached max attempts. Proceeding with the last generated recipe.")
    return recipe, review_result.ingredients_to_buy


def image_pipeline(recipe: str, config, model, processor, num_iterations=3):
    """
    Generate images for a recipe and pick the best one based on CLIP similarity.
    
    Args:
        recipe (str): Recipe description to generate image for
        config: Configuration object with API URLs and models
        model: CLIP model for similarity scoring
        processor: CLIP processor for image/text processing
        num_iterations (int): Number of images to generate and compare
    
    Returns:
        PIL.Image: The best matching image
    """
    prompt = get_image_prompt_from_llm(recipe, config.LLM_API_URL)
    similarity_scores = []
    images = []
    
    for i in range(num_iterations):
        image = create_image_from_prompt(prompt, config.IMAGE_API_URL)
        similarity_score = compute_image_text_similarity(image, recipe, model, processor)
        print(f"Iteration {i+1} - Similarity: {similarity_score:.4f}")
        similarity_scores.append(similarity_score)
        images.append(image)

    best_image = images[similarity_scores.index(max(similarity_scores))]
    print("Best image based on similarity score:")
    display(best_image)
    
    return best_image
