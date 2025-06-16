# src/recipe_search.py
from .data_processing import load_and_preprocess_data
from . import config
from .embedding_utils import load_embedding_model
from .llm_interaction import get_keywords_from_llm
from pinecone import Pinecone
import numpy as np

# Load the embedding model globally
_model_emb = load_embedding_model(config.EMBEDDING_MODEL, config.DEVICE)

def search_recipes(query: str, ingredients: str, index: Pinecone, top_k: int = 3) -> list:
    """
    Search for recipes using Pinecone vector search with weighted query combination.

    Args:
        query (str): The user's question about what they want to cook
        ingredients (str): Available ingredients
        index (Pinecone): Pinecone index instance
        top_k (int): Number of recipes to return

    Returns:
        list: List of dictionaries with recipe info
    """
    # Step 1: Get enriched query using LLM
    q_ext = get_keywords_from_llm(query, config.LLM_API_URL, config.LLM_MODEL)
    query_text1 = query + " " + ingredients
    query_text2 = q_ext

    # Step 2: Embed both queries using the embedding model
    query_vector1 = _model_emb.encode(query_text1, show_progress_bar=False).tolist()
    query_vector2 = _model_emb.encode(query_text2, show_progress_bar=False).tolist()

    # Step 3: Combine vectors with weights (70% original query, 30% enriched query)
    query_vector = (0.7 * np.array(query_vector1) + 0.3 * np.array(query_vector2)).tolist()

    # Step 4: Search Pinecone
    results = index.query(
        vector=query_vector,
        top_k=top_k,
        namespace="recipes-namespace",
        include_metadata=True
    )

    # Step 5: Format results
    recipes_for_llm = []
    for match in results["matches"]:
        metadata = match["metadata"]
        recipes_for_llm.append({
            "title": metadata.get("title", ""),
            "ingredients": metadata.get("ingredients", ""),
            "directions": metadata.get("directions", "")
        })

    return recipes_for_llm

