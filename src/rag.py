# src/recipe_search.py
from .data_processing import load_and_preprocess_data
from . import config
from .embedding_utils import load_embedding_model
from .llm_interaction import get_keywords_from_llm
import torch

# Global variables to cache loaded data
_df = None
_model_emb = None
_texts = None

def _load_data_once():
    """Load data only once and cache it"""
    global _df, _model_emb, _texts
    
    if _df is None:
        print("Loading data for the first time...")
        _df = load_and_preprocess_data(config.RECIPE_DATASET_PATH)
        _model_emb = load_embedding_model(config.EMBEDDING_MODEL, config.DEVICE)
        _texts = torch.load(config.RECIPE_EMBEDDING_PATH, map_location=config.DEVICE)
        print(f"Data loaded and cached! {_df.shape[0]} recipes, embedding shape: {_texts.shape}")
    
    return _df, _model_emb, _texts

def search_recipes(query, top_k=3):
    """
    Search for recipes based on query (always uses LLM expansion)
    
    Args:
        query (str): Combined question and ingredients
        top_k (int): Number of recipes to return
    
    Returns:
        list: List of recipe dictionaries
    """
    # Load cached data
    df, model_emb, texts = _load_data_once()
    
    # Always get extended query from LLM
    try:
        q_ext = get_keywords_from_llm(query, config.LLM_API_URL, config.LLM_MODEL)
        print(f"Extended query: {q_ext}")
        combined_query = f"{query} {q_ext}"
    except Exception as e:
        print(f"LLM expansion failed: {e}. Using original query.")
        combined_query = query
    
    # Encode the query
    question_vec = model_emb.encode([combined_query], show_progress_bar=True)
    question_vec = torch.tensor(question_vec, device=config.DEVICE)
    
    # Calculate similarities
    similarities = torch.cosine_similarity(question_vec, texts, dim=1)
    
    # Get top k results
    top_k_results = torch.topk(similarities, k=min(top_k, len(similarities)))
    top_indices = top_k_results.indices.cpu().numpy()  # Convert to CPU for pandas
    
    # Get recipes
    recipes = df.iloc[top_indices]
    recipes = recipes[["title", "ingredients", "directions"]].reset_index(drop=True)
    
    # Convert to list of dictionaries
    recipes_for_llm = recipes.to_dict(orient="records")
    
    return recipes_for_llm