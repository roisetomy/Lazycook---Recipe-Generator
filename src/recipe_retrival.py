import torch

def find_top_recipes(query_embedding, recipe_embeddings, df, top_k=3):
    """Finds the top-k most similar recipes."""
    similarities = torch.nn.functional.cosine_similarity(
        torch.tensor(query_embedding), torch.tensor(recipe_embeddings)
    )
    top_indices = torch.topk(similarities, k=top_k).indices
    return df.iloc[top_indices][["title", "ingredients", "directions"]].reset_index(drop=True)