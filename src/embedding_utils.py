from sentence_transformers import SentenceTransformer

def load_embedding_model(model_name: str, device: 'cuda'):
    """Loads the sentence transformer model."""
    return SentenceTransformer(model_name, device=device)

def generate_embeddings(model, texts, batch_size=32, device='cuda'):
    """Generates embeddings for a list of texts."""
    return model.encode(texts, show_progress_bar=True, batch_size=batch_size, device=device)