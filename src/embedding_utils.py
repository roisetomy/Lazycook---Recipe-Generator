"""This module provides utilities for loading a sentence transformer model,
generating text embeddings, and uploading (upserting) them in batches
to a vector database index"""
from sentence_transformers import SentenceTransformer

def load_embedding_model(model_name: str, device: 'cuda'):
    """
    Load a sentence transformer model for generating embeddings.

    Args:
        model_name (str): Name or path of the model
        device (str): Device to load the model on ('cpu' or 'cuda').

    Returns:
        SentenceTransformer: The loaded sentence transformer model.
    """
    return SentenceTransformer(model_name, device=device)

def generate_embeddings(model, texts, batch_size=128, device='cuda'):
    """
    Generate embeddings for one or more text inputs using a sentence transformer model.

    Args:
        model (SentenceTransformer): Preloaded sentence transformer model.
        texts (str or List[str]): A single string or a list of strings to encode.
        batch_size (int, optional): Batch size for processing. Defaults to 128.
        device (str, optional): Device to use for encoding. Defaults to 'cuda'.

    Returns:
        np.ndarray: An array of vector embeddings.
    """
    return model.encode(texts, show_progress_bar=True, batch_size=batch_size, device=device)

def batch_upsert(index, vectors, namespace, batch_size=100):
    """
    Upload vector embeddings to a vector index in batches.

    Args:
        index: The target Pinecone index.
        vectors (list): List of vectors.
        namespace (str): The namespace under which to store vectors.
        batch_size (int, optional): Number of vectors to upload per batch. Defaults to 100.

    Returns:
        None
    """
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i+batch_size]
        index.upsert(vectors=batch, namespace=namespace)
