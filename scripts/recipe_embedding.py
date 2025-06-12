# scripts/recipe_embedding.py
import os
import torch
from src.data_processing import load_and_preprocess_data
from src import config
from src.embedding_utils import load_embedding_model, generate_embeddings

def recipe_embedding():
    print("Running embedding program")
    
    # Load and preprocess data
    df = load_and_preprocess_data(config.RECIPE_DATASET_PATH)
    
    # Load model and generate embeddings
    model = load_embedding_model(config.EMBEDDING_MODEL, config.DEVICE)
    embeddings = generate_embeddings(model, df.full_text.tolist(), device=config.DEVICE)
    
    print(f"Generated embeddings for {len(df)} recipes.")
    
    # Prepare output path
    embeddings_path = os.path.join(config.RECIPE_EMBEDDING_PATH)
    os.makedirs("data", exist_ok=True)

    
    # Save embeddings (check if already tensor)
    if not isinstance(embeddings, torch.Tensor):
        embeddings = torch.tensor(embeddings)
    
    torch.save(embeddings, embeddings_path)
    print(f"Embeddings saved to {embeddings_path}")

    # Cleanup
    del model, embeddings, df
    if config.DEVICE == 'cuda':
        torch.cuda.empty_cache()

if __name__ == "__main__":
    recipe_embedding()