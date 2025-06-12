# scripts/01_preprocess_and_embed.py

import pandas as pd
import torch
import os
from src import config
from src.data_processing import load_and_preprocess_data
from src.embedding_utils import load_embedding_model, generate_embeddings

def main():
    print("Starting data preprocessing and embedding generation...")

    # Create a directory for processed data if it doesn't exist
    os.makedirs("data_processed", exist_ok=True)

    # 1. Load and process data
    df = load_and_preprocess_data(config.RECIPE_DATASET_PATH)
    print(f"Loaded and processed {len(df)} recipes.")

    # 2. Load embedding model
    embedding_model = load_embedding_model(config.EMBEDDING_MODEL, config.DEVICE)

    # 3. Generate embeddings
    print("Generating embeddings for all recipes... (This may take a while)")
    recipe_embeddings = generate_embeddings(embedding_model, df.full_text.tolist(), device=config.DEVICE)
    
    # 4. Save the processed DataFrame and the embeddings
    processed_df_path = "data_processed/recipes_processed.parquet"
    embeddings_path = "data_processed/recipe_embeddings.pt"
    
    df.to_parquet(processed_df_path)
    torch.save(torch.tensor(recipe_embeddings), embeddings_path)
    
    print(f"Processed DataFrame saved to: {processed_df_path}")
    print(f"Embeddings tensor saved to: {embeddings_path}")
    print("Preprocessing complete.")

if __name__ == "__main__":
    main()