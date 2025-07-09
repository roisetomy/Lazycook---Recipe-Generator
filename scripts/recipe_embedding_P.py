# scripts/recipe_embedding.py

import torch
from src.data_processing import load_and_preprocess_data
from src import config
from src.embedding_utils import load_embedding_model, generate_embeddings, batch_upsert
from pinecone import Pinecone


def recipe_embedding():
    """
    Loads recipe data, generates embeddings for each recipe, and upserts them into a Pinecone vector database.
    Steps:
    1. Loads and preprocesses the recipe dataset.
    2. Loads the embedding model and generates embeddings for all recipes.
    3. Connects to Pinecone and prepares metadata for each recipe.
    4. Upserts the embeddings and metadata as vectors into the Pinecone index.
    5. Cleans up resources and empties CUDA cache if needed.
    """

    print("Running embedding and upsert...")

    # Load and preprocess data
    df = load_and_preprocess_data(config.RECIPE_DATASET_PATH)

    # Load model and generate embeddings
    model = load_embedding_model(config.EMBEDDING_MODEL, config.DEVICE)
    embeddings = generate_embeddings(model, df.full_text.tolist(), device=config.DEVICE)

    print(f"Generated embeddings for {len(df)} recipes.")

    # Connect to Pinecone
    pc = Pinecone(api_key=config.PINECONE_API_KEY)
    index = pc.Index("lazycook")

    # Convert IDs and embeddings
    ids = df["Unnamed: 0"].astype(str).tolist()
    embeddings_list = embeddings.tolist()

    # Prepare metadata per row
    metadata_list = []
    for _, row in df.iterrows():
        metadata_list.append({
            "title": row["title"],
            "ingredients": row["ingredients"],
            "directions": row["directions"]
        })

    # Prepare Pinecone vector payload
    vectors = [
        {"id": id_, "values": vec, "metadata": meta}
        for id_, vec, meta in zip(ids, embeddings_list, metadata_list)
    ]

    # Then call it:
    batch_upsert(index, vectors, namespace="recipes-namespace", batch_size=100)

    print(f"Upserted {len(vectors)} vectors to Pinecone.")

    # Cleanup
    del model, embeddings, df
    if config.DEVICE == 'cuda':
        torch.cuda.empty_cache()


if __name__ == "__main__":
    recipe_embedding()
