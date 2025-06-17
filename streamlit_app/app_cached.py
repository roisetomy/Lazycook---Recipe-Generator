import streamlit as st
import sys
import os
import warnings
import torch
import asyncio
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

# Fix for asyncio on Windows
if sys.platform == "win32" and (3, 8, 0) <= sys.version_info < (3, 9, 0):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Suppress warnings
warnings.filterwarnings('ignore', message='.*no running event loop.*')
warnings.filterwarnings('ignore', message='.*Trying to unpickle estimator.*')
warnings.filterwarnings('ignore', message=".*'torch.classes'.*")

# Ensure the root directory is on the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.rag import search_recipes
from src import config
from src.image_evaluation import load_clip_model
from src.embedding_utils import load_embedding_model
from scripts.pipelines import image_pipeline, generate_validated_recipe

@st.cache_resource(show_spinner=False)
def init_pinecone():
    """Initialize and cache Pinecone connection"""
    print("Initializing Pinecone connection for the first time...")
    pc = Pinecone(api_key=config.PINECONE_API_KEY)
    return pc.Index("lazycook")

@st.cache_resource(show_spinner=False)
def load_embedding_cached():
    """Load and cache embedding model for recipe search"""
    print("Loading embedding model for the first time...")
    try:
        return load_embedding_model(config.EMBEDDING_MODEL, config.DEVICE)
    except RuntimeError as e:
        if "no running event loop" in str(e):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return load_embedding_model(config.EMBEDDING_MODEL, config.DEVICE)
        raise

@st.cache_resource(show_spinner=False)
def load_clip_cached():
    """Load and cache CLIP model for image similarity"""
    print("Loading CLIP model for the first time...")
    try:
        return load_clip_model(config.CLIP_MODEL, config.DEVICE)
    except RuntimeError as e:
        if "no running event loop" in str(e):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return load_clip_model(config.CLIP_MODEL, config.DEVICE)
        raise

def main():
    st.title("LazyCook Recipe Generator")
    
    # Initialize all services with caching
    with st.spinner("Loading models..."):
        # Initialize all models and services using cached functions
        index = init_pinecone()
        model, processor = load_clip_cached()
    
    # User inputs
    question = st.text_input("What kind of recipe are you looking for?", 
                           placeholder="E.g., a healthy breakfast, quick dinner, vegetarian meal...")
    
    ingredients = st.text_input("What ingredients do you have?",
                              placeholder="E.g., eggs, milk, flour, sugar...")
    
    if st.button("Generate Recipe"):
        if question and ingredients:
            with st.spinner("Searching for recipes..."):
                recipes = search_recipes(question, ingredients, index=index, top_k=3)
            
            with st.spinner("Generating your perfect recipe..."):
                recipe, ingredients_to_buy = generate_validated_recipe(question, ingredients, recipes, config)
            
            # Display recipe sections
            st.header(recipe.title)
            
            st.subheader("Ingredients:")
            for ingredient in recipe.ingredients:
                st.write(f"• {ingredient}")

            if ingredients_to_buy:
                st.subheader("Shopping List:")
                for ingredient in ingredients_to_buy:
                    st.write(f"• {ingredient}")
            
            st.subheader("Directions:")
            for i, step in enumerate(recipe.directions, 1):
                st.write(f"{i}. {step}")
            
            # Generate image
            with st.spinner("Creating a delicious image for your recipe..."):
                image = image_pipeline(f"{recipe.title} with {', '.join(recipe.ingredients)}", 
                                    config, model, processor)
                if image:
                    st.image(image, caption=recipe.title)
                else:
                    st.warning("Could not generate an image for this recipe.")
        else:
            st.warning("Please provide both a question and ingredients.")

if __name__ == "__main__":
    main()
