import streamlit as st
import sys
import os
import warnings
from pinecone import Pinecone

# Suppress warnings
warnings.filterwarnings('ignore', message='.*no running event loop.*')
warnings.filterwarnings('ignore', message='.*Trying to unpickle estimator.*')
warnings.filterwarnings('ignore', message=".*'torch.classes'.*")

# Ensure the root directory is on the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.rag import search_recipes
from src import config
from src.image_evaluation import load_clip_model
from scripts.pipelines import image_pipeline, generate_validated_recipe

def main():
    st.title("LazyCook Recipe Generator")
    
    # Initialize Pinecone
    pc = Pinecone(api_key=config.PINECONE_API_KEY)
    index = pc.Index("lazycook")
    
    # Load CLIP model
    model, processor = load_clip_model(config.CLIP_MODEL, config.DEVICE)
    
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
                recipe = generate_validated_recipe(question, ingredients, recipes, config)
            
            # Display recipe
            st.header(recipe.title)
            
            st.subheader("Ingredients:")
            for ingredient in recipe.ingredients:
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
