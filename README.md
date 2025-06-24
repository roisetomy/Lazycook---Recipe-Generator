# lazycook

LazyCook is an AI-powered recipe generator and meal assistant. It helps you create new recipes based on the ingredients you have at home and your dietary preferences, using advanced language models and image generation.

## Features
- **Recipe Generation:** Enter your available ingredients and a description of what you want to cook. LazyCook will generate a custom recipe for you.
- **Ingredient Matching:** The app intelligently matches your input ingredients to recipes and suggests what else you may need to buy.
- **Recipe Review Loop:** Recipes are validated and improved using an LLM-based reviewer, ensuring they fit your preferences and constraints.
- **Image Generation:** For each recipe, LazyCook generates a beautiful AI-created image of the dish using Stable Diffusion and CLIP similarity scoring.
- **Shopping List:** If you’re missing ingredients, LazyCook provides a shopping list for your convenience.
- **Modern UI:** Use the interactive Streamlit web app for a seamless experience.

## How It Works
1. **Input:** You provide a cooking question (e.g., "quick vegetarian dinner") and a list of ingredients you have.
2. **Recipe Search:** The app uses semantic search (Pinecone + embeddings) to find relevant recipes from a large dataset.
3. **Recipe Generation:** An LLM generates a new recipe tailored to your question and ingredients.
4. **Review & Validation:** The recipe is reviewed by another LLM for logic, dietary fit, and completeness. If needed, the process loops to improve the recipe.
5. **Image Generation:** The app creates an image prompt and generates a food image, selecting the best one using CLIP similarity.
6. **Output:** You get the recipe, a shopping list, and a generated image.

## Quickstart
1. **Clone the repository:**
   ```sh
   git clone <repo-url>
   cd lazycook
   ```
2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
3. **Set up environment variables:**
   - Copy `.env.example` to `.env` and fill in your API keys (Pinecone, Google Gemini, etc.)
4. **Run the Streamlit app:**
   ```sh
   streamlit run streamlit_app/app_cached.py
   ```

## Project Structure
- `streamlit_app/` – Streamlit UI and app logic
- `scripts/` – Pipelines and main orchestration scripts
- `src/` – Core modules: recipe search, LLM interaction, image generation, config, etc.
- `data/` – Recipe datasets and embeddings
- `notebooks/` – Jupyter notebooks for development and experiments

## Requirements
- Python 3.8+
- See `requirements.txt` for all dependencies

## Configuration
- Set your API keys and model names in the `.env` file or `src/config.py`.
- Supported LLMs: OpenAI-compatible, Gemini, etc.
- Pinecone is used for semantic search.

## Acknowledgements
- Built with [Streamlit](https://streamlit.io/), [Pinecone](https://www.pinecone.io/), [Sentence Transformers](https://www.sbert.net/), [Google Gemini](https://ai.google.dev/), and [Stable Diffusion](https://stability.ai/).

## License
MIT License