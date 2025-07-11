# LazyCook

**LazyCook** is a recipe assistant that helps you cook with what you already have. It suggests meal ideas based on your available ingredients and preferences, while also offering smart suggestions, images of the dishes, and shopping lists when needed.

---

## Features

- **Custom Recipe Generation**  
  Enter ingredients and a description (e.g., "quick vegetarian dinner"), and LazyCook will generate a tailored recipe.

- **Ingredient Matching**  
  Matches your input with known recipes and tells you what you might still need.

- **Recipe Review Loop**  
  Recipes are automatically reviewed and refined using a second language model to ensure they’re complete, logical, and suitable.

- **Image Generation**  
  Creates a dish image using Stable Diffusion. The best image is selected using CLIP similarity.

- **Shopping List**  
  If you're missing anything, LazyCook creates a shopping list for you.

- **Modern UI**  
  Access LazyCook through an interactive Streamlit web app.

---

## How It Works

1. **Input**: Provide a cooking prompt and list your ingredients.
2. **Search**: The app uses semantic search (via Pinecone) to find relevant recipes.
3. **Generation**: A language model creates a custom recipe.
4. **Review**: A second pass validates and improves the recipe.
5. **Image**: A dish image is generated using Stable Diffusion.
6. **Output**: Get your recipe, image, and shopping list.

---

## Quickstart

```bash
# 1. Clone the repo
git clone <repo-url>
cd lazycook

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up environment variables
cp .env.example .env
# Then fill in your API keys

# 4. Set up LM Studio & WebUI
# - Download and install LM Studio: https://lmstudio.ai/
# - Download a compatible LLM model (e.g., Qwen3-0.3)
# - Download WebUI and make it accessible via localhost

# 5. Run the Streamlit app
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

## Data
- The data is not fully provided in this repo but can be found [here](https://huggingface.co/datasets/mbien/recipe_nlg) 

## Configuration
- Set your API keys and model names in the `.env` file or `src/config.py`.
- Supported LLMs: OpenAI-compatible, Gemini, etc.
- Pinecone is used for semantic search.

## Acknowledgements

Built with:

- [Streamlit](https://streamlit.io/)
- [Stable Diffusion](https://stability.ai/)
- [Pinecone](https://www.pinecone.io/)
- [Sentence Transformers](https://www.sbert.net/)
- [Google Gemini](https://ai.google.dev/)
- [LM Studio](https://lmstudio.ai/)
- [WebUI](https://github.com/oobabooga/text-generation-webui)
- [Recipe NLG dataset](https://huggingface.co/datasets/mbien/recipe_nlg) — for recipe data
- [Qwen](https://huggingface.co/Qwen) — multilingual and open-source large language models by Alibaba
- 
## License
MIT License
