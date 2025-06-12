import torch

# --- Device Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Model Configurations ---
EMBEDDING_MODEL = "avsolatorio/GIST-large-Embedding-v0"
LLM_MODEL = "qwen3-0.6b"
CLIP_MODEL = "openai/clip-vit-base-patch32"

# --- API Endpoints ---
LLM_API_URL = "http://localhost:1234/v1/chat/completions"
IMAGE_API_URL = "http://localhost:7860/sdapi/v1/txt2img"

# --- File Paths ---
RECIPE_DATASET_PATH = "data/1000000recipes.csv"
RECIPE_EMBEDDING_PATH = "data/recipe_embedding.pt"

# --- Retrieval and Generation Parameters ---
TOP_K_RECIPES = 3
IMAGE_GENERATION_COUNT = 3