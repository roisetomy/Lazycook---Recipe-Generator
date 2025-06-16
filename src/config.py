import torch
import os
from dotenv import load_dotenv
load_dotenv()

# --- Device Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Model Configurations ---
EMBEDDING_MODEL = "avsolatorio/GIST-Embedding-v0"
EMBEDDING_MODEL_UNUSED = "avsolatorio/GIST-large-Embedding-v0"
EMBEDDING_MODEL_UNUSED= "Qwen/Qwen3-Embedding-0.6B"
LLM_MODEL = "qwen3-0.6b"
LLM_MODEL_BIG = "qwen3-4b"
LLM_MODEL_Goog = "gemini-1.5-flash"
CLIP_MODEL = "openai/clip-vit-base-patch32"

# --- API Endpoints ---
LLM_API_URL = "http://localhost:1234/v1/chat/completions"
IMAGE_API_URL = "http://localhost:7860/sdapi/v1/txt2img"

# --- API Keys ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# --- File Paths ---
# Convert relative paths to absolute paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RECIPE_DATASET_PATH = os.path.join(ROOT_DIR, "data", "1000000recipes.csv")
RECIPE_EMBEDDING_PATH = os.path.join(ROOT_DIR, "data", "recipe_embedding.pt")

# --- Retrieval and Generation Parameters ---
TOP_K_RECIPES = 3
IMAGE_GENERATION_COUNT = 3