# app.py  – USE THIS WHOLE FILE OR MERGE THE CHUNK INTO YOUR EXISTING ONE
import os, sys, warnings, asyncio, streamlit as st
from pinecone import Pinecone

if sys.platform == "win32" and (3, 8, 0) <= sys.version_info < (3, 9, 0):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

warnings.filterwarnings("ignore", message=".*no running event loop.*")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import config
from src.rag import search_recipes
from scripts.pipelines import generate_validated_recipe, image_pipeline
from src.image_evaluation import load_clip_model
from src.embedding_utils import load_embedding_model
from src.shopping_agent import create_shopping_agent

# ── cached resources ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def init_pinecone():
    return Pinecone(api_key=config.PINECONE_API_KEY).Index("lazycook")

@st.cache_resource(show_spinner=False)
def load_clip_cached():
    return load_clip_model(config.CLIP_MODEL, config.DEVICE)

@st.cache_resource(show_spinner=False)
def load_embedding_cached():
    return load_embedding_model(config.EMBEDDING_MODEL, config.DEVICE)

# ── session-state initialisation ────────────────────────────────
if "shopping_agent" not in st.session_state:
    st.session_state.shopping_agent = create_shopping_agent()

if "recipes" not in st.session_state:
    st.session_state.recipes = []          # will hold dicts {recipe, img, missing}

# ── sidebar: live shopping list ─────────────────────────────────
with st.sidebar:
    st.markdown("### 🛒 Shopping list")
    items = st.session_state.shopping_agent.get_current_list()
    st.write("*(empty)*" if not items else "\n".join(f"• {i}" for i in items))

# ── page title & inputs ─────────────────────────────────────────
st.title("LazyCook 🍳")

question = st.text_input("What kind of recipe do you want?",
                         placeholder="e.g. a healthy breakfast…")
ingredients = st.text_input("What ingredients do you have?",
                            placeholder="e.g. eggs, milk, flour…")

# ── generate button ─────────────────────────────────────────────
if st.button("Generate Recipe"):
    if not (question and ingredients):
        st.warning("Please fill both fields.")
        st.stop()

    with st.spinner("Finding inspiration…"):
        index = init_pinecone()
        _emb = load_embedding_cached()
        similar = search_recipes(question, ingredients, index=index, top_k=3)

    with st.spinner("Cooking up your recipe…"):
        recipe, missing = generate_validated_recipe(
            question, ingredients, similar, config
        )

    with st.spinner("Painting a tasty image…"):
        clip_model, clip_proc = load_clip_cached()
        img = image_pipeline(
            f"{recipe.title} with {', '.join(recipe.ingredients)}",
            config, clip_model, clip_proc
        )

    # save everything to history
    st.session_state.recipes.insert(0, {
        "recipe": recipe,
        "missing": missing,
        "image": img,
    })

# ── render all stored recipes ───────────────────────────────────
for idx, entry in enumerate(st.session_state.recipes):
    recipe = entry["recipe"]
    missing = entry["missing"]
    img     = entry["image"]

    with st.expander(f"🍽️  {recipe.title}", expanded=(idx == 0)):
        st.subheader("Ingredients")
        st.write("\n".join(f"• {i}" for i in recipe.ingredients))

        if missing:
            st.warning("Missing ingredients:\n" + "\n".join(f"• {m}" for m in missing))

            add_btn = st.button(
                f"➕ Add to shopping list ({recipe.title})",
                key=f"add_{idx}"
            )
            if add_btn:
                with st.spinner("Updating shopping list…"):
                    agent = st.session_state.shopping_agent
                    msg, _ = agent.process_ingredients(
                        missing,
                        (
                            f"I'm making {recipe.title} and need "
                            f"{', '.join(missing)}. "
                            "Check my list and add what's missing."
                        )
                    )
                st.success("Shopping list updated!")
                st.rerun()

        st.subheader("Directions")
        st.write("\n".join(f"{i+1}. {step}" for i, step in enumerate(recipe.directions)))

        if img is not None:
            st.image(img, caption=recipe.title)
        else:
            st.info("No image available.")

