# src/data_processing.py

import pandas as pd

def load_and_preprocess_data(file_path: str) -> pd.DataFrame:
    """Loads the recipe dataset and prepares it for embedding."""
    df = pd.read_csv(file_path)

    def make_full_text(row):
        ingredients = " ".join(eval(row["NER"])) if isinstance(row["NER"], str) else ""
        directions = " ".join(eval(row["directions"])) if isinstance(row["directions"], str) else ""
        return f"{row['title']} {ingredients} {directions}"

    df["full_text"] = df.apply(make_full_text, axis=1)
    return df