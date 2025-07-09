"""This module provides a utility function to load and preprocess a recipe dataset
for use in downstream tasks such as embedding generation."""

import pandas as pd

def load_and_preprocess_data(file_path: str) -> pd.DataFrame:
    """
    This function reads a CSV file containing recipe data and constructs a 'full_text'
    column by concatenating the title, ingredients (from the 'NER' column), and directions.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: A DataFrame with the original data and an additional 'full_text' column.
    """
    df = pd.read_csv(file_path)

    def make_full_text(row):
        ingredients = " ".join(eval(row["NER"])) if isinstance(row["NER"], str) else ""
        directions = " ".join(eval(row["directions"])) if isinstance(row["directions"], str) else ""
        return f"{row['title']} {ingredients} {directions}"

    df["full_text"] = df.apply(make_full_text, axis=1)
    return df
