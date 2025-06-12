# scripts/main.py

# Import from the src folder
from src.data_processing import load_and_preprocess_data
from src import config

def main():
    print("Running main program")
    df = load_and_preprocess_data(config.RECIPE_DATASET_PATH)
    print(df.head())


if __name__ == "__main__":
    main()