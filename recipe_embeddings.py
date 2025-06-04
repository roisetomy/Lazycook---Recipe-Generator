df = pd.read_csv("100recipes.csv")

model = SentenceTransformer("avsolatorio/GIST-large-Embedding-v0")

# Combine relevant text fields into one string per recipe
def make_full_text(row):
    ingredients = " ".join(eval(row["ingredients"])) if isinstance(row["ingredients"], str) else ""
    directions = " ".join(eval(row["directions"])) if isinstance(row["directions"], str) else ""
    return f"{row['title']} {ingredients} {directions}"

df["full_text"] = df.apply(make_full_text, axis=1)

texts = model.encode(df.full_text, show_progress_bar= True)

# TO DO: Upload to database