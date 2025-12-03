import pandas as pd

PATH = "project/data/News_Category_Dataset_v3.json"

# Read JSONL as a table
df = pd.read_json(PATH, lines=True)

# Ensure date stays as plain text (string), not a datetime or numeric type
df["date"] = df["date"].astype(str)

# Build the main text field for embeddings (no links here)
# Example format: "<headline>. <short_description> [Category: <category>, Date: <date>]"
df["text"] = (
    df["headline"].fillna("").astype(str).str.strip()
    + ". "
    + df["short_description"].fillna("").astype(str).str.strip()
    + " [Category: "
    + df["category"].fillna("").astype(str)
    + ", Date: "
    + df["date"]
    + "]"
)

# Keep a small subset to stay well within the assignment limits
# (You can adjust n if you want a different size, but keep it <= 200)
# 130 rows * ~220 characters per row ≈ 28,600 characters (< 30,000 limit)
df_subset = df[df["short_description"].notna()].sample(n=130, random_state=42)

# Clean table ready for Chroma:
# - text: what will be embedded
# - category, date: text metadata
# - link: kept only as metadata (not part of the text)
rag_table = df_subset[["text", "category", "date", "link"]].reset_index(drop=True)

print("RAG table preview:")
print(rag_table.head())
print("\nColumns:", rag_table.columns.tolist())

# Optionally, save the clean table for later use
rag_table.to_csv("project/data/news_rag_table.csv", index=False)