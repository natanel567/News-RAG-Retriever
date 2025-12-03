"""
Build a Chroma vector database from the prepared news_rag_table.csv file.

This script:
- Loads the CSV we created earlier (news_rag_table.csv)
- Calls OpenAI's embedding API (e.g., text-embedding-3-small) on the `text` column
- Stores embeddings + metadata in a persistent Chroma collection under project/vector_store/

Run this script once (or whenever the CSV changes) to rebuild the vector store.
"""

import os
import uuid

import pandas as pd
from chromadb import PersistentClient
from openai import OpenAI


# ---------- Configuration ----------

# Path to the cleaned CSV we prepared earlier
CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "news_rag_table.csv")

# Directory where Chroma will store its persistent data
PERSIST_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")

# Name of the Chroma collection
COLLECTION_NAME = "news_articles"

# OpenAI embedding model to use (required by the assignment PDF)
# See: text-embedding-3-small recommended in the spec.
EMBEDDING_MODEL = "text-embedding-3-small"


def get_openai_client() -> OpenAI:
    """
    Create an OpenAI client.

    Make sure you have OPENAI_API_KEY set in your environment, e.g.:
      export OPENAI_API_KEY="sk-..."
    """
    return OpenAI()


def load_data(csv_path: str) -> pd.DataFrame:
    """
    Load the prepared CSV into a pandas DataFrame.
    We expect columns: text, category, date, link.
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        raise

    # Basic sanity checks to fail fast if something is off
    required_cols = {"text", "category", "date", "link"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    return df


def embed_texts(client: OpenAI, texts: list[str]) -> list[list[float]]:
    """
    Call OpenAI's embedding API on a list of texts and return a list of vectors.
    """
    # OpenAI's API supports batching; for this small dataset we can embed in one call,
    # but we keep the function flexible if you want to batch later.
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in response.data]


def build_chroma_collection(df: pd.DataFrame) -> None:
    """
    Create (or recreate) a Chroma collection and populate it with embeddings + metadata.
    """
    # Initialize Chroma client with a persistent directory
    client = PersistentClient(path=PERSIST_DIR)

    # For a clean rebuild each time, drop the collection if it exists, then recreate it.
    # This avoids issues with `delete(where={})` on newer Chroma versions.
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        # It's fine if the collection didn't exist yet.
        pass

    # Create a fresh, empty collection that will hold our vectors
    collection = client.create_collection(name=COLLECTION_NAME)

    # Prepare the data for embedding:
    # - `texts` is what we embed
    # - `metadatas` holds category/date/link for each item
    # - `ids` are Chroma's document IDs (we'll generate UUIDs)
    texts: list[str] = df["text"].astype(str).tolist()
    metadatas: list[dict] = []
    ids: list[str] = []

    for _, row in df.iterrows():
        metadatas.append(
            {
                "category": str(row["category"]),
                "date": str(row["date"]),
                "link": str(row["link"]),
            }
        )
        ids.append(str(uuid.uuid4()))

    # Compute embeddings using OpenAI
    client_oa = get_openai_client()
    embeddings = embed_texts(client_oa, texts)

    # Add everything to Chroma
    collection.add(documents=texts, metadatas=metadatas, ids=ids, embeddings=embeddings)

    print(f"Built Chroma collection '{COLLECTION_NAME}' with {len(texts)} items.")
    print(f"Chroma data persisted under: {PERSIST_DIR}")


def main() -> None:
    df = load_data(CSV_PATH)
    build_chroma_collection(df)


if __name__ == "__main__":
    main()


