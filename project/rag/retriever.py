"""
RAG retrieval logic for the news dataset.

This module:
- Connects to the existing Chroma vector store built in project/vector_store/chroma_db
- Embeds an incoming user query using the same OpenAI embedding model
- Runs a similarity search over all article-level chunks (one row = one chunk)
- Applies a similarity threshold and returns the top-K most relevant chunks

The assignment requires:
- No LLM generation (we only retrieve and display context)
- Clear retrieval parameters: similarity threshold and Top-K
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List

from chromadb import PersistentClient
from openai import OpenAI


# ---------- Configuration ----------

# Chroma settings must match build_chroma_store.py
PERSIST_DIR = os.path.join(os.path.dirname(__file__), "..", "vector_store", "chroma_db")
COLLECTION_NAME = "news_articles"

# Embedding model must match what was used to build the store
EMBEDDING_MODEL = "text-embedding-3-small"

# Default retrieval parameters (can be overridden per-call)
DEFAULT_SIMILARITY_THRESHOLD = 0.25
DEFAULT_TOP_K = 4

# Global minimum similarity floor:
# if even the best matches are below this, we consider it "no relevant info".
MIN_ACCEPTABLE_SCORE = -0.55


@dataclass
class RetrievedChunk:
    """
    Simple data structure to represent a retrieved chunk.
    """

    text: str
    category: str
    date: str
    link: str
    score: float  # similarity score between query and this chunk


def get_openai_client() -> OpenAI:
    """
    Create an OpenAI client.

    Make sure you have OPENAI_API_KEY set in your environment, e.g.:
      export OPENAI_API_KEY="sk-..."
    """
    return OpenAI()


def get_chroma_collection():
    """
    Open the existing Chroma collection created by build_chroma_store.py.
    """
    client = PersistentClient(path=PERSIST_DIR)
    return client.get_collection(name=COLLECTION_NAME)


def embed_query(client: OpenAI, query: str) -> List[float]:
    """
    Embed a user query into a vector using the same embedding model as the index.
    """
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=[query])
    return resp.data[0].embedding


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> List[RetrievedChunk]:
    """
    Retrieve the most relevant chunks for a given user query.

    Steps:
    1. Embed the query text with OpenAI.
    2. Use Chroma to run a similarity search against all article-level chunks.
    3. Apply a similarity threshold to filter out weak matches.
    4. Return up to top_k chunks as RetrievedChunk objects.
    """
    if not query.strip():
        # Empty query: return nothing
        return []

    # Decide behavior based on query length (approximate "token" count by words).
    words = query.split()
    is_single_word = len(words) == 1

    # For single-word queries, we treat results as exploratory:
    # - Relax the similarity threshold a bit
    # - Show more results (5-6) to give the user something to browse
    effective_top_k = 6 if is_single_word else 3
    effective_threshold = similarity_threshold * 0.8 if is_single_word else similarity_threshold

    oa_client = get_openai_client()
    collection = get_chroma_collection()

    query_vec = embed_query(oa_client, query)

    # Ask Chroma for more results than we strictly need so we can apply
    # our own threshold filtering afterward.
    n_results = max(effective_top_k * 2, effective_top_k)

    results = collection.query(
        query_embeddings=[query_vec],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    # Chroma returns batched results: lists inside a single-element list
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    # Depending on configuration, distances may be "distance" or "1 - similarity".
    # Chroma by default uses cosine distance, so similarity = 1 - distance.
    all_candidates: List[RetrievedChunk] = []
    passed_threshold: List[RetrievedChunk] = []

    for doc, meta, dist in zip(documents, metadatas, distances):
        similarity = 1.0 - float(dist)
        chunk = RetrievedChunk(
            text=str(doc),
            category=str(meta.get("category", "")),
            date=str(meta.get("date", "")),
            link=str(meta.get("link", "")),
            score=similarity,
        )
        all_candidates.append(chunk)

        if similarity >= effective_threshold:
            passed_threshold.append(chunk)

    # If some chunks pass the threshold, use them with additional logic
    # for exploratory (single-word) queries and a global quality floor.
    if passed_threshold:
        if is_single_word:
            # Keep only chunks above the global quality floor.
            strong_enough = [c for c in passed_threshold if c.score >= MIN_ACCEPTABLE_SCORE]
            if strong_enough:
                strong_enough.sort(key=lambda c: c.score, reverse=True)
                return strong_enough[:effective_top_k]
            # If nothing is strong enough, fall through to fallback logic below.
        else:
            # Non-exploratory: also enforce the global quality floor.
            strong_enough = [c for c in passed_threshold if c.score >= MIN_ACCEPTABLE_SCORE]
            if not strong_enough:
                return []
            strong_enough.sort(key=lambda c: c.score, reverse=True)
            return strong_enough[:effective_top_k]

    # Fallback: no chunk met the threshold *or* none were strong enough
    # for exploratory (single-word) queries.
    all_candidates.sort(key=lambda c: c.score, reverse=True)

    # Apply the same global floor to all candidates.
    strong_from_all = [c for c in all_candidates if c.score >= MIN_ACCEPTABLE_SCORE]
    if not strong_from_all:
        return []

    if is_single_word:
        return strong_from_all[:effective_top_k]

    # Non-exploratory fallback: just show the top few closest chunks (respecting floor).
    return strong_from_all[:3]


if __name__ == "__main__":
    # Small manual test hook: run this file directly and type a query.
    print("Simple RAG retriever test. Type a query (or just Enter to exit).")
    while True:
        try:
            q = input("\nQuery: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not q:
            # Empty query: prompt again instead of exiting
            print("Please enter a non-empty query.")
            continue

        # For single-word queries, make it explicit that we are exploring
        # around that keyword and results may be broader / less precise.
        if len(q.split()) == 1:
            print(f"\nExploring around keyword '{q}' (results may be less precise)...")

        chunks = retrieve(q)
        if not chunks:
            print("No relevant articles found (above threshold).")
            continue

        for i, c in enumerate(chunks, start=1):
            print(f"\nResult #{i} (score={c.score:.3f})")
            print(f"Category: {c.category} | Date: {c.date}")
            print(f"Link: {c.link}")
            print("Text:")
            print(c.text)


