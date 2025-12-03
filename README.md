## SoluGenAI Home Assignment – News RAG Retriever

This repository implements the retrieval component of a simple RAG system, following the requirements in the home assignment.

To view the original database, please download it from the following link: https://www.kaggle.com/datasets/rmisra/news-category-dataset?resource=download

If for any reason the link doesn't work, I used the database called "News Category Dataset" from Kaggle

I didn't upload it here because even when archived its 27MB and GitHub allows only 25MB file size max.

After downloading, make sure to unzip the file in the data repository using the jsonl_conversion_to_table.py file

The system:
- Ingests a small news dataset derived from a Kaggle JSONL file.
- Builds OpenAI embeddings and stores them in a Chroma vector database.
- Exposes a minimal Flask + HTML UI where users can type queries and see the most relevant news snippets (no LLM generation).

### 1. Project Structure

- `project/data/`
  - `News_Category_Dataset_v3.json` – original HuffPost news JSONL dataset.
  - `jsonl_conversion_to_table.py` – script to:
    - Load the JSONL file.
    - Build a small  subset (`≤200` rows, `<30,000` characters).
    - Create `news_rag_table.csv` with columns: `text`, `category`, `date`, `link`.
  - `news_rag_table.csv` – final tabular dataset used for embeddings and retrieval.

- `project/vector_store/`
  - `build_chroma_store.py` – script to:
    - Read `news_rag_table.csv`.
    - Call OpenAI embeddings (`text-embedding-3-small`) on the `text` column.
    - Persist vectors + metadata (`category`, `date`, `link`) into a Chroma collection.
  - `chroma_db/` – Chroma’s on-disk vector store (generated; can be deleted and rebuilt).

- `project/rag/`
  - `retriever.py` – RAG retrieval logic:
    - Loads the Chroma collection.
    - Embeds the user query.
    - Applies a similarity threshold and Top‑K.
    - Special handling for short/one-word “exploratory” queries.

- `project/backend/`
  - `app.py` – Flask backend:
    - `GET /` – render search form.
    - `POST /search` – run retrieval and display results.
  - `templates/index.html` – minimal HTML/CSS UI:
    - Input field for queries.
    - Shows retrieved chunks, similarity scores, rank, category, date, and link.

### 2. Setup Instructions

#### 2.1. Python environment

Use Python 3.9+ (the system was developed and tested with Python 3.9).

From the repository root:



Install the minimal dependencies:

```bash
pip3 install pandas chromadb openai flask
```

#### 2.2. OpenAI API key

Set the `OPENAI_API_KEY` environment variable before running any scripts that call the OpenAI API:

```bash
export OPENAI_API_KEY="sk-your-key-here"
```

### 3. Data Preparation Pipeline

1. **Convert JSONL to CSV subset**

  **If you want to do this step, make sure you downloaded the database from the link above.**

   ```bash
   cd project
   python3 data/jsonl_conversion_to_table.py
   ```

   This creates/overwrites `project/data/news_rag_table.csv` with:
   - Exactly one row per news article.
   - A `text` field: `"<headline>. <short_description> [Category: <category>, Date: <date>]"`.
   - Total text length under 30,000 characters.
   - 131 rows in the table.

3. **Build the Chroma vector store**

   ```bash
   cd project
   python3 vector_store/build_chroma_store.py
   ```

   This:
   - Embeds each `text` row with `text-embedding-3-small` (OpenAI).
   - Stores embeddings + metadata into a persistent Chroma collection (`news_articles`) under `project/vector_store/chroma_db/`.

   If the vector store becomes corrupted or you change the CSV, you can:
   - Delete `project/vector_store/chroma_db/`.
   - Re-run `build_chroma_store.py`.


### 4. Running the Minimal UI

1. Ensure the Chroma store is built (see section 3).
2. From the `project` directory, run the Flask app:

   ```bash
   cd project
   python3 backend/app.py
   ```

3. Open a browser and navigate to:

   - `http://localhost:8000`

4. Interact with the UI:

   - Enter a query (word or phrase) in the text box and click **Search** or press Enter.
   - The page displays:
     - A status message (e.g. “Exploring around keyword 'travel'…”, or “Showing the most relevant articles…”).
     - A list of retrieved results, each with:
       - Rank (`#1`, `#2`, …).
       - **Similarity score** (derived from cosine distance, higher is more similar).
       - **Category** and **Date**.
       - The full `text` (headline + short description).
       - A **Source** link to the original article.

### 5. RAG Design Choices

#### 5.1. Dataset and chunking strategy

- Source: HuffPost News Category dataset (JSONL) from a Kaggle-like source.
- We build a **small subset**:
  - 130 rows.
  - Total `text` length ≈ 28,500 characters (< 30,000 char requirement).
- **Chunking strategy**: one row = one chunk.
  - Each row already represents a short, coherent news snippet (headline + short summary).
  - Further splitting would fragment already small texts and hurt retrieval quality.
  - Category and date are appended in brackets for additional context but not used for chunking.

#### 5.2. Vector database: Chroma

- Chosen over Pinecone, Weaviate, Mongo, and pgvector because:
  - It is lightweight, local, and Python-first.
  - Requires no external infrastructure or cloud account.
  - Perfectly matches the small dataset and simple retrieval requirements of the assignment.

#### 5.3. Embeddings

- Model: `text-embedding-3-small` (OpenAI), as recommended in the assignment PDF.
- Rationale:
  - Very low cost and latency.
  - Quality is more than sufficient for short news snippets.
  - Keeps total cost well under the \$1 limit, even including some extra test queries.

### 6. Retrieval Parameters

All logic is implemented in `project/rag/retriever.py` and reused by the CLI and Flask backend.

- **Similarity / distance**:
  - Chroma returns cosine distances; we convert to a similarity-like score:
    - `similarity = 1.0 - distance` (higher is better).
    - This is what is shown as `score` in the UI and CLI.

- **Similarity threshold**:
  - Base threshold: `DEFAULT_SIMILARITY_THRESHOLD = 0.25`.
  - Used to filter out very weak matches.

- **Top‑K**:
  - For multi-word queries: show **top 3** similar chunks.
  - For single-word queries: show **up to 6** chunks (exploratory mode).

- **Exploratory behavior for single-word queries**:
  - Single-word queries are often vague; to avoid random-feeling results:
    - We slightly relax the main threshold and increase Top‑K to 6.
    - We enforce an **exploration quality floor** (`EXPLORATION_MIN_SCORE`).
    - If no chunks exceed that floor, we return no results with a clear “no relevant information” message.
  - This is documented in `retriever.py` and surfaced via the UI message (“Exploring around keyword 'X'…”).

### 7. Flask vs FastAPI

- The assignment allows either Flask or FastAPI for the backend.
- This project uses **Flask** because:
  - The UI is a simple HTML form posting to the server.
  - Only a couple of routes are needed (`/` and `/search`).
  - Flask provides a very small, readable code footprint for HTML templates.
- FastAPI would be a good choice for a JSON-first API with automatic documentation, but those strengths are not required for this minimal assignment UI.

### 8. Edge Cases and Error Handling

- **Empty query**:
  - Backend rejects empty input and prompts the user to enter a non-empty query.
- **Very short / single-word queries**:
  - Treated as exploratory; UI warns that results may be broader and less precise.
  - If even the best match is below the exploration quality floor, the system explicitly reports “no relevant information found” instead of returning arbitrary snippets.
- **No matches above threshold**:
  - For multi-word queries: show a clear “No relevant articles found for this query.” message.
  - For single-word queries: behave as described above.
- **OpenAI quota issues**:
  - If the account has insufficient quota, `build_chroma_store.py` will raise an OpenAI `insufficient_quota` error. This is an environment/billing issue, not a logic bug.

### 9. Notes for Reviewers

- To reproduce the entire flow:
  1. Ensure Python dependencies are installed and `OPENAI_API_KEY` is set.
  2. Run `project/data/jsonl_conversion_to_table.py` to regenerate `news_rag_table.csv` (optional if already present).
  3. Run `project/vector_store/build_chroma_store.py` to (re)build the Chroma store.
  4. Start the Flask app with `python3 backend/app.py` from within the `project` directory.
  5. Visit `http://localhost:8000` and try several queries:
     - Single-word: `travel`, `politics`, `sports`.
     - Multi-word: `travel hotels`, `politics guns`, `health vaccines`.

This README, along with the code structure above, is designed to make it easy to understand the reasoning behind each design choice, as emphasized in the assignment instructions in the PDF.


