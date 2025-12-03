"""
Minimal Flask backend to expose the RAG retriever via a simple HTML UI.

Why Flask?
- The assignment only needs a very small backend with 1–2 routes.
- Flask is lightweight, easy to read, and integrates naturally with HTML templates.
- FastAPI would also work well, but its strengths (automatic OpenAPI docs, async, etc.)
  are less critical for this small, form-based UI.

This app:
- Serves an HTML page with a text input for queries.
- Calls the RAG `retrieve` function on the backend.
- Displays the retrieved chunks, including similarity scores, category, date, and link.
"""

from __future__ import annotations

from pathlib import Path
import sys

from flask import Flask, render_template, request

# Ensure the project root is on the Python path so we can import rag.retriever
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.retriever import retrieve

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
)


@app.route("/", methods=["GET"])
def index():
    """
    Render the search page with an empty form.
    """
    return render_template(
        "index.html",
        query_text="",
        results=[],
        message="Enter a search query to find relevant news articles.",
    )


@app.route("/search", methods=["POST"])
def search():
    """
    Handle search queries from the HTML form:
    - Read the query string from the form.
    - Call the RAG retriever.
    - Render the same template with results and a status message.
    """
    query_text = (request.form.get("query") or "").strip()

    if not query_text:
        return render_template(
            "index.html",
            query_text="",
            results=[],
            message="Please enter a non-empty query.",
        )

    is_single_word = len(query_text.split()) == 1

    # Call the RAG retriever
    chunks = retrieve(query_text)

    if not chunks:
        # Either nothing met the threshold, or exploration floor filtered everything.
        if is_single_word:
            message = (
                f"No relevant information found around keyword '{query_text}'. "
                "Try a more descriptive query (e.g. 'travel hotels', 'politics guns')."
            )
        else:
            message = "No relevant articles found for this query."

        return render_template(
            "index.html",
            query_text=query_text,
            results=[],
            message=message,
        )

    # Build a friendly message depending on query type
    if is_single_word:
        message = (
            f"Exploring around keyword '{query_text}'. "
            "Results may be broader and less precise."
        )
    else:
        message = "Showing the most relevant articles for your query."

    # Zip results with ranks so the template can show IDs / indices
    ranked_results = list(enumerate(chunks, start=1))

    return render_template(
        "index.html",
        query_text=query_text,
        results=ranked_results,
        message=message,
    )


if __name__ == "__main__":
    # Start the Flask development server.
    # In production you'd use a proper WSGI/ASGI server instead.
    app.run(host="0.0.0.0", port=8000, debug=True)


