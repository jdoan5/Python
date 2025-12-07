# retrieval.py to use TF-IDF search
# retrieval.py (Stage 2 – local TF-IDF + simple answer)
from __future__ import annotations

from pathlib import Path
from typing import List, Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# ---------- Paths ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
INDEX_DIR = BASE_DIR / "index"
CHUNKS_PARQUET = INDEX_DIR / "chunks.parquet"
VECTORIZER_PATH = INDEX_DIR / "vectorizer.joblib"
TFIDF_MATRIX_PATH = INDEX_DIR / "tfidf_matrix.joblib"


# ---------- Load index & TF-IDF -------------------------------------------
def load_index() -> pd.DataFrame:
    """Load chunk metadata (doc_id, chunk_id, text)."""
    if not CHUNKS_PARQUET.exists():
        raise FileNotFoundError(
            f"Index file not found at {CHUNKS_PARQUET}. Run build_index2.py first."
        )
    return pd.read_parquet(CHUNKS_PARQUET)


def load_tfidf():
    """Load vectorizer + TF-IDF matrix."""
    if not VECTORIZER_PATH.exists() or not TFIDF_MATRIX_PATH.exists():
        raise FileNotFoundError(
            f"TF-IDF artifacts not found in {INDEX_DIR}. Run build_index2.py first."
        )
    vectorizer = joblib.load(VECTORIZER_PATH)
    tfidf_matrix = joblib.load(TFIDF_MATRIX_PATH)
    return vectorizer, tfidf_matrix


# ---------- Search --------------------------------------------------------
def search(query: str, k: int = 4) -> List[Dict]:
    """
    TF-IDF cosine similarity search over chunks.

    Returns up to k best-matching chunks as dictionaries:
    { 'doc_id': ..., 'chunk_id': ..., 'text': ..., 'score': ... }
    """
    df = load_index()
    vectorizer, tfidf_matrix = load_tfidf()

    # Vector for the query
    q_vec = vectorizer.transform([query])
    scores = cosine_similarity(q_vec, tfidf_matrix)[0]

    # If everything is zero, treat as "no hits"
    if np.max(scores) <= 0:
        return []

    top_idx = scores.argsort()[::-1][:k]
    results = df.iloc[top_idx].copy()
    results["score"] = scores[top_idx]

    return results[["doc_id", "chunk_id", "text", "score"]].to_dict(orient="records")


# ---------- Simple “answer” helper ---------------------------------------
def answer_question(question: str) -> str:
    """
    Local-only Stage 2:
      1. Retrieve top-k chunks with TF-IDF.
      2. Return the best chunk text as the “answer”.

    No OpenAI / network used here.
    """
    hits = search(question, k=4)

    if not hits:
        return "I couldn't find anything in the documents for that question yet."

    # Best hit is first (highest score)
    best = hits[0]

    # You can tweak this format however you like
    return f"Source: {best['doc_id']}\n\n{best['text']}"