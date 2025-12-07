from __future__ import annotations

from pathlib import Path
from typing import List, Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# adjust if your paths are different
BASE_DIR = Path(__file__).resolve().parents[1]
INDEX_DIR = BASE_DIR / "index"
CHUNKS_PARQUET = INDEX_DIR / "chunks.parquet"
VECTORIZER_PATH = INDEX_DIR / "vectorizer.joblib"
TFIDF_MATRIX_PATH = INDEX_DIR / "tfidf_matrix.joblib"


def load_index() -> pd.DataFrame:
    """Load chunk metadata (doc_id, chunk_id, text)."""
    if not CHUNKS_PARQUET.exists():
        raise FileNotFoundError(
            f"Index file not found at {CHUNKS_PARQUET}. "
            f"Run build_index3.py first."
        )
    return pd.read_parquet(CHUNKS_PARQUET)


def search(query: str, k: int = 4) -> List[Dict]:
    """
    Stage 3 search:

    * Uses TF-IDF vectors + cosine similarity
    * Returns top-k chunks with a numeric score
    """
    df = load_index()

    # load vectorizer + precomputed matrix
    vectorizer = joblib.load(VECTORIZER_PATH)
    tfidf_matrix = joblib.load(TFIDF_MATRIX_PATH)

    # query vector
    q_vec = vectorizer.transform([query])

    # cosine similarity against all chunks
    sims = cosine_similarity(q_vec, tfidf_matrix)[0]  # shape: (n_chunks,)

    # add scores to dataframe
    df = df.copy()
    df["score"] = sims

    # keep best k chunks
    top = df.sort_values("score", ascending=False).head(k)

    # this is what chat_cli_stage3 expects
    return top[["doc_id", "chunk_id", "text", "score"]].to_dict(orient="records")
