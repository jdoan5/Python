import os
import pathlib
from typing import List, Dict

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib  # comes with sklearn, usually; if not: pip install joblib


# ----- Paths ------------------------------------------------------------
BASE_DIR = pathlib.Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
SOURCE_DIR = DATA_DIR / "source"
INDEX_DIR = BASE_DIR / "index"

INDEX_DIR.mkdir(exist_ok=True)


def load_documents() -> List[Dict]:
    """
    Walk through data/source and load all .txt files.
    Each file can be split into chunks by blank lines.
    """
    docs = []
    for path in SOURCE_DIR.glob("*.txt"):
        text = path.read_text(encoding="utf-8")
        chunks = [c.strip() for c in text.split("\n\n") if c.strip()]
        for i, chunk in enumerate(chunks):
            docs.append(
                {
                    "doc_id": path.name,
                    "chunk_id": i,
                    "text": chunk,
                }
            )
    return docs


def build_index():
    docs = load_documents()
    if not docs:
        raise SystemExit(f"No .txt files found in {SOURCE_DIR}")

    df = pd.DataFrame(docs)
    print(f"Loaded {len(df)} text chunks from {SOURCE_DIR}")

    # TF-IDF over the chunk text
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        stop_words="english",
    )
    tfidf_matrix = vectorizer.fit_transform(df["text"].values)

    # Save artifacts
    joblib.dump(vectorizer, INDEX_DIR / "vectorizer.joblib")
    joblib.dump(tfidf_matrix, INDEX_DIR / "tfidf_matrix.joblib")
    df.to_parquet(INDEX_DIR / "chunks.parquet", index=False)

    print(f"Index saved to {INDEX_DIR}")
    print("Columns in chunks.parquet:", list(df.columns))


if __name__ == "__main__":
    build_index()