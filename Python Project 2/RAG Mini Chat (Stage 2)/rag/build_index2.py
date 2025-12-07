# build_index2.py
from __future__ import annotations

from pathlib import Path
from typing import List, Dict

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib  # if missing: pip install joblib

# ----- Paths ------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
SOURCE_DIR = DATA_DIR / "source"
INDEX_DIR = PROJECT_ROOT / "index"

# Make sure index/ exists
INDEX_DIR.mkdir(parents=True, exist_ok=True)


def load_documents() -> List[Dict]:
    """
    Walk through data/source and load all .txt files.
    Each file is split into chunks by blank lines.
    """
    docs: List[Dict] = []

    txt_files = sorted(SOURCE_DIR.glob("*.txt"))
    print(f"Found {len(txt_files)} source files in {SOURCE_DIR}:")
    for path in txt_files:
        print("  -", path.name)
        text = path.read_text(encoding="utf-8")

        # Split on double newlines; drop empty chunks
        chunks = [c.strip() for c in text.split("\n\n") if c.strip()]

        for i, chunk in enumerate(chunks):
            docs.append(
                {
                    "source": path.name,  # <-- filename, used by retrieval.py
                    "chunk_id": i,
                    "text": chunk,
                }
            )

    return docs


def build_index() -> None:
    docs = load_documents()
    if not docs:
        raise SystemExit(f"No .txt files found in {SOURCE_DIR}")

    df = pd.DataFrame(docs)
    print(f"\nLoaded {len(df)} text chunks from {SOURCE_DIR}")

    # TF-IDF over the chunk text
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        stop_words="english",
    )
    tfidf_matrix = vectorizer.fit_transform(df["text"].values)

    # Save artifacts into index/
    vectorizer_path = INDEX_DIR / "vectorizer.joblib"
    matrix_path = INDEX_DIR / "tfidf_matrix.joblib"
    chunks_path = INDEX_DIR / "chunks.parquet"

    joblib.dump(vectorizer, vectorizer_path)
    joblib.dump(tfidf_matrix, matrix_path)
    df.to_parquet(chunks_path, index=False)

    print(f"\nIndex saved to {INDEX_DIR}")
    print("  -", vectorizer_path.name)
    print("  -", matrix_path.name)
    print("  -", chunks_path.name)
    print("Columns in chunks.parquet:", list(df.columns))


if __name__ == "__main__":
    build_index()