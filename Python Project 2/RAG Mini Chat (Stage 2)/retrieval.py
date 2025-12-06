# retrieval.py
from __future__ import annotations  # must be first (after comments/docstring)

from pathlib import Path
from typing import List, Dict

import pandas as pd

from llm_client import ask_llm


# Path to your chunks parquet created by build_index1.py
INDEX_DIR = Path(__file__).resolve().parent / "index"
CHUNKS_PARQUET = INDEX_DIR / "chunks.parquet"


def load_index() -> pd.DataFrame:
    """Load the chunk index from disk."""
    if not CHUNKS_PARQUET.exists():
        raise FileNotFoundError(
            f"Index file not found at {CHUNKS_PARQUET}. "
            f"Run build_index.py first."
        )
    return pd.read_parquet(CHUNKS_PARQUET)


def search(query: str, k: int = 4) -> List[Dict]:
    """
    Very simple text search over chunks.

    For Stage 2 we keep it basic:
      - case-insensitive substring match on the chunk text
      - return up to k best matches
    """
    df = load_index()

    # crude relevance: text contains the query (case-insensitive)
    if "text" not in df.columns:
        raise KeyError("Expected a 'text' column in chunks.parquet, but did not find one.")

    mask = df["text"].str.contains(query, case=False, na=False)
    results = df[mask].copy()

    if results.empty:
        return []

    # optional: sort shorter chunks first
    results["len"] = results["text"].str.len()
    results = results.sort_values("len").head(k)

    # Build records in a defensive way, so we don't require 'source' or 'chunk_id'
    hits: List[Dict] = []
    for idx, row in results.iterrows():
        hits.append(
            {
                # if there's a 'source' column, use it; otherwise use a simple default label
                "source": row.get("source", "index/chunks.parquet"),
                # fall back to the dataframe index if there is no explicit chunk_id column
                "chunk_id": int(row.get("chunk_id", idx)),
                "text": row["text"],
            }
        )

    return hits


def answer_question(question: str) -> str:
    """
    Stage 2 helper:
    1) retrieve top-k chunks for the question
    2) build a context block
    3) ask the LLM using that context
    """

    # 1) retrieve relevant chunks
    hits = search(question, k=4)

    if not hits:
        return "I couldn't find anything in the documents for that question yet."

    # 2) build context for the LLM
    context = "\n\n".join(
        f"Source: {h['source']}\n{h['text']}"
        for h in hits
    )

    # 3) ask the LLM
    prompt = (
        "You are a helpful assistant. Use the context below to answer.\n\n"
        f"{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )

    return ask_llm(prompt)