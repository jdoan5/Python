import re
from pathlib import Path
from typing import List, Dict

import pandas as pd
from llm_client import ask_llm

INDEX_DIR = Path(__file__).resolve().parent.parent / "index"
CHUNKS_PARQUET = INDEX_DIR / "chunks.parquet"

# very small stopword list just to ignore glue words
STOPWORDS = {"the", "a", "an", "and", "or", "to", "of", "for", "in", "on", "at",
             "is", "are", "am", "be", "was", "were", "can", "i", "my", "how"}


def load_index() -> pd.DataFrame:
    if not CHUNKS_PARQUET.exists():
        raise FileNotFoundError(
            f"Index file not found at {CHUNKS_PARQUET}. Run build_index2.py first."
        )
    return pd.read_parquet(CHUNKS_PARQUET)


def search(query: str, k: int = 4) -> List[Dict]:
    """
    Simple word-overlap search:
    - tokenize the question
    - score each chunk by how many query words it contains
    """
    df = load_index()

    # tokenize & filter stopwords
    words = [
        w for w in re.findall(r"\w+", query.lower())
        if w not in STOPWORDS
    ]
    if not words:
        return []

    def score(text: str) -> int:
        t = text.lower()
        return sum(1 for w in words if w in t)

    df["score"] = df["text"].apply(score)

    hits = df[df["score"] > 0].sort_values(
        "score", ascending=False
    ).head(k)

    if hits.empty:
        return []

    # if your index has 'doc_id' instead of 'source', map it here
    if "source" in hits.columns:
        src_col = "source"
    else:
        src_col = "doc_id"

    return hits[[src_col, "chunk_id", "text"]].rename(
        columns={src_col: "source"}
    ).to_dict(orient="records")

def answer_question(question: str) -> str:
    """
    Stage 2 helper:

    1) retrieve top-k chunks for the question
    2) build a context block
    3) ask the LLM using that context
    """
    hits = search(question, k=4)

    if not hits:
        return "I couldn't find anything in the documents for that question yet."

    # Build context for the LLM
    context = "\n\n".join(
        f"Source: {h['source']}\n{h['text']}"
        for h in hits
    )

    prompt = (
        "You are a helpful assistant. Use the context below to answer.\n\n"
        f"{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )

    return ask_llm(prompt)