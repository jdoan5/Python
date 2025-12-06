import pathlib
from typing import List

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = pathlib.Path(__file__).resolve().parents[1]
INDEX_DIR = BASE_DIR / "index"

CHUNKS_PATH = INDEX_DIR / "chunks.parquet"
VECTORIZER_PATH = INDEX_DIR / "vectorizer.joblib"
TFIDF_PATH = INDEX_DIR / "tfidf_matrix.joblib"


def load_index():
    if not CHUNKS_PATH.exists():
        raise SystemExit("Index not found. Run build_index1.py first.")

    chunks = pd.read_parquet(CHUNKS_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    tfidf_matrix = joblib.load(TFIDF_PATH)
    return chunks, vectorizer, tfidf_matrix


def retrieve(query: str, top_k: int = 3):
    chunks, vectorizer, tfidf_matrix = load_index()

    query_vec = vectorizer.transform([query])
    sims = cosine_similarity(query_vec, tfidf_matrix)[0]  # 1D array
    top_idx = np.argsort(sims)[::-1][:top_k]

    results = []
    for idx in top_idx:
        row = chunks.iloc[idx]
        results.append(
            {
                "score": float(sims[idx]),
                "doc_id": row["doc_id"],
                "chunk_id": int(row["chunk_id"]),
                "text": row["text"],
            }
        )
    return results


def format_answer(query: str, results: List[dict]) -> str:
    if not results:
        return "I couldn't find anything relevant for that question."

    parts = [f"Q: {query}", "", "Top matching passages:"]
    for r in results:
        parts.append(
            f"- [score={r['score']:.3f}] {r['doc_id']} (chunk {r['chunk_id']}):\n"
            f"  {r['text']}"
        )

    # Simple template “answer” – later we can plug in an LLM
    parts.append("")
    parts.append("Draft answer (based on the passages above):")
    parts.append("→ " + " ".join(r["text"] for r in results))

    return "\n".join(parts)


def main():
    print("=== RAG Mini Bot (Stage 1: Retriever only) ===")
    print("Type your question, or 'exit' to quit.\n")

    while True:
        q = input("You: ").strip()
        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            print("Bye!")
            break

        results = retrieve(q, top_k=3)
        answer = format_answer(q, results)
        print()
        print(answer)
        print("\n" + "-" * 80 + "\n")


if __name__ == "__main__":
    main()