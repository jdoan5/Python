from __future__ import annotations

from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import CFG


def build_index():
    CFG.runs_dir.mkdir(parents=True, exist_ok=True)

    docs: list[str] = []
    paths: list[str] = []

    for p in CFG.runs_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".md", ".json"}:
            try:
                text = p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            docs.append(text)
            paths.append(str(p))

    if not docs:
        raise RuntimeError("No run docs found. Train at least one run first.")

    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    X = vectorizer.fit_transform(docs)

    return {
        "paths": paths,
        "vectorizer": vectorizer,
        "matrix": X,
    }


def main():
    import joblib

    idx = build_index()
    out = CFG.runs_dir / "rag_index.joblib"
    out.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(idx, out)
    print(f"Saved RAG index to: {out}")


if __name__ == "__main__":
    main()