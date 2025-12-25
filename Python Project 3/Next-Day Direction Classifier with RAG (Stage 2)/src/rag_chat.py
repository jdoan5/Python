# src/rag_chat.py
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np

from src.config import CFG  # Stage 2 pattern

LOGGER = logging.getLogger(__name__)


def _ensure_dirs() -> None:
    """Create required directories (works even if config.py has no ensure_dirs())."""
    for attr in ("runs_dir", "data_raw", "data_processed"):
        p = getattr(CFG, attr, None)
        if p is not None:
            Path(p).mkdir(parents=True, exist_ok=True)


def _find_run_dir(run_id: Optional[str]) -> Tuple[str, Path]:
    """
    Resolve run_id and run_dir.

    - If run_id is None: pick the most recently modified run_* directory.
    - If run_id is provided:
        * If user passes full folder name 'run_AAPL_...' use as-is
        * If user passes just 'AAPL_...' prefix with 'run_'
    """
    _ensure_dirs()

    runs_root = Path(CFG.runs_dir)

    if run_id:
        rid_in = run_id.strip()
        folder = rid_in if rid_in.startswith("run_") else f"run_{rid_in}"
        run_dir = runs_root / folder
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")
        rid = folder.replace("run_", "", 1)
        return rid, run_dir

    candidates = [p for p in runs_root.glob("run_*") if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No run directories found under: {runs_root}")

    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    rid = latest.name.replace("run_", "", 1)
    return rid, latest


def _load_index(run_dir: Path, index_filename: str) -> Dict[str, Any]:
    idx_path = run_dir / index_filename
    if not idx_path.exists():
        raise FileNotFoundError(
            f"RAG index not found: {idx_path}\n"
            f"Run: python -m src.rag_index --run-id {run_dir.name.replace('run_', '')}"
        )
    payload = joblib.load(idx_path)
    # expected keys: vectorizer, X (sparse), docs (list of metadata dicts)
    for k in ("vectorizer", "X", "docs"):
        if k not in payload:
            raise ValueError(f"Invalid index payload: missing key '{k}' in {idx_path}")
    return payload


def _search(payload: Dict[str, Any], query: str, top_k: int) -> List[Tuple[float, Dict[str, Any]]]:
    """
    TF-IDF cosine similarity search: score = (q · doc) / (||q|| ||doc||)
    With normalized TF-IDF vectors, dot product approximates cosine similarity.
    """
    vectorizer = payload["vectorizer"]
    X = payload["X"]
    docs = payload["docs"]

    q = vectorizer.transform([query])
    scores = (X @ q.T).toarray().ravel()  # shape: (n_docs,)

    if scores.size == 0:
        return []

    top_idx = np.argsort(scores)[::-1][:top_k]
    results: List[Tuple[float, Dict[str, Any]]] = []
    for i in top_idx:
        results.append((float(scores[i]), docs[int(i)]))
    return results


def _format_results(results: List[Tuple[float, Dict[str, Any]]]) -> str:
    if not results:
        return "No matches.\n"

    lines: List[str] = []
    for rank, (score, meta) in enumerate(results, start=1):
        src = meta.get("source", "unknown")
        path = meta.get("path", "")
        lines.append(f"{rank}. score={score:.4f} | {src}")
        if path:
            lines.append(f"   {path}")
    return "\n".join(lines) + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Local TF-IDF RAG chat over run artifacts (Stage 2).")
    p.add_argument("--run-id", default=None, help="Run id (AAPL_YYYYMMDD_HHMMSS) or folder name (run_AAPL_...). Default: latest.")
    p.add_argument("--index", default="rag_index.joblib", help="Index filename inside the run directory.")
    p.add_argument("--top-k", type=int, default=3, help="How many documents to retrieve per query.")
    p.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    return p


def main() -> int:
    args = build_arg_parser().parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s | %(message)s",
    )

    rid, run_dir = _find_run_dir(args.run_id)
    payload = _load_index(run_dir, args.index)

    LOGGER.info("Using run_id=%s", rid)
    LOGGER.info("Using run_dir=%s", run_dir.as_posix())
    LOGGER.info("Docs indexed=%s", len(payload["docs"]))

    print("\nRAG Run Assistant (local TF-IDF)")
    print(f"Run: {rid}")
    print("Type a question and press Enter. Type 'exit' to quit.\n")

    while True:
        try:
            q = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not q:
            continue
        if q.lower() in {"exit", "quit", "q"}:
            break

        results = _search(payload, q, top_k=args.top_k)
        print("\nTop matches:")
        print(_format_results(results))

        # Optional hint: show what files exist
        if not results:
            available = [d.get("source", "") for d in payload["docs"]]
            print("Indexed sources:", ", ".join([a for a in available if a]))
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())