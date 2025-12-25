# src/rag_chat.py
from __future__ import annotations

import argparse
import json
import logging
import re
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


def _safe_read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _safe_read_text(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None


def _load_index(run_dir: Path, index_filename: str) -> Dict[str, Any]:
    idx_path = run_dir / index_filename
    if not idx_path.exists():
        raise FileNotFoundError(
            f"RAG index not found: {idx_path}\n"
            f"Run: python -m src.rag_index --run-id {run_dir.name.replace('run_', '')}"
        )
    payload = joblib.load(idx_path)
    for k in ("vectorizer", "X", "docs"):
        if k not in payload:
            raise ValueError(f"Invalid index payload: missing key '{k}' in {idx_path}")
    return payload


def _search(payload: Dict[str, Any], query: str, top_k: int) -> List[Tuple[float, Dict[str, Any]]]:
    """
    TF-IDF cosine similarity search: score = q · doc
    (If TF-IDF vectors are normalized, dot product approximates cosine similarity.)
    """
    vectorizer = payload["vectorizer"]
    X = payload["X"]
    docs = payload["docs"]

    q = vectorizer.transform([query])
    scores = (X @ q.T).toarray().ravel()

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


def _pretty_cm(cm: Any) -> str:
    try:
        m = np.array(cm, dtype=int)
        return f"[[{m[0,0]}, {m[0,1]}], [{m[1,0]}, {m[1,1]}]]"
    except Exception:
        return str(cm)


def _quick_answer(query: str, rid: str, run_dir: Path) -> Optional[str]:
    """
    Direct answers from run artifacts (preferred over TF-IDF when possible).
    """
    q = query.strip().lower()

    metrics = _safe_read_json(run_dir / "metrics.json") or {}
    valm = _safe_read_json(run_dir / "val_metrics.json") or {}
    testm = _safe_read_json(run_dir / "test_metrics.json") or {}

    symbol = str(metrics.get("symbol") or rid.split("_", 1)[0])

    def pick(d: Dict[str, Any], key: str) -> Optional[float]:
        try:
            v = d.get(key)
            return float(v) if v is not None else None
        except Exception:
            return None

    # Test/Val accuracy
    if ("test" in q and "acc" in q) or q in {"test accuracy", "accuracy test"}:
        v = pick(testm, "accuracy")
        if v is not None:
            return f"TEST accuracy for {symbol} ({rid}) = {v:.4f}"
        return "TEST accuracy not found. (Run `python -m src.evaluate --run-id <RID>` first.)"

    if ("val" in q and "acc" in q) or q in {"val accuracy", "validation accuracy"}:
        v = pick(valm, "accuracy")
        if v is not None:
            return f"VAL accuracy for {symbol} ({rid}) = {v:.4f}"
        return "VAL accuracy not found. (Run `python -m src.evaluate --run-id <RID>` first.)"

    # Macro F1 / Weighted F1
    if "macro" in q and ("f1" in q or "f-1" in q):
        v_val = pick(valm, "macro_f1")
        v_test = pick(testm, "macro_f1")
        parts = []
        if v_val is not None:
            parts.append(f"VAL macro-F1={v_val:.4f}")
        if v_test is not None:
            parts.append(f"TEST macro-F1={v_test:.4f}")
        if parts:
            return f"{symbol} ({rid}) — " + " | ".join(parts)
        return "macro-F1 not found in val/test metrics yet. Run evaluation."

    if "weighted" in q and ("f1" in q or "f-1" in q):
        v_val = pick(valm, "weighted_f1")
        v_test = pick(testm, "weighted_f1")
        parts = []
        if v_val is not None:
            parts.append(f"VAL weighted-F1={v_val:.4f}")
        if v_test is not None:
            parts.append(f"TEST weighted-F1={v_test:.4f}")
        if parts:
            return f"{symbol} ({rid}) — " + " | ".join(parts)
        return "weighted-F1 not found in val/test metrics yet. Run evaluation."

    # Confusion matrix
    if "confusion" in q or "cm" in q:
        cm = None
        try:
            cm = (testm.get("confusion_matrix") or {}).get("matrix")
        except Exception:
            cm = None
        if cm is not None:
            return f"TEST confusion matrix for {symbol} ({rid}) = {_pretty_cm(cm)}"
        return "Confusion matrix not found in test_metrics.json yet. Run evaluation."

    # Best model
    if ("best" in q and "model" in q) or q in {"model", "best"}:
        bm = metrics.get("best_model")
        if bm:
            return f"Best model for {symbol} ({rid}) = {bm}"
        return "best_model not found in metrics.json."

    # Features
    if "feature" in q or "features" in q:
        feats = metrics.get("features") or getattr(CFG, "feature_cols", None) or []
        if isinstance(feats, list) and feats:
            return f"Features used for {symbol} ({rid}) = {', '.join(map(str, feats))}"
        return "Feature list not found in metrics.json or CFG."

    # Rows / dataset size
    if "rows" in q or "n_rows" in q or "samples" in q:
        n = metrics.get("n_rows_after_features")
        if n is not None:
            return f"Rows after features for {symbol} ({rid}) = {int(n)}"
        return "Row count not found in metrics.json."

    # Date range (best-effort)
    if "date" in q or "range" in q:
        s = metrics.get("start_date") or getattr(CFG, "start_date", None)
        e = metrics.get("end_date") or getattr(CFG, "end_date", None)
        if s or e:
            return f"Date range for {symbol} ({rid}) = {s} → {e}"
        return "Date range not found."

    return None


def _snippet_from_meta(meta: Dict[str, Any], query: str, max_chars: int = 600) -> Optional[str]:
    """
    Show a short snippet either from meta['text'] (if stored) or by reading meta['path'].
    """
    # If rag_index stored content
    t = meta.get("text")
    if isinstance(t, str) and t.strip():
        return (t.strip()[:max_chars] + ("…" if len(t) > max_chars else ""))

    # Otherwise read file path (if any)
    p = meta.get("path")
    if not p:
        return None

    path = Path(str(p))
    txt = _safe_read_text(path)
    if not txt:
        return None

    # Best-effort: find first line containing a query token
    tokens = [w for w in re.split(r"\W+", query.lower()) if len(w) >= 3]
    lines = txt.splitlines()
    hit_idx = None
    for i, line in enumerate(lines):
        low = line.lower()
        if any(tok in low for tok in tokens):
            hit_idx = i
            break

    if hit_idx is None:
        # fallback: head snippet
        snippet = "\n".join(lines[:20]).strip()
        return snippet[:max_chars] + ("…" if len(snippet) > max_chars else "")

    start = max(0, hit_idx - 6)
    end = min(len(lines), hit_idx + 8)
    snippet = "\n".join(lines[start:end]).strip()
    return snippet[:max_chars] + ("…" if len(snippet) > max_chars else "")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Local TF-IDF RAG chat over run artifacts (Stage 2).")
    p.add_argument("--run-id", default=None, help="Run id (AAPL_YYYYMMDD_HHMMSS) or folder name (run_AAPL_...). Default: latest.")
    p.add_argument("--index", default="rag_index.joblib", help="Index filename inside the run directory.")
    p.add_argument("--top-k", type=int, default=3, help="How many documents to retrieve per query.")
    p.add_argument("--min-score", type=float, default=0.05, help="If best similarity < min-score, treat as 'no strong match'.")
    p.add_argument("--show-snippets", action="store_true", help="Print short snippets from retrieved documents.")
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

    print("\nRAG Run Assistant (local TF-IDF + direct answers)")
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

        # 1) Direct answer from JSON artifacts when possible
        ans = _quick_answer(q, rid, run_dir)
        if ans:
            print(ans)

        # 2) TF-IDF retrieval (still useful for “show me where this is in the report”)
        results = _search(payload, q, top_k=args.top_k)
        best = results[0][0] if results else 0.0

        if not results or best < args.min_score:
            if not ans:
                print(f"No strong TF-IDF match for that query (best similarity ~ {best:.3f}).")
            print()
            continue

        print("\nTop matches:")
        print(_format_results(results))

        if args.show_snippets:
            for rank, (score, meta) in enumerate(results, start=1):
                snip = _snippet_from_meta(meta, q)
                if snip:
                    print(f"[Snippet {rank} | score={score:.4f} | {meta.get('source','unknown')}]")
                    print(snip)
                    print()

        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())