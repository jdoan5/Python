# src/rag_index.py
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import CFG

LOGGER = logging.getLogger(__name__)


def _json_safe(obj: Any) -> Any:
    """Recursively convert Path objects (and nested structures) into JSON-safe values."""
    if isinstance(obj, Path):
        return obj.as_posix()
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def _ensure_dirs() -> None:
    for attr in ("runs_dir", "data_raw", "data_processed"):
        p = getattr(CFG, attr, None)
        if p is not None:
            Path(p).mkdir(parents=True, exist_ok=True)


def _find_run_dir(run_id: Optional[str]) -> Tuple[str, Path]:
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


def _read_text_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def _load_json_pretty(path: Path) -> str:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return json.dumps(obj, indent=2)
    except Exception:
        return _read_text_file(path)


def _default_inputs() -> List[str]:
    return [
        "report.md",
        "metrics.json",
        "val_metrics.json",
        "test_metrics.json",
        "classification_report_test.md",
    ]


def _chunk_text(text: str, max_chars: int = 1400, overlap: int = 150) -> List[str]:
    """
    Simple, effective chunker for TF-IDF:
    - split by blank lines
    - pack into chunks <= max_chars
    - add small overlap for continuity
    """
    blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
    if not blocks:
        return []

    chunks: List[str] = []
    cur: List[str] = []
    cur_len = 0

    def flush():
        nonlocal cur, cur_len
        if not cur:
            return
        chunk = "\n\n".join(cur).strip()
        if chunk:
            chunks.append(chunk)
        cur = []
        cur_len = 0

    for b in blocks:
        if cur_len + len(b) + 2 > max_chars:
            flush()
            # If the block itself is huge, hard-slice it
            if len(b) > max_chars:
                start = 0
                while start < len(b):
                    end = min(len(b), start + max_chars)
                    chunks.append(b[start:end].strip())
                    start = max(0, end - overlap)
                continue

        cur.append(b)
        cur_len += len(b) + 2

    flush()

    # Apply overlap between chunks (light)
    if overlap > 0 and len(chunks) >= 2:
        out: List[str] = [chunks[0]]
        for i in range(1, len(chunks)):
            prev = out[-1]
            tail = prev[-overlap:] if len(prev) > overlap else prev
            out.append((tail + "\n\n" + chunks[i]).strip())
        return out

    return chunks


def build_index_for_run(
    run_dir: Path,
    input_files: List[str],
    out_filename: str = "rag_index.joblib",
    max_features: int = 8000,
    ngram_max: int = 2,
    chunk_chars: int = 1400,
) -> Dict[str, Path]:
    docs: List[Dict[str, Any]] = []
    missing: List[str] = []

    for name in input_files:
        p = run_dir / name
        if not p.exists():
            missing.append(name)
            continue

        if p.suffix.lower() == ".json":
            text = _load_json_pretty(p)
        else:
            text = _read_text_file(p)

        text = text.strip()
        if not text:
            continue

        chunks = _chunk_text(text, max_chars=chunk_chars)
        for j, ch in enumerate(chunks):
            docs.append(
                {
                    "source": name,
                    "path": p.as_posix(),
                    "chunk_id": j,
                    "text": ch,
                }
            )

    if not docs:
        raise FileNotFoundError(
            f"No readable documents found in {run_dir}. "
            f"Tried: {input_files}. Missing: {missing}"
        )

    texts = [d["text"] for d in docs]

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=max_features,
        ngram_range=(1, ngram_max),
    )
    X = vectorizer.fit_transform(texts)

    payload = {
        "vectorizer": vectorizer,
        "X": X,          # sparse matrix
        "docs": docs,    # KEEP text for snippets + better UX in rag_chat
    }

    out_path = run_dir / out_filename
    joblib.dump(payload, out_path)

    try:
        cfg_snapshot = asdict(CFG)
    except Exception:
        cfg_snapshot = str(CFG)

    meta: Dict[str, Any] = {
        "index_file": out_path,
        "run_dir": run_dir,
        "n_docs_indexed": int(len(docs)),
        "missing_files": missing,
        "tfidf": {
            "max_features": int(max_features),
            "ngram_range": [1, int(ngram_max)],
            "vocab_size": int(len(vectorizer.vocabulary_)),
            "chunk_chars": int(chunk_chars),
        },
        "docs": [{k: v for k, v in d.items() if k != "text"} for d in docs],
        "cfg_snapshot": cfg_snapshot,
    }

    meta_path = run_dir / "rag_index_meta.json"
    meta_path.write_text(json.dumps(_json_safe(meta), indent=2), encoding="utf-8")

    return {"rag_index": out_path, "rag_index_meta": meta_path}


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build TF-IDF RAG index over a run folder (Stage 2).")
    p.add_argument("--run-id", default=None, help="Run id (AAPL_YYYYMMDD_HHMMSS) or folder name (run_AAPL_...). Default: latest.")
    p.add_argument("--out", default="rag_index.joblib", help="Output index filename inside the run directory.")
    p.add_argument("--inputs", nargs="*", default=None, help="List of filenames inside the run directory to index.")
    p.add_argument("--max-features", type=int, default=8000, help="TF-IDF max_features.")
    p.add_argument("--ngram-max", type=int, default=2, help="TF-IDF ngram max (1=unigram, 2=up to bigram).")
    p.add_argument("--chunk-chars", type=int, default=1400, help="Chunk size (characters) per indexed passage.")
    p.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    return p


def main() -> int:
    args = build_arg_parser().parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s | %(message)s",
    )

    rid, run_dir = _find_run_dir(args.run_id)
    inputs = args.inputs if args.inputs else _default_inputs()

    LOGGER.info("Using run_id=%s", rid)
    LOGGER.info("Using run_dir=%s", run_dir.as_posix())
    LOGGER.info("Index inputs=%s", inputs)

    outputs = build_index_for_run(
        run_dir=run_dir,
        input_files=inputs,
        out_filename=args.out,
        max_features=args.max_features,
        ngram_max=args.ngram_max,
        chunk_chars=args.chunk_chars,
    )

    for k, v in outputs.items():
        print(f"{k}={v.as_posix()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())