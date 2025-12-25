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

from src.config import CFG  # Stage 2 pattern (train.py uses CFG)

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
    """
    Files we typically want in the TF-IDF index (per run folder).
    Add/remove as you like.
    """
    return [
        "report.md",
        "metrics.json",
        "val_metrics.json",
        "test_metrics.json",
        "classification_report_test.md",
    ]


def build_index_for_run(
    run_dir: Path,
    input_files: List[str],
    out_filename: str = "rag_index.joblib",
    max_features: int = 4000,
    ngram_max: int = 2,
) -> Dict[str, Path]:
    """
    Build a lightweight TF-IDF index over selected run artifacts.

    Saves:
      - <run_dir>/<out_filename> (joblib)
      - <run_dir>/rag_index_meta.json
    """
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

        if text.strip():
            docs.append(
                {
                    "source": name,
                    "path": p.as_posix(),
                    "text": text,
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
        "X": X,  # sparse matrix
        "docs": [
            {k: v for k, v in d.items() if k != "text"}  # keep metadata only
            for d in docs
        ],
    }

    out_path = run_dir / out_filename
    joblib.dump(payload, out_path)

    # Meta (JSON-safe)
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
        },
        "docs": [
            {k: v for k, v in d.items() if k != "text"}
            for d in docs
        ],
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
    p.add_argument("--max-features", type=int, default=4000, help="TF-IDF max_features.")
    p.add_argument("--ngram-max", type=int, default=2, help="TF-IDF ngram max (1=unigram, 2=up to bigram).")
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
    )

    for k, v in outputs.items():
        print(f"{k}={v.as_posix()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())