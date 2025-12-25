# make_latest.py
from __future__ import annotations

import argparse
import shutil
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Optional, List, Tuple

# Prefer Stage-2 CFG pattern if present
try:
    from src.config import CFG  # type: ignore
except Exception:
    CFG = None  # fallback


DEFAULT_REPORT_FILES = [
    "report.md",
    "metrics.json",
    "val_metrics.json",
    "test_metrics.json",
    "confusion_matrix_test.png",
    "classification_report_test.md",
]

DEFAULT_RAG_FILES = [
    "rag_index.joblib",
    "rag_index_meta.json",
]


def _project_root() -> Path:
    if CFG is not None and hasattr(CFG, "project_dir"):
        return Path(getattr(CFG, "project_dir"))
    return Path.cwd()


def _runs_dir(root: Path) -> Path:
    if CFG is not None and hasattr(CFG, "runs_dir"):
        return Path(getattr(CFG, "runs_dir"))
    return root / "runs"


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _latest_run_dir(runs_dir: Path) -> Path:
    candidates = [p for p in runs_dir.glob("run_*") if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No run directories found under: {runs_dir}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _copy_or_link(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists() or dst.is_symlink():
        dst.unlink()

    if mode == "symlink":
        # Make relative symlinks so the folder is portable
        rel = src.relative_to(dst.parent) if src.is_relative_to(dst.parent) else src
        dst.symlink_to(rel)
    else:
        shutil.copy2(src, dst)


def _export_files(
    run_dir: Path,
    out_dir: Path,
    files: List[str],
    prefix: str,
    mode: str,
) -> List[Tuple[str, Optional[Path]]]:
    results: List[Tuple[str, Optional[Path]]] = []
    for name in files:
        src = run_dir / name
        dst = out_dir / f"{prefix}{name}"
        if not src.exists():
            results.append((name, None))
            continue
        _copy_or_link(src, dst, mode)
        results.append((name, dst))
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Export newest run artifacts into reports/ and rag/ as 'latest_*'.")
    parser.add_argument("--mode", choices=["copy", "symlink"], default="copy", help="How to export files (default: copy).")
    args = parser.parse_args()

    root = _project_root()
    runs = _runs_dir(root)
    reports_dir = root / "reports"
    rag_dir = root / "rag"

    _safe_mkdir(reports_dir)
    _safe_mkdir(rag_dir)

    run_dir = _latest_run_dir(runs)
    run_name = run_dir.name  # e.g. run_AAPL_20251224_164008
    prefix = "latest_"

    # Export key artifacts
    rep = _export_files(run_dir, reports_dir, DEFAULT_REPORT_FILES, prefix=prefix, mode=args.mode)
    rag = _export_files(run_dir, rag_dir, DEFAULT_RAG_FILES, prefix=prefix, mode=args.mode)

    # Write latest run pointer
    (reports_dir / "latest_run.txt").write_text(f"{run_name}\n", encoding="utf-8")

    # Optional: dump CFG snapshot for sanity/debug
    if CFG is not None:
        try:
            snap = asdict(CFG) if is_dataclass(CFG) else str(CFG)
            (reports_dir / "latest_cfg.txt").write_text(f"{snap}\n", encoding="utf-8")
        except Exception:
            pass

    print(f"Latest run: {run_name}")
    print(f"Export mode: {args.mode}")
    print("\nReports exported to:")
    for name, dst in rep:
        print(f"  - {name}: {dst.as_posix() if dst else 'MISSING'}")

    print("\nRAG exported to:")
    for name, dst in rag:
        print(f"  - {name}: {dst.as_posix() if dst else 'MISSING'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())