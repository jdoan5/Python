# src/evaluate.py
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional, Any, List

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from src.config import CFG

LOGGER = logging.getLogger(__name__)


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
        rdir = runs_root / folder
        if not rdir.exists():
            raise FileNotFoundError(f"Run directory not found: {rdir}")
        rid = folder.replace("run_", "", 1)
        return rid, rdir

    candidates = [p for p in runs_root.glob("run_*") if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No run directories found under: {runs_root}")

    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    rid = latest.name.replace("run_", "", 1)
    return rid, latest


def _load_run_meta(run_dir: Path) -> Dict[str, Any]:
    meta_path = run_dir / "metrics.json"
    if not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _default_processed_path(run_dir: Path, rid: str) -> Path:
    meta = _load_run_meta(run_dir)
    processed_name = meta.get("processed_data_path")

    processed_dir = Path(CFG.data_processed)

    if processed_name:
        p = processed_dir / processed_name
        return p

    # fallback: newest matching features csv
    symbol = rid.split("_", 1)[0].strip()
    matches = sorted(
        processed_dir.glob(f"{symbol}_*_features.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if matches:
        return matches[0]

    return processed_dir / f"{rid}_processed.csv"


def load_processed_df(processed_csv: Path) -> pd.DataFrame:
    if not processed_csv.exists():
        raise FileNotFoundError(f"Processed CSV not found: {processed_csv}")

    df = pd.read_csv(processed_csv)

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.sort_values("Date")

    return df


def time_split_indices(n: int, train_frac: float, val_frac: float) -> Dict[str, np.ndarray]:
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    n_test = n - n_train - n_val

    if n_train <= 0 or n_val <= 0 or n_test <= 0:
        raise ValueError(f"Split too small: n={n} -> train={n_train}, val={n_val}, test={n_test}")

    idx = np.arange(n)
    return {
        "train": idx[:n_train],
        "val": idx[n_train : n_train + n_val],
        "test": idx[n_train + n_val :],
    }


def _metrics_dict(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": {"labels": ["0", "1"], "matrix": cm.tolist()},
        "classification_report": report,
        "support": int(len(y_true)),
    }


def save_json(obj: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def save_confusion_matrix_png(cm: np.ndarray, out_path: Path, title: str = "Confusion Matrix (TEST)") -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(cm)

    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["0", "1"])
    ax.set_yticklabels(["0", "1"])

    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_classification_report_md(report_dict: Dict[str, Any], out_path: Path, title: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: List[str] = []
    for k in report_dict.keys():
        if k in ("accuracy", "macro avg", "weighted avg") or str(k).isdigit():
            rows.append(k)

    lines: List[str] = []
    lines.append(f"# {title}\n")
    lines.append("| class | precision | recall | f1-score | support |")
    lines.append("|---|---:|---:|---:|---:|")

    for k in rows:
        if k == "accuracy":
            acc = float(report_dict["accuracy"])
            lines.append(f"| accuracy |  |  | **{acc:.4f}** |  |")
            continue
        d = report_dict[k]
        lines.append(
            f"| {k} | {float(d.get('precision', 0)):.4f} | {float(d.get('recall', 0)):.4f} | "
            f"{float(d.get('f1-score', 0)):.4f} | {int(d.get('support', 0))} |"
        )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate_run(
    run_id: Optional[str] = None,
    processed_csv: Optional[str] = None,
    model_filename: str = "model.joblib",
) -> Dict[str, Path]:
    rid, run_dir = _find_run_dir(run_id)
    meta = _load_run_meta(run_dir)

    processed_path = Path(processed_csv) if processed_csv else _default_processed_path(run_dir, rid)

    df = load_processed_df(processed_path)

    # Prefer the features used during training (stored in metrics.json)
    feats = meta.get("features") or getattr(CFG, "feature_cols", None) or getattr(CFG, "FEATURE_COLS", None) or []
    feats = list(feats)

    target = meta.get("target_col") or getattr(CFG, "target_col", None) or getattr(CFG, "TARGET_COL", None) or "target_up"

    if not feats:
        raise ValueError("No feature columns available. Ensure train wrote 'features' into metrics.json.")

    missing_features = [c for c in feats if c not in df.columns]
    if missing_features:
        raise ValueError(
            f"Missing feature columns in processed CSV: {missing_features}\n"
            f"Loaded file: {processed_path}\n"
            f"Columns found: {list(df.columns)}"
        )

    if target not in df.columns:
        raise ValueError(f"Missing target column '{target}' in processed CSV.")

    df_eval = df.dropna(subset=feats + [target]).copy()
    if df_eval.empty:
        raise ValueError("No rows left after dropping NA in features/target.")

    X = df_eval[feats].to_numpy()
    y = df_eval[target].astype(int).to_numpy()

    train_ratio = float(meta.get("split", {}).get("train", 0))  # not used
    # Use CFG ratios (same as train.py time_split)
    splits = time_split_indices(len(df_eval), float(CFG.train_ratio), float(CFG.val_ratio))

    model_path = run_dir / model_filename
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")

    model = joblib.load(model_path)

    y_val_pred = model.predict(X[splits["val"]])
    y_test_pred = model.predict(X[splits["test"]])

    val_metrics = _metrics_dict(y[splits["val"]], y_val_pred)
    test_metrics = _metrics_dict(y[splits["test"]], y_test_pred)

    run_context = {
        "run_id": rid,
        "processed_csv": processed_path.as_posix(),
        "model_path": model_path.as_posix(),
        "features_used": feats,
        "target_col": target,
        "split": {
            "train": int(len(splits["train"])),
            "val": int(len(splits["val"])),
            "test": int(len(splits["test"])),
        },
    }
    val_metrics.update(run_context)
    test_metrics.update(run_context)

    out_val_json = run_dir / "val_metrics.json"
    out_test_json = run_dir / "test_metrics.json"
    save_json(val_metrics, out_val_json)
    save_json(test_metrics, out_test_json)

    cm_test = np.array(test_metrics["confusion_matrix"]["matrix"], dtype=int)
    out_cm_png = run_dir / "confusion_matrix_test.png"
    save_confusion_matrix_png(cm_test, out_cm_png, title="Confusion Matrix (TEST)")

    out_report_md = run_dir / "classification_report_test.md"
    save_classification_report_md(
        test_metrics["classification_report"],
        out_report_md,
        title=f"Classification Report (TEST) — {rid}",
    )

    return {
        "val_metrics_json": out_val_json,
        "test_metrics_json": out_test_json,
        "confusion_matrix_png": out_cm_png,
        "classification_report_md": out_report_md,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate a saved run (Stage 2).")
    p.add_argument("--run-id", default=None, help="Run id (e.g., AAPL_20251224_163023) or folder name (run_AAPL_...).")
    p.add_argument("--processed-csv", default=None, help="Path to processed CSV. If omitted, uses metrics.json mapping.")
    p.add_argument("--model-filename", default="model.joblib", help="Model filename inside the run directory.")
    p.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s | %(message)s",
    )

    LOGGER.info("CFG: %s", CFG)

    outputs = evaluate_run(
        run_id=args.run_id,
        processed_csv=args.processed_csv,
        model_filename=args.model_filename,
    )

    for k, v in outputs.items():
        print(f"{k}={v.as_posix()}")


if __name__ == "__main__":
    main()