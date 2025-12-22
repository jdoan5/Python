#!/usr/bin/env python3
"""\
Predict / batch-score with a trained Fake News model (Stage 3).

Stage 3 notes (vs Stage 2):
- The saved model may be probability-calibrated (e.g., LinearSVC wrapped with CalibratedClassifierCV),
  which means predict_proba() is often available.
- Training may also choose a tuned classification threshold (stored in reports/metrics.json as
  chosen_threshold). This script can auto-load and use that threshold when you do not specify one.

Typical batch scoring:
  python src/predict.py \
    --model artifacts/best_model.joblib \
    --metrics reports/metrics.json \
    --data data/fake_and_real_news.csv \
    --text-col Text \
    --label-col label \
    --out reports/preds_scored.csv

One-off text scoring:
  python src/predict.py --model artifacts/best_model.joblib --metrics reports/metrics.json --text "Breaking: ..."

Outputs (batch mode):
- y_pred: numeric label (0/1 best-effort)
- pred_label: display label ("Real"/"Fake" by default)
- y_proba_fake: probability of Fake when available (preferred)
- score_fake: raw decision score when probabilities are not available
- threshold_used: threshold used when probabilities are available
- If --label-col is provided, also includes y_true + is_correct.

If you see scikit-learn warnings or runtime errors after loading a joblib, retrain in the SAME
virtual environment you use for prediction/Streamlit.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Score text as Fake/Real using a trained Stage-3 model.")

    p.add_argument("--model", default="artifacts/best_model.joblib", help="Path to joblib model/pipeline.")
    p.add_argument("--labelmap", default="artifacts/label_map.json", help="Optional: label map created at training time.")
    p.add_argument(
        "--metrics",
        default="reports/metrics.json",
        help="Optional: Stage-3 metrics.json (used to auto-load chosen_threshold when --threshold is omitted).",
    )

    p.add_argument("--data", default="", help="Optional: CSV file to score (batch mode).")
    p.add_argument("--text-col", default="Text", help="Text column name for --data.")
    p.add_argument("--label-col", default="", help="Optional: ground-truth label column for evaluation (e.g., label).")
    p.add_argument("--id-col", default="", help="Optional: id column to carry through into the output.")
    p.add_argument("--out", default="reports/preds_scored.csv", help="Where to write scored CSV (batch mode).")

    p.add_argument("--text", default="", help="Optional: score a single text string (one-off mode).")

    # If omitted, we try to read chosen_threshold from --metrics, else fall back to 0.5.
    p.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Threshold for Fake when probabilities are available. If omitted, uses chosen_threshold from metrics.json when available.",
    )

    p.add_argument("--pos-name", default="Fake", help="Display name for label=1.")
    p.add_argument("--neg-name", default="Real", help="Display name for label=0.")
    p.add_argument("--max-rows", type=int, default=0, help="Optional: limit rows scored in batch mode (0 = no limit).")

    return p.parse_args()


# -----------------------------
# Helpers
# -----------------------------
def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_model(path: Path) -> Tuple[Any, list[str]]:
    """Load model and capture warnings as strings."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m = joblib.load(path)
    warn_msgs = [str(x.message) for x in w]
    return m, warn_msgs


def get_final_estimator(model: Any) -> Any:
    """If this is a sklearn Pipeline, return the last estimator; else return the model itself."""
    if hasattr(model, "named_steps"):
        try:
            return list(model.named_steps.values())[-1]
        except Exception:
            return model
    return model


def read_chosen_threshold(metrics_path: Path) -> Optional[float]:
    """Read chosen threshold from Stage-3 metrics.json, if present."""
    mj = load_json(metrics_path)
    if not mj:
        return None

    # Stage 3 key
    if "chosen_threshold" in mj:
        try:
            return float(mj["chosen_threshold"])
        except Exception:
            return None

    # Stage 2/other fallback: tolerate older schemas
    for k in ("threshold", "best_threshold"):
        if k in mj:
            try:
                return float(mj[k])
            except Exception:
                return None

    return None


def build_label_maps(
    label_map_json: Dict[str, Any],
    pos_name: str,
    neg_name: str,
) -> Tuple[Dict[int, str], Dict[str, int]]:
    """\
    Returns:
      id_to_name: {0: "Real", 1: "Fake"}
      name_to_id: {"Real": 0, "Fake": 1} (plus lowercase variants, plus "0"/"1")
    """
    if label_map_json:
        # accept either {"0":"Real","1":"Fake"} OR {"Real":0,"Fake":1}
        if all(str(k).isdigit() for k in label_map_json.keys()):
            id_to_name = {int(k): str(v) for k, v in label_map_json.items()}
        else:
            id_to_name = {int(v): str(k) for k, v in label_map_json.items()}
    else:
        id_to_name = {0: neg_name, 1: pos_name}

    # Ensure defaults exist
    id_to_name.setdefault(0, neg_name)
    id_to_name.setdefault(1, pos_name)

    name_to_id: Dict[str, int] = {}
    for k, v in id_to_name.items():
        name_to_id[str(v)] = int(k)
        name_to_id[str(v).strip().lower()] = int(k)

    name_to_id["0"] = 0
    name_to_id["1"] = 1
    return id_to_name, name_to_id


def pos_index_from_estimator(estimator: Any, pos_label: int = 1) -> Optional[int]:
    """\
    Return the column index in predict_proba/decision_function corresponding to pos_label (default 1).
    Uses estimator.classes_ when available.
    """
    classes = getattr(estimator, "classes_", None)
    if classes is None:
        # best-effort: binary assumes second column is "positive"
        return 1

    classes = np.asarray(classes)
    for i, c in enumerate(classes):
        if c == pos_label or str(c) == str(pos_label):
            return i

    if len(classes) == 2:
        return 1
    return None


def coerce_pred_to_int(y_pred: np.ndarray, name_to_id: Dict[str, int]) -> np.ndarray:
    """\
    Convert predicted labels (could be ints, numeric strings, or 'Fake'/'Real') into 0/1 when possible.
    Unknown values become -1.
    """
    y_pred = np.asarray(y_pred)

    if y_pred.dtype.kind in {"i", "u"}:
        return y_pred.astype(int, copy=False)

    out = []
    for v in y_pred:
        s = str(v).strip()
        if s.isdigit():
            out.append(int(s))
        else:
            out.append(name_to_id.get(s, name_to_id.get(s.lower(), -1)))
    return np.asarray(out, dtype=int)


def predict_texts(
    model: Any,
    texts: np.ndarray,
    threshold: float,
    name_to_id: Dict[str, int],
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """\
    Returns:
      y_pred_int (0/1 best-effort; unknown -> -1)
      y_proba_fake (probability of Fake when available)
      score_fake (decision score when available but probabilities are not)
    """
    est = get_final_estimator(model)

    # 1) Probabilities (preferred)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(texts)
        idx = pos_index_from_estimator(est, pos_label=1)

        y_proba_fake = None
        if idx is not None and proba is not None and proba.ndim == 2 and proba.shape[1] >= 2:
            y_proba_fake = proba[:, idx].astype(float)

        if y_proba_fake is not None:
            y_pred_int = (y_proba_fake >= threshold).astype(int)
        else:
            y_pred_raw = model.predict(texts)
            y_pred_int = coerce_pred_to_int(y_pred_raw, name_to_id)

        return y_pred_int, y_proba_fake, None

    # 2) Decision function (when predict_proba is unavailable)
    if hasattr(model, "decision_function"):
        score = np.asarray(model.decision_function(texts))

        idx = pos_index_from_estimator(est, pos_label=1)
        if score.ndim == 2 and score.shape[1] >= 2:
            score_fake = score[:, idx] if idx is not None else score[:, 1]
        else:
            score_fake = score

        y_pred_raw = model.predict(texts)
        y_pred_int = coerce_pred_to_int(y_pred_raw, name_to_id)
        return y_pred_int, None, score_fake.astype(float)

    # 3) Fallback
    y_pred_raw = model.predict(texts)
    y_pred_int = coerce_pred_to_int(y_pred_raw, name_to_id)
    return y_pred_int, None, None


def coerce_y_true(series: pd.Series, name_to_id: Dict[str, int]) -> pd.Series:
    """\
    Converts labels like 'Fake'/'Real' OR 0/1 into 0/1 ints (nullable).
    Unknown labels become <NA>.
    """
    if series.dtype.kind in {"i", "u"}:
        return series.astype("Int64")

    s = series.astype(str).str.strip()
    mapped = s.map(lambda x: name_to_id.get(x, name_to_id.get(x.lower(), None)))
    return mapped.astype("Int64")


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    args = parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model, warn_msgs = load_model(model_path)
    if warn_msgs:
        print("WARNING: scikit-learn warnings detected while loading the model.")
        print("Recommendation: align scikit-learn versions (or retrain) to avoid invalid results.\n")
        for w in warn_msgs[:6]:
            print(f"- {w}")
        if len(warn_msgs) > 6:
            print(f"- ... ({len(warn_msgs) - 6} more)")
        print()

    label_map_json = load_json(Path(args.labelmap))
    id_to_name, name_to_id = build_label_maps(label_map_json, args.pos_name, args.neg_name)

    # Resolve threshold
    threshold_used: float
    if args.threshold is not None:
        threshold_used = float(args.threshold)
    else:
        t = read_chosen_threshold(Path(args.metrics))
        threshold_used = float(t) if t is not None else 0.5

    # Mode 1: one-off scoring
    if args.text.strip():
        x = np.array([args.text.strip()], dtype=object)
        try:
            y_pred, y_proba_fake, score_fake = predict_texts(model, x, threshold_used, name_to_id)
        except Exception as e:
            raise RuntimeError(
                "Prediction failed. If you loaded an old joblib with a different sklearn version, "
                "retrain the model in this environment and try again."
            ) from e

        pred_id = int(y_pred[0])
        pred_label = id_to_name.get(pred_id, f"Unknown({pred_id})")
        print(f"prediction={pred_id} ({pred_label})")

        if y_proba_fake is not None:
            print(f"p_fake={float(y_proba_fake[0]):.4f}  threshold_used={threshold_used:.3f}")

        if score_fake is not None:
            print(f"score_fake={float(score_fake[0]):.4f}  (raw decision score; not a calibrated probability)")

        return

    # Mode 2: batch scoring
    if not args.data:
        raise ValueError("Provide either --text '...' (one-off) OR --data path/to/file.csv (batch).")

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Data not found: {data_path}")

    df = pd.read_csv(data_path)
    if args.text_col not in df.columns:
        raise KeyError(f"CSV must contain text column {args.text_col!r}. Found: {list(df.columns)}")

    if args.max_rows and args.max_rows > 0:
        df = df.head(args.max_rows).copy()

    text_series = df[args.text_col].astype(str).fillna("")
    x = text_series.to_numpy(dtype=object)

    try:
        y_pred, y_proba_fake, score_fake = predict_texts(model, x, threshold_used, name_to_id)
    except Exception as e:
        raise RuntimeError(
            "Batch prediction failed. If you loaded an old joblib with a different sklearn version, "
            "retrain the model in this environment and try again."
        ) from e

    out_df = pd.DataFrame()

    if args.id_col:
        if args.id_col not in df.columns:
            raise KeyError(f"--id-col {args.id_col!r} not found in CSV columns.")
        out_df[args.id_col] = df[args.id_col]

    out_df[args.text_col] = text_series
    out_df["y_pred"] = y_pred.astype(int)
    out_df["pred_label"] = out_df["y_pred"].map(lambda i: id_to_name.get(int(i), f"Unknown({i})"))

    if y_proba_fake is not None:
        out_df["y_proba_fake"] = y_proba_fake
        out_df["threshold_used"] = threshold_used

    if score_fake is not None:
        out_df["score_fake"] = score_fake

    # Optional evaluation if y_true available
    if args.label_col:
        if args.label_col not in df.columns:
            raise KeyError(f"--label-col {args.label_col!r} not found in CSV columns.")

        out_df["label_raw"] = df[args.label_col]
        out_df["y_true"] = coerce_y_true(df[args.label_col], name_to_id)
        out_df["is_correct"] = (out_df["y_true"] == out_df["y_pred"]).astype("Int64")

        valid = out_df["y_true"].notna()
        if valid.any():
            acc = float((out_df.loc[valid, "y_true"] == out_df.loc[valid, "y_pred"]).mean())
            print(f"Evaluation (rows with labels): {valid.sum():,} / {len(out_df):,}")
            print(f"Accuracy: {acc:.4f}")
        else:
            print("Note: label column provided, but labels could not be mapped to 0/1 (check label_map.json).")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    print("Saved predictions:")
    print(f"  out:  {out_path}")
    print(f"  rows: {len(out_df):,}")
    if y_proba_fake is not None:
        print(f"  threshold_used: {threshold_used:.3f} (from CLI if provided, else metrics.json if available)")


if __name__ == "__main__":
    main()
