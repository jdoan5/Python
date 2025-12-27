#!/usr/bin/env python3
"""
Predict with a trained TF-IDF + Logistic Regression fake-news baseline (Stage 1).

Batch:
  python src/predict.py --model artifacts/fake_news_tfidf_logreg.joblib \
    --data data/fake_and_real_news.csv --text-col Text --out artifacts/preds_scored.csv

One-off:
  python src/predict.py --model artifacts/fake_news_tfidf_logreg.joblib \
    --text "Breaking: ..."
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Score text as Fake/Real using a trained Stage-1 model.")
    p.add_argument("--model", default="artifacts/fake_news_tfidf_logreg.joblib", help="Path to joblib model pipeline.")
    p.add_argument("--labelmap", default="artifacts/label_map.json", help="Optional label map JSON (0/1 -> name).")

    p.add_argument("--data", default="", help="Optional: CSV file to score (batch mode).")
    p.add_argument("--text-col", default="text", help="Text column name for --data.")
    p.add_argument("--id-col", default="", help="Optional: id column to carry through (e.g., id).")
    p.add_argument("--out", default="artifacts/preds_scored.csv", help="Output CSV for batch predictions.")

    p.add_argument("--text", default="", help="Optional: score a single text string (one-off mode).")
    p.add_argument("--threshold", type=float, default=0.5, help="Threshold for label=1 when probabilities exist.")
    return p.parse_args()


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_model(path: Path) -> Tuple[Any, list[str]]:
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m = joblib.load(path)
    return m, [str(x.message) for x in w]


def display_label(label: Any, label_map: Dict[str, Any]) -> str:
    if not label_map:
        return "Fake" if int(label) == 1 else "Real"
    return str(label_map.get(str(label), label))


def predict_array(model: Any, texts: np.ndarray, threshold: float) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(texts)[:, 1]
            y_pred = (proba >= threshold).astype(int)
            return y_pred, proba
    except Exception:
        pass

    y_pred = model.predict(texts).astype(int)
    return y_pred, None


def main() -> None:
    args = parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model, warn_msgs = load_model(model_path)
    if warn_msgs:
        print("WARNING: scikit-learn version mismatch detected while loading the model.")
        print("Recommendation: align scikit-learn versions (or retrain) to avoid invalid results.\n")

    label_map = read_json(Path(args.labelmap))

    if args.text.strip():
        x = np.array([args.text.strip()], dtype=object)
        y_pred, y_proba = predict_array(model, x, args.threshold)
        lab = display_label(int(y_pred[0]), label_map)
        print(f"prediction={int(y_pred[0])} ({lab})")
        if y_proba is not None:
            print(f"p_fake={float(y_proba[0]):.4f} threshold={args.threshold:.2f}")
        return

    if not args.data:
        raise ValueError("Provide either --text '...' (one-off) OR --data path/to/file.csv (batch).")

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Data not found: {data_path}")

    df = pd.read_csv(data_path)
    if args.text_col not in df.columns:
        raise KeyError(f"CSV must contain text column {args.text_col!r}. Found: {list(df.columns)}")

    keep_cols = []
    if args.id_col:
        if args.id_col not in df.columns:
            raise KeyError(f"--id-col {args.id_col!r} not found in CSV columns.")
        keep_cols.append(args.id_col)

    text_series = df[args.text_col].astype(str).fillna("")
    x = text_series.to_numpy(dtype=object)

    y_pred, y_proba = predict_array(model, x, args.threshold)

    out_df = pd.DataFrame()
    if keep_cols:
        out_df[keep_cols] = df[keep_cols]
    out_df[args.text_col] = text_series
    out_df["y_pred"] = y_pred
    out_df["pred_label"] = [display_label(int(v), label_map) for v in y_pred]

    if y_proba is not None:
        out_df["y_proba_fake"] = y_proba

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    print("Saved predictions:")
    print(f"  out:  {out_path}")
    print(f"  rows: {len(out_df)}")


if __name__ == "__main__":
    main()