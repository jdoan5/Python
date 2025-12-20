#!/usr/bin/env python3
"""
Train a TF-IDF + Logistic Regression baseline for fake-news classification (Stage 1).

Supports label columns that are either:
- integers: 0/1
- strings: "Real"/"Fake" (case-insensitive)

Outputs (by default):
  artifacts/fake_news_tfidf_logreg.joblib
  artifacts/metrics.json
  artifacts/label_map.json
  artifacts/preds.csv   (test split predictions)

Example:
  python src/train.py --data data/fake_and_real_news.csv --text-col Text --label-col label
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train TF-IDF + Logistic Regression fake-news baseline (Stage 1).")
    p.add_argument("--data", default="data/fake_and_real_news.csv", help="Path to CSV dataset.")
    p.add_argument("--text-col", default="text", help="Name of text column in CSV.")
    p.add_argument("--label-col", default="label", help="Name of label column in CSV.")
    p.add_argument("--test-size", type=float, default=0.2, help="Test split size.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument("--out-model", default="artifacts/fake_news_tfidf_logreg.joblib", help="Output joblib model path.")
    p.add_argument("--out-metrics", default="artifacts/metrics.json", help="Output metrics JSON path.")
    p.add_argument("--out-labelmap", default="artifacts/label_map.json", help="Output label map JSON path.")
    p.add_argument("--out-preds", default="artifacts/preds.csv", help="Output test predictions CSV path.")
    return p.parse_args()


def normalize_labels(y: pd.Series) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Convert labels to {0,1}. Returns (y_int, metadata_dict).

    Accepted:
    - 0/1
    - "Real"/"Fake" (any casing, surrounding whitespace ok)
    """
    y_clean = y.astype(str).str.strip()

    # If already numeric-like 0/1, keep.
    uniq = set(y_clean.unique())
    if uniq.issubset({"0", "1"}):
        return y_clean.astype(int).to_numpy(), {"type": "numeric", "mapping": {"0": 0, "1": 1}}

    # Common string labels
    lower = y_clean.str.lower()
    uniq_lower = set(lower.unique())

    mapping = {
        "real": 0,
        "fake": 1,
        # optional extras
        "true": 0,
        "false": 1,
    }

    if uniq_lower.issubset(set(mapping.keys())):
        y_int = lower.map(mapping).astype(int).to_numpy()
        seen_map = {lab: mapping[lab] for lab in sorted(uniq_lower)}
        return y_int, {"type": "string", "mapping": seen_map}

    raise ValueError(
        "Unsupported label values. Expected 0/1 or Real/Fake. "
        f"Found: {sorted(list(uniq))[:10]} (showing up to 10 unique values)"
    )


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path)

    if args.text_col not in df.columns:
        raise KeyError(f"Missing text column {args.text_col!r}. Columns: {list(df.columns)}")
    if args.label_col not in df.columns:
        raise KeyError(f"Missing label column {args.label_col!r}. Columns: {list(df.columns)}")

    # Basic cleaning
    df = df[[args.text_col, args.label_col]].dropna()
    df[args.text_col] = df[args.text_col].astype(str).str.strip()
    df = df[df[args.text_col] != ""]

    y, label_meta = normalize_labels(df[args.label_col])
    X = df[args.text_col].to_numpy(dtype=object)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    model = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 1))),
            ("clf", LogisticRegression(max_iter=2000)),
        ]
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred).tolist()

    metrics = {
        "accuracy": acc,
        "confusion_matrix": cm,
        "classification_report": report,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "label_meta": label_meta,
        "note": "Stage 1 baseline: TF-IDF (unigrams) + Logistic Regression.",
    }

    out_model = Path(args.out_model)
    out_metrics = Path(args.out_metrics)
    out_labelmap = Path(args.out_labelmap)
    out_preds = Path(args.out_preds)

    for pth in (out_model, out_metrics, out_labelmap, out_preds):
        ensure_parent(pth)

    joblib.dump(model, out_model)
    out_metrics.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Display map used by dashboard/UI
    label_map = {"0": "Real", "1": "Fake"}
    out_labelmap.write_text(json.dumps(label_map, indent=2), encoding="utf-8")

    # Save test predictions for review
    preds_df = pd.DataFrame({"text": X_test, "y_true": y_test, "y_pred": y_pred.astype(int)})
    if hasattr(model, "predict_proba"):
        preds_df["y_proba_fake"] = model.predict_proba(X_test)[:, 1]
    preds_df.to_csv(out_preds, index=False)

    print("Training complete.")
    print(f"  model:   {out_model}")
    print(f"  metrics: {out_metrics}")
    print(f"  labels:  {out_labelmap}")
    print(f"  preds:   {out_preds}")
    print(f"  accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()