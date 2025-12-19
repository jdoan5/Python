#!/usr/bin/env python3
"""
train.py — Fake News Learning with Machine Learning

Trains a baseline NLP classifier (TF-IDF + Logistic Regression) on a CSV file.

Default label convention:
- 0 = Real
- 1 = Fake

If your labels are strings (e.g., "REAL"/"FAKE"), pass --label-map-json.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from joblib import dump
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


DEFAULT_LABEL_MAP = {
    "REAL": 0, "FAKE": 1,
    "Real": 0, "Fake": 1,
    "real": 0, "fake": 1,
    0: 0, 1: 1, "0": 0, "1": 1,
}


@dataclass
class TrainConfig:
    data_path: Path
    text_col: str
    label_col: str
    test_size: float
    random_state: int
    outdir: Path
    reportdir: Path
    label_map: Dict[Any, int]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Fake News (TF-IDF + Logistic Regression).")
    p.add_argument("--data", required=True, help="Path to CSV (must include text + label columns).")
    p.add_argument("--text-col", default="text", help="Name of the text column. Default: text")
    p.add_argument("--label-col", default="label", help="Name of the label column. Default: label")
    p.add_argument("--test-size", type=float, default=0.2, help="Test split fraction. Default: 0.2")
    p.add_argument("--random-state", type=int, default=42, help="Random seed. Default: 42")
    p.add_argument("--outdir", default="models", help="Directory to write model artifacts. Default: models")
    p.add_argument("--reportdir", default="reports", help="Directory to write metrics. Default: reports")
    p.add_argument(
        "--label-map-json",
        default=None,
        help="Optional JSON string or file path mapping raw labels to {0,1}. "
             "Example: '{\"REAL\":0,\"FAKE\":1}'",
    )
    return p.parse_args()


def load_label_map(arg: Optional[str]) -> Dict[Any, int]:
    if not arg:
        return dict(DEFAULT_LABEL_MAP)

    try:
        if os.path.exists(arg):
            with open(arg, "r", encoding="utf-8") as f:
                obj = json.load(f)
        else:
            obj = json.loads(arg)
    except Exception as e:
        raise ValueError(f"Could not parse --label-map-json: {e}") from e

    label_map: Dict[Any, int] = {}
    for k, v in obj.items():
        if int(v) not in (0, 1):
            raise ValueError("Label map values must be 0 or 1.")
        label_map[k] = int(v)
        label_map[str(k)] = int(v)
    return label_map

# Checking for the .csv file
def validate_dataframe(df: pd.DataFrame, text_col: str, label_col: str) -> None:
    missing = [c for c in (text_col, label_col) if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found: {list(df.columns)}")
    if df.empty:
        raise ValueError("Input CSV is empty.")
    if df[text_col].isna().all():
        raise ValueError(f"Column '{text_col}' has no usable text (all null).")
    if df[label_col].isna().all():
        raise ValueError(f"Column '{label_col}' has no usable labels (all null).")


def map_labels(series: pd.Series, label_map: Dict[Any, int]) -> pd.Series:
    mapped = series.map(label_map)
    if mapped.isna().any():
        bad_vals = sorted(set(series[mapped.isna()].astype(str).tolist()))
        raise ValueError(
            "Some labels could not be mapped to 0/1. "
            f"Unmapped values (sample): {bad_vals[:15]} "
            "Fix your CSV labels or pass --label-map-json."
        )
    return mapped.astype(int)


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(
                lowercase=True,
                stop_words="english",
                ngram_range=(1, 2),
                max_features=50000,
            )),
            ("clf", LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
            )),
        ]
    )


def compute_metrics(y_true, y_pred) -> Dict[str, Any]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred, digits=4, output_dict=True),
        "note": "Label convention: 0=Real, 1=Fake.",
    }


def main() -> None:
    args = parse_args()
    cfg = TrainConfig(
        data_path=Path(args.data),
        text_col=args.text_col,
        label_col=args.label_col,
        test_size=args.test_size,
        random_state=args.random_state,
        outdir=Path(args.outdir),
        reportdir=Path(args.reportdir),
        label_map=load_label_map(args.label_map_json),
    )

    df = pd.read_csv(cfg.data_path)
    validate_dataframe(df, cfg.text_col, cfg.label_col)

    X = df[cfg.text_col].astype(str).fillna("")
    y = map_labels(df[cfg.label_col], cfg.label_map)

    stratify = y if y.nunique() > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=stratify,
    )

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    metrics = compute_metrics(y_test, y_pred)

    cfg.outdir.mkdir(parents=True, exist_ok=True)
    cfg.reportdir.mkdir(parents=True, exist_ok=True)

    model_path = cfg.outdir / "fake_news_tfidf_logreg.joblib"
    dump(pipe, model_path)

    label_map_path = cfg.outdir / "label_map.json"
    with open(label_map_path, "w", encoding="utf-8") as f:
        json.dump(
            {"label_to_name": {"0": "REAL", "1": "FAKE"}},
            f,
            indent=2,
        )

    metrics_path = cfg.reportdir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Training complete.")
    print(f"Model:   {model_path}")
    print(f"Labels:  {label_map_path}")
    print(f"Metrics: {metrics_path}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")


if __name__ == "__main__":
    main()
