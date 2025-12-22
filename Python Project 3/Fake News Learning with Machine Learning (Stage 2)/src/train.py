#!/usr/bin/env python3
"""
Stage 2 — Fake News Learning with Machine Learning

Keeps Stage 1 baseline (TF-IDF + Logistic Regression) and adds:
  1) Model comparison: Logistic Regression vs Linear SVM vs Naive Bayes
  2) Optional n-grams (bigrams): TfidfVectorizer(ngram_range=(1,2))

Outputs (defaults):
  artifacts/best_model.joblib
  artifacts/label_map.json
  reports/metrics.json
  reports/preds.csv
  reports/model_comparison.csv

Example:
  python src/train.py --data data/fake_and_real_news.csv --text-col Text --label-col label --use-bigrams
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

try:
    import sklearn
    SKLEARN_VERSION = sklearn.__version__
except Exception:
    SKLEARN_VERSION = "unknown"


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Stage-2 fake news classifier with model comparison.")
    p.add_argument("--data", required=True, help="CSV path (e.g., data/fake_and_real_news.csv)")
    p.add_argument("--text-col", default="text", help="Text column name (e.g., Text)")
    p.add_argument("--label-col", default="label", help="Label column name (e.g., label)")

    p.add_argument("--test-size", type=float, default=0.2, help="Test split fraction (default 0.2)")
    p.add_argument("--random-state", type=int, default=42, help="Random seed (default 42)")

    p.add_argument("--max-features", type=int, default=50000, help="TF-IDF max_features (default 50000)")
    p.add_argument("--use-bigrams", action="store_true", help="Use ngram_range=(1,2) instead of (1,1)")

    p.add_argument("--artifacts-dir", default="artifacts", help="Where to write model/label_map (default artifacts/)")
    p.add_argument("--reports-dir", default="reports", help="Where to write metrics/preds/comparison (default reports/)")

    return p.parse_args()


# -----------------------------
# Label handling
# -----------------------------
def build_label_map(y_raw: pd.Series) -> Dict[str, int]:
    """
    Build a stable label map. If labels include Real/Fake, force:
      Real -> 0
      Fake -> 1
    Otherwise, map sorted unique labels to 0..N-1.
    """
    y_str = y_raw.astype(str).str.strip()

    uniq = sorted(y_str.unique().tolist())
    lower = [u.lower() for u in uniq]

    if "real" in lower and "fake" in lower:
        real_token = uniq[lower.index("real")]
        fake_token = uniq[lower.index("fake")]
        return {real_token: 0, fake_token: 1}

    return {label: i for i, label in enumerate(uniq)}


def encode_labels(y_raw: pd.Series, label_map: Dict[str, int]) -> np.ndarray:
    y_str = y_raw.astype(str).str.strip()
    unknown = sorted(set(y_str.unique()) - set(label_map.keys()))
    if unknown:
        raise ValueError(f"Found labels not in label_map: {unknown}. label_map={label_map}")
    return y_str.map(label_map).astype(int).to_numpy()


def invert_label_map(label_map: Dict[str, int]) -> Dict[str, str]:
    # int -> string label (nice for JSON + Streamlit)
    inv: Dict[str, str] = {}
    for k, v in label_map.items():
        inv[str(v)] = str(k)
    return inv


# -----------------------------
# Experiment setup
# -----------------------------
@dataclass(frozen=True)
class Experiment:
    name: str
    estimator: Any


def make_vectorizer(max_features: int, use_bigrams: bool) -> TfidfVectorizer:
    return TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        max_features=max_features,
        ngram_range=(1, 2) if use_bigrams else (1, 1),
    )


def build_experiments() -> List[Experiment]:
    """
    Stage 2 comparison set:
      - Logistic Regression (baseline)
      - Linear SVM (LinearSVC)
      - Multinomial Naive Bayes
    """
    return [
        Experiment("tfidf_logreg", LogisticRegression(max_iter=2000, solver="liblinear")),
        Experiment("tfidf_linear_svm", LinearSVC()),
        Experiment("tfidf_multinomial_nb", MultinomialNB()),
    ]


def safe_predict_proba(pipeline: Pipeline, X: np.ndarray) -> np.ndarray | None:
    """
    Some models (LinearSVC) do not implement predict_proba.
    Return probability of class 1 if available, else None.
    """
    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba(X)
        if proba is not None and proba.shape[1] >= 2:
            return proba[:, 1]
    return None


def evaluate_run(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Metrics: accuracy, precision/recall/f1 for label=1 (Fake by convention), plus macro_f1.
    """
    out: Dict[str, float] = {}
    out["accuracy"] = float(accuracy_score(y_true, y_pred))

    labels = sorted(np.unique(y_true).tolist())
    if 1 in labels:
        out["precision_fake"] = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
        out["recall_fake"] = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
        out["f1_fake"] = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))
    else:
        out["precision_fake"] = float("nan")
        out["recall_fake"] = float("nan")
        out["f1_fake"] = float("nan")

    out["macro_f1"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    return out


def pick_best(df: pd.DataFrame) -> pd.Series:
    """
    Rank by macro_f1 first, then f1_fake, then accuracy.
    """
    tmp = df.copy()
    for c in ["macro_f1", "f1_fake", "accuracy"]:
        if c in tmp.columns:
            tmp[c] = pd.to_numeric(tmp[c], errors="coerce").fillna(-1.0)

    tmp = tmp.sort_values(
        by=["macro_f1", "f1_fake", "accuracy"],
        ascending=[False, False, False],
        kind="mergesort",
    )
    return tmp.iloc[0]


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    args = parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Data not found: {data_path}")

    artifacts_dir = Path(args.artifacts_dir)
    reports_dir = Path(args.reports_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)

    if args.text_col not in df.columns:
        raise KeyError(f"Missing text column {args.text_col!r}. Found: {list(df.columns)}")
    if args.label_col not in df.columns:
        raise KeyError(f"Missing label column {args.label_col!r}. Found: {list(df.columns)}")

    # Prepare X / y
    X_all = df[args.text_col].astype(str).fillna("").to_numpy(dtype=object)
    y_raw = df[args.label_col]

    label_map_str_to_int = build_label_map(y_raw)
    y_all = encode_labels(y_raw, label_map_str_to_int)

    # Save label map for UI (int -> string)
    label_map_int_to_str = invert_label_map(label_map_str_to_int)
    (artifacts_dir / "label_map.json").write_text(json.dumps(label_map_int_to_str, indent=2), encoding="utf-8")

    # Split
    stratify = y_all if len(np.unique(y_all)) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X_all,
        y_all,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=stratify,
    )

    # Run experiments (each gets fresh clones)
    experiments = build_experiments()
    rows: List[Dict[str, Any]] = []

    for exp in experiments:
        pipe = Pipeline(
            [
                ("tfidf", make_vectorizer(args.max_features, args.use_bigrams)),
                ("clf", clone(exp.estimator)),
            ]
        )

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        metrics = evaluate_run(y_test, y_pred)
        rows.append(
            {
                "experiment": exp.name,
                "ngram_range": "(1,2)" if args.use_bigrams else "(1,1)",
                "max_features": args.max_features,
                **metrics,
            }
        )

    comparison_df = pd.DataFrame(rows)
    comparison_path = reports_dir / "model_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)

    # Pick best
    best_row = pick_best(comparison_df)
    best_name = str(best_row["experiment"])

    winner_lookup = {e.name: e for e in experiments}
    if best_name not in winner_lookup:
        raise RuntimeError(f"Best experiment name not found: {best_name}")

    winner = winner_lookup[best_name]

    # Train final best model (fresh clone)
    best_model = Pipeline(
        [
            ("tfidf", make_vectorizer(args.max_features, args.use_bigrams)),
            ("clf", clone(winner.estimator)),
        ]
    )
    best_model.fit(X_train, y_train)

    # Final evaluation
    y_pred_best = best_model.predict(X_test)
    best_metrics = evaluate_run(y_test, y_pred_best)

    # preds.csv
    preds_df = pd.DataFrame({"text": X_test, "y_true": y_test, "y_pred": y_pred_best})
    proba_fake = safe_predict_proba(best_model, X_test)
    if proba_fake is not None:
        preds_df["y_proba_fake"] = proba_fake

    preds_path = reports_dir / "preds.csv"
    preds_df.to_csv(preds_path, index=False)

    # metrics.json
    metrics_json: Dict[str, Any] = {
        "stage": 2,
        "best_experiment": best_name,
        "accuracy": best_metrics["accuracy"],
        "macro_f1": best_metrics["macro_f1"],
        "f1_fake": best_metrics.get("f1_fake"),
        "precision_fake": best_metrics.get("precision_fake"),
        "recall_fake": best_metrics.get("recall_fake"),
        "ngram_range": "(1,2)" if args.use_bigrams else "(1,1)",
        "max_features": args.max_features,
        "label_convention": "0=Real, 1=Fake (if dataset provided Real/Fake labels).",
        "classification_report": classification_report(y_test, y_pred_best, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred_best).tolist(),
        "env": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "scikit_learn": SKLEARN_VERSION,
        },
        "note": (
            "Stage 2 adds model comparison and optional bigrams. "
            "Higher macro_f1/f1_fake generally means better balance on Fake vs Real."
        ),
    }
    metrics_path = reports_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics_json, indent=2), encoding="utf-8")

    # Save best model
    best_model_path = artifacts_dir / "best_model.joblib"
    joblib.dump(best_model, best_model_path)

    print("Stage 2 training complete.")
    print(f"Best model: {best_name}")
    print(f"Saved: {best_model_path}")
    print(f"Saved: {artifacts_dir / 'label_map.json'}")
    print(f"Saved: {metrics_path}")
    print(f"Saved: {preds_path}")
    print(f"Saved: {comparison_path}")
    print(
        f"Accuracy={best_metrics['accuracy']:.4f}  "
        f"macro_f1={best_metrics['macro_f1']:.4f}  "
        f"f1_fake={best_metrics.get('f1_fake')}"
    )


if __name__ == "__main__":
    main()