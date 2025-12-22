#!/usr/bin/env python3
"""
Evaluate + compare multiple baseline text classifiers (Stage 2).

What it does:
- Loads a CSV dataset (default: data/fake_and_real_news.csv)
- Splits train/test
- Trains multiple TF-IDF baselines:
    1) Logistic Regression
    2) Linear SVM (LinearSVC)
    3) Multinomial Naive Bayes
- Compares metrics, selects best by macro F1 (tie-break by accuracy)
- Writes artifacts:
    reports/model_comparison.csv
    reports/metrics.json         (best model)
    reports/preds.csv            (best model)
    artifacts/best_model.joblib
    artifacts/label_map.json     (int -> display name)

Run:
  python src/evaluate.py --data data/fake_and_real_news.csv --text-col Text --label-col label
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 2 evaluator: compare baseline text classifiers and write reports.")
    p.add_argument("--data", default="data/fake_and_real_news.csv", help="Path to CSV dataset.")
    p.add_argument("--text-col", default="Text", help="Name of the text column.")
    p.add_argument("--label-col", default="label", help="Name of the label column.")
    p.add_argument("--test-size", type=float, default=0.2, help="Test split size.")
    p.add_argument("--random-state", type=int, default=42, help="Random seed.")
    p.add_argument("--ngram-max", type=int, default=1, help="1 for unigrams, 2 for unigrams+bigrams, etc.")
    p.add_argument("--min-df", type=int, default=2, help="Min document frequency for TF-IDF.")
    p.add_argument("--max-features", type=int, default=50000, help="Max TF-IDF features.")
    p.add_argument("--outdir", default="artifacts", help="Artifacts output directory.")
    p.add_argument("--reportdir", default="reports", help="Reports output directory.")
    return p.parse_args()


# -----------------------------
# Helpers
# -----------------------------
def ensure_dirs(*dirs: Path) -> None:
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def build_label_map(y: pd.Series) -> Tuple[np.ndarray, Dict[int, str]]:
    """
    Normalize labels to ints 0/1 (or more, if dataset has >2 classes).
    Returns:
      y_int: ndarray[int]
      label_map: dict[int -> display_name]
    Supports:
      - labels like "Fake"/"Real"
      - labels like 0/1
      - labels like "0"/"1"
    """
    y_raw = y.astype(str).str.strip()

    # Common binary case: Fake/Real
    lowered = y_raw.str.lower()
    if set(lowered.unique()) <= {"fake", "real"}:
        label_map = {0: "Real", 1: "Fake"}
        y_int = (lowered == "fake").astype(int).to_numpy()
        return y_int, label_map

    # If looks numeric already
    try:
        y_int = y_raw.astype(int).to_numpy()
        uniq = sorted(set(y_int.tolist()))
        # Provide generic mapping if not binary
        if uniq == [0, 1]:
            label_map = {0: "Real", 1: "Fake"}
        else:
            label_map = {int(u): str(u) for u in uniq}
        return y_int, label_map
    except Exception:
        # Fallback: factorize
        codes, uniques = pd.factorize(y_raw)
        label_map = {int(i): str(name) for i, name in enumerate(uniques)}
        return codes.astype(int), label_map


def make_pipelines(ngram_max: int, min_df: int, max_features: int) -> Dict[str, Pipeline]:
    vec = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, ngram_max),
        min_df=min_df,
        max_features=max_features,
    )

    return {
        "tfidf_logreg": Pipeline(
            steps=[
                ("tfidf", vec),
                ("clf", LogisticRegression(max_iter=2000, n_jobs=None)),
            ]
        ),
        "tfidf_linear_svm": Pipeline(
            steps=[
                ("tfidf", vec),
                ("clf", LinearSVC()),
            ]
        ),
        "tfidf_nb": Pipeline(
            steps=[
                ("tfidf", vec),
                ("clf", MultinomialNB()),
            ]
        ),
    }


def safe_proba(model: Pipeline, X: np.ndarray) -> np.ndarray | None:
    """
    Returns probability for class 1 when available, else None.
    Note: LinearSVC doesn't provide predict_proba.
    """
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        if probs.ndim == 2 and probs.shape[1] >= 2:
            return probs[:, 1]
    return None


@dataclass
class ModelResult:
    model_name: str
    accuracy: float
    f1_macro: float
    train_seconds: float


def evaluate_one(model_name: str, pipe: Pipeline, X_train, y_train, X_test, y_test) -> Tuple[ModelResult, np.ndarray, np.ndarray | None]:
    start = time.time()
    pipe.fit(X_train, y_train)
    train_seconds = time.time() - start

    y_pred = pipe.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))
    f1m = float(f1_score(y_test, y_pred, average="macro"))
    proba = safe_proba(pipe, X_test)

    return ModelResult(model_name, acc, f1m, float(train_seconds)), y_pred, proba


def pick_best(results: list[ModelResult]) -> ModelResult:
    # Best by macro F1; tie-break by accuracy
    return sorted(results, key=lambda r: (r.f1_macro, r.accuracy), reverse=True)[0]


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    args = parse_args()

    root = Path(__file__).resolve().parents[1]
    data_path = (root / args.data).resolve() if not Path(args.data).is_absolute() else Path(args.data)
    outdir = root / args.outdir
    reportdir = root / args.reportdir
    ensure_dirs(outdir, reportdir)

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path)
    if args.text_col not in df.columns:
        raise KeyError(f"Text column '{args.text_col}' not found. Found: {list(df.columns)}")
    if args.label_col not in df.columns:
        raise KeyError(f"Label column '{args.label_col}' not found. Found: {list(df.columns)}")

    X = df[args.text_col].astype(str).fillna("").to_numpy(dtype=object)
    y_int, label_map = build_label_map(df[args.label_col])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_int,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y_int if len(np.unique(y_int)) > 1 else None,
    )

    pipelines = make_pipelines(args.ngram_max, args.min_df, args.max_features)

    results: list[ModelResult] = []
    preds_cache: Dict[str, Tuple[Pipeline, np.ndarray, np.ndarray | None]] = {}

    for name, pipe in pipelines.items():
        res, y_pred, proba = evaluate_one(name, pipe, X_train, y_train, X_test, y_test)
        results.append(res)
        preds_cache[name] = (pipe, y_pred, proba)

    best = pick_best(results)
    best_model, best_pred, best_proba = preds_cache[best.model_name]

    # Write model comparison CSV
    comp_df = pd.DataFrame([asdict(r) for r in results]).sort_values(
        ["f1_macro", "accuracy"], ascending=False
    )
    comp_path = reportdir / "model_comparison.csv"
    comp_df.to_csv(comp_path, index=False)

    # Write metrics.json for best model
    metrics: Dict[str, Any] = {
        "model_name": best.model_name,
        "accuracy": float(best.accuracy),
        "f1_macro": float(best.f1_macro),
        "confusion_matrix": confusion_matrix(y_test, best_pred).tolist(),
        "classification_report": classification_report(y_test, best_pred, output_dict=True),
        "label_map_int_to_name": {str(k): v for k, v in label_map.items()},
        "dataset": str(data_path),
        "text_col": args.text_col,
        "label_col": args.label_col,
        "tfidf": {
            "ngram_range": [1, args.ngram_max],
            "min_df": args.min_df,
            "max_features": args.max_features,
            "stop_words": "english",
        },
        "split": {"test_size": args.test_size, "random_state": args.random_state},
        "note": "Stage 2 compares baseline models; best model selected by macro F1.",
    }
    metrics_path = reportdir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Write preds.csv for best model
    preds_df = pd.DataFrame(
        {
            args.text_col: X_test,
            "y_true": y_test,
            "y_pred": best_pred,
        }
    )
    if best_proba is not None:
        preds_df["pred_proba_fake"] = best_proba.astype(float)

    preds_path = reportdir / "preds.csv"
    preds_df.to_csv(preds_path, index=False)

    # Save best model + label map to artifacts/
    best_model_path = outdir / "best_model.joblib"
    joblib.dump(best_model, best_model_path)

    label_map_path = outdir / "label_map.json"
    label_map_path.write_text(json.dumps({str(k): v for k, v in label_map.items()}, indent=2), encoding="utf-8")

    print("Stage 2 evaluation complete.")
    print(f"- reports/model_comparison.csv: {comp_path}")
    print(f"- reports/metrics.json:         {metrics_path}")
    print(f"- reports/preds.csv:            {preds_path}")
    print(f"- artifacts/best_model.joblib:  {best_model_path}")
    print(f"- artifacts/label_map.json:     {label_map_path}")
    print(f"Best model: {best.model_name} (f1_macro={best.f1_macro:.4f}, acc={best.accuracy:.4f})")


if __name__ == "__main__":
    main()