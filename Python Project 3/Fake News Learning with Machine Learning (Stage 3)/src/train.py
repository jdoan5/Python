#!/usr/bin/env python3
"""
Stage 3 — Fake News Learning with Machine Learning

Stage 3 builds on Stage 2 (model comparison + optional n-grams) and adds:
  1) Cross-validated hyperparameter tuning per model (GridSearchCV)
  2) Probability calibration for models that don't output probabilities (e.g., LinearSVC)
  3) Threshold tuning (choose the decision threshold that maximizes a chosen metric on a validation split)
  4) Cleaner, recruiter-friendly reporting artifacts under reports/

Default outputs:
  artifacts/best_model.joblib
  artifacts/label_map.json

  reports/model_comparison.csv         # best CV score + best params per model
  reports/threshold_sweep.csv          # threshold tuning curve (validation split)
  reports/metrics.json                 # final test metrics + chosen threshold
  reports/preds.csv                    # test predictions (incl. y_proba_fake + thresholded y_pred)

Example:
  python src/train.py --data data/fake_and_real_news.csv --text-col Text --label-col label --search-ngrams --calibrate
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
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
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage-3 training: CV tuning + calibration + threshold selection.")
    p.add_argument("--data", required=True, help="CSV path (e.g., data/fake_and_real_news.csv)")
    p.add_argument("--text-col", default="Text", help="Text column name (e.g., Text)")
    p.add_argument("--label-col", default="label", help="Label column name (e.g., label)")

    p.add_argument("--test-size", type=float, default=0.2, help="Test split fraction (default 0.2)")
    p.add_argument("--val-size", type=float, default=0.2, help="Validation fraction from train split (default 0.2)")
    p.add_argument("--random-state", type=int, default=42, help="Random seed (default 42)")

    p.add_argument("--max-features", type=int, default=50000, help="TF-IDF max_features (default 50000)")
    p.add_argument("--use-bigrams", action="store_true", help="Use ngram_range=(1,2) (fixed).")
    p.add_argument("--search-ngrams", action="store_true", help="Search ngram_range in {(1,1),(1,2)} via CV.")

    p.add_argument("--cv-folds", type=int, default=5, help="Cross-validation folds (default 5)")
    p.add_argument("--n-jobs", type=int, default=-1, help="Parallel jobs for GridSearchCV (default -1)")

    p.add_argument(
        "--calibrate",
        action="store_true",
        help="Calibrate probabilities for the final model (recommended if using LinearSVC).",
    )
    p.add_argument(
        "--calibration-method",
        choices=["sigmoid", "isotonic"],
        default="sigmoid",
        help="Calibration method (default sigmoid).",
    )

    p.add_argument(
        "--threshold-metric",
        choices=["f1_fake", "macro_f1", "accuracy"],
        default="f1_fake",
        help="Metric to optimize when selecting the Fake threshold (default f1_fake).",
    )
    p.add_argument("--threshold-grid", type=int, default=181, help="Number of thresholds from 0.05..0.95 (default 181)")

    p.add_argument("--artifacts-dir", default="artifacts", help="Where to write model/label_map (default artifacts/)")
    p.add_argument("--reports-dir", default="reports", help="Where to write reports (default reports/)")

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
    return {str(v): str(k) for k, v in label_map.items()}


# -----------------------------
# Experiments
# -----------------------------
@dataclass(frozen=True)
class ModelSpec:
    name: str
    estimator: Any
    param_grid: Dict[str, List[Any]]


def make_base_pipeline(max_features: int, ngram_range: Tuple[int, int], estimator: Any) -> Pipeline:
    tfidf = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        max_features=max_features,
        ngram_range=ngram_range,
    )
    return Pipeline([("tfidf", tfidf), ("clf", estimator)])


def build_model_specs() -> List[ModelSpec]:
    """
    Stage 3 includes small, practical grids. You can expand grids later.
    """
    return [
        ModelSpec(
            name="tfidf_logreg",
            estimator=LogisticRegression(max_iter=3000, solver="liblinear"),
            param_grid={
                "clf__C": [0.25, 0.5, 1.0, 2.0, 4.0],
            },
        ),
        ModelSpec(
            name="tfidf_linear_svm",
            estimator=LinearSVC(),
            param_grid={
                "clf__C": [0.25, 0.5, 1.0, 2.0, 4.0],
            },
        ),
        ModelSpec(
            name="tfidf_multinomial_nb",
            estimator=MultinomialNB(),
            param_grid={
                "clf__alpha": [0.25, 0.5, 1.0, 2.0],
            },
        ),
    ]


def evaluate_binary(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Binary metrics where label 1 corresponds to Fake by convention (when Real/Fake exists).
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


def get_proba_fake(model: Any, X: np.ndarray) -> Optional[np.ndarray]:
    """
    Returns P(class=1) when available, else None.
    Works for Pipeline or calibrated estimators as long as predict_proba exists.
    """
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba is not None and proba.shape[1] >= 2:
            return proba[:, 1].astype(float)
    return None


def threshold_sweep(
    y_true: np.ndarray,
    p_fake: np.ndarray,
    thresholds: np.ndarray,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for t in thresholds:
        y_pred = (p_fake >= t).astype(int)
        m = evaluate_binary(y_true, y_pred)
        rows.append({"threshold": float(t), **m})
    return pd.DataFrame(rows)


def pick_threshold(df: pd.DataFrame, metric: str) -> Tuple[float, Dict[str, float]]:
    if df.empty:
        return 0.5, {}
    if metric not in df.columns:
        metric = "f1_fake" if "f1_fake" in df.columns else "macro_f1"
    best = df.sort_values(metric, ascending=False, kind="mergesort").iloc[0]
    t = float(best["threshold"])
    m = {k: float(best[k]) for k in ["accuracy", "macro_f1", "f1_fake", "precision_fake", "recall_fake"] if k in best}
    return t, m


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

    # Prepare X/y
    X_all = df[args.text_col].astype(str).fillna("").to_numpy(dtype=object)
    y_raw = df[args.label_col]
    label_map_str_to_int = build_label_map(y_raw)
    y_all = encode_labels(y_raw, label_map_str_to_int)

    # Save label map (int->string)
    (artifacts_dir / "label_map.json").write_text(
        json.dumps(invert_label_map(label_map_str_to_int), indent=2),
        encoding="utf-8",
    )

    # Train/Test split
    stratify = y_all if len(np.unique(y_all)) > 1 else None
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X_all,
        y_all,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=stratify,
    )

    # Train/Val split for threshold tuning
    stratify_tv = y_trainval if len(np.unique(y_trainval)) > 1 else None
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=args.val_size,
        random_state=args.random_state,
        stratify=stratify_tv,
    )

    # Decide n-gram search space
    if args.search_ngrams:
        ngram_candidates = [(1, 1), (1, 2)]
    else:
        ngram_candidates = [(1, 2) if args.use_bigrams else (1, 1)]

    cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.random_state)

    model_specs = build_model_specs()
    comparison_rows: List[Dict[str, Any]] = []

    best_overall: Dict[str, Any] = {"score": -1.0}
    best_search: Optional[GridSearchCV] = None
    best_ngram: Tuple[int, int] = (1, 1)

    # CV tuning per model (and optionally per ngram setting)
    for spec in model_specs:
        for ngram in ngram_candidates:
            pipe = make_base_pipeline(args.max_features, ngram, spec.estimator)

            search = GridSearchCV(
                estimator=pipe,
                param_grid=spec.param_grid,
                scoring="f1_macro",
                cv=cv,
                n_jobs=args.n_jobs,
                verbose=0,
            )
            search.fit(X_train, y_train)

            best_cv_score = float(search.best_score_)
            best_params = search.best_params_

            comparison_rows.append(
                {
                    "experiment": spec.name,
                    "ngram_range": f"({ngram[0]},{ngram[1]})",
                    "max_features": args.max_features,
                    "cv_f1_macro": best_cv_score,
                    "best_params": json.dumps(best_params, sort_keys=True),
                }
            )

            if best_cv_score > float(best_overall["score"]):
                best_overall = {
                    "score": best_cv_score,
                    "experiment": spec.name,
                    "ngram_range": ngram,
                    "best_params": best_params,
                }
                best_search = search
                best_ngram = ngram

    comparison_df = pd.DataFrame(comparison_rows).sort_values("cv_f1_macro", ascending=False)
    (reports_dir / "model_comparison.csv").write_text(comparison_df.to_csv(index=False), encoding="utf-8")

    if best_search is None:
        raise RuntimeError("No successful model search. Check your data and parameter grids.")

    # Fit best pipeline on TRAIN (not trainval) then calibrate on VAL (or via CV) as configured.
    best_pipeline: Pipeline = best_search.best_estimator_

    # Refit on full train split with best params (GridSearch already refit=True by default).
    # Still, for clarity and determinism, rebuild and fit:
    winner_name = str(best_overall["experiment"])
    winner_params = dict(best_overall["best_params"])
    winner_ngram = tuple(best_overall["ngram_range"])

    # Build a fresh estimator instance of the winning type
    spec_lookup = {s.name: s for s in model_specs}
    winner_spec = spec_lookup[winner_name]

    winner_pipe = make_base_pipeline(args.max_features, winner_ngram, winner_spec.estimator)
    winner_pipe.set_params(**winner_params)
    winner_pipe.fit(X_train, y_train)

    # Optional calibration (recommended for LinearSVC)
    final_model: Any = winner_pipe
    calibrated = False
    if args.calibrate:
        # CalibratedClassifierCV will wrap the pipeline and provide predict_proba.
        final_model = CalibratedClassifierCV(
            estimator=winner_pipe,
            method=args.calibration_method,
            cv=3,
        )
        final_model.fit(X_val, y_val)  # fit calibration using the validation split labels
        calibrated = True

    # Threshold tuning (needs probabilities)
    p_val = get_proba_fake(final_model, X_val)
    thresholds = np.linspace(0.05, 0.95, args.threshold_grid)

    if p_val is None:
        # No probabilities; default thresholding via predict().
        threshold_df = pd.DataFrame()
        chosen_threshold = 0.5
        chosen_val_metrics = {}
    else:
        threshold_df = threshold_sweep(y_val, p_val, thresholds)
        threshold_df.to_csv(reports_dir / "threshold_sweep.csv", index=False)
        chosen_threshold, chosen_val_metrics = pick_threshold(threshold_df, args.threshold_metric)

    # Final evaluation on TEST
    if p_val is not None:
        p_test = get_proba_fake(final_model, X_test)
        if p_test is None:
            y_pred_test = np.asarray(final_model.predict(X_test)).astype(int, copy=False)
            p_test = None
        else:
            y_pred_test = (p_test >= chosen_threshold).astype(int)
    else:
        p_test = None
        y_pred_test = np.asarray(final_model.predict(X_test)).astype(int, copy=False)

    test_metrics = evaluate_binary(y_test, y_pred_test)

    preds_df = pd.DataFrame(
        {
            args.text_col: X_test,
            "y_true": y_test,
            "y_pred": y_pred_test.astype(int),
        }
    )
    if p_test is not None:
        preds_df["y_proba_fake"] = p_test
        preds_df["threshold"] = chosen_threshold
    preds_df["is_correct"] = (preds_df["y_true"] == preds_df["y_pred"]).astype(int)
    preds_df.to_csv(reports_dir / "preds.csv", index=False)

    # Save final model
    joblib.dump(final_model, artifacts_dir / "best_model.joblib")

    metrics_json: Dict[str, Any] = {
        "stage": 3,
        "best_experiment": winner_name,
        "best_params": {k: str(v) for k, v in winner_params.items()},
        "ngram_range": f"({winner_ngram[0]},{winner_ngram[1]})",
        "max_features": args.max_features,
        "calibrated": calibrated,
        "calibration_method": args.calibration_method if calibrated else None,
        "threshold_metric": args.threshold_metric,
        "chosen_threshold": chosen_threshold,
        "val_metrics_at_chosen_threshold": chosen_val_metrics,
        "test_metrics": test_metrics,
        "classification_report": classification_report(y_test, y_pred_test, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred_test).tolist(),
        "label_convention": "0=Real, 1=Fake (if dataset includes Real/Fake).",
        "env": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "note": (
            "Stage 3 adds CV tuning, calibration, and threshold tuning. "
            "Threshold tuning is applied only when probabilities are available."
        ),
    }
    (reports_dir / "metrics.json").write_text(json.dumps(metrics_json, indent=2), encoding="utf-8")

    print("Stage 3 training complete.")
    print(f"Best model: {winner_name}  ngram={winner_ngram}  cv_f1_macro={best_overall['score']:.4f}")
    print(f"Saved model: {artifacts_dir / 'best_model.joblib'}")
    print(f"Saved label map: {artifacts_dir / 'label_map.json'}")
    print(f"Saved reports: {reports_dir / 'model_comparison.csv'}, {reports_dir / 'metrics.json'}, {reports_dir / 'preds.csv'}")
    if p_val is not None:
        print(f"Chosen threshold: {chosen_threshold:.3f}  (optimized {args.threshold_metric} on validation)")
        print(f"Saved threshold sweep: {reports_dir / 'threshold_sweep.csv'}")


if __name__ == "__main__":
    main()
