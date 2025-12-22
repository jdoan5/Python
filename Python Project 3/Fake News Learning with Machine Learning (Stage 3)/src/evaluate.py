#!/usr/bin/env python3
"""
Evaluate a trained Fake News model (Stage 3): threshold sweep + chosen threshold.

Why Stage 3 evaluate.py exists (vs Stage 2):
- Stage 2 compares multiple baseline models and selects the best.
- Stage 3 focuses on probabilities + thresholding (often after calibration), producing:
    reports/threshold_sweep.csv
    reports/metrics.json  (updated with chosen_threshold + thresholded metrics)

This script:
- Loads a trained model (default: artifacts/best_model.joblib)
- Loads labeled data (default: data/fake_and_real_news.csv)
- Gets P(Fake) (prefer predict_proba; else decision_function -> sigmoid best-effort)
- Sweeps thresholds and writes threshold_sweep.csv
- Chooses the best threshold (default: maximize F1 for Fake) and updates metrics.json

Run:
  python src/evaluate.py \
    --model artifacts/best_model.joblib \
    --data data/fake_and_real_news.csv \
    --text-col Text \
    --label-col label \
    --out reports/threshold_sweep.csv \
    --metrics reports/metrics.json
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
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 3 evaluator: threshold sweep + chosen threshold.")
    p.add_argument("--model", default="artifacts/best_model.joblib", help="Path to trained joblib model/pipeline.")
    p.add_argument("--labelmap", default="artifacts/label_map.json", help="Optional label map from training.")
    p.add_argument("--data", default="data/fake_and_real_news.csv", help="CSV dataset path (must include labels).")
    p.add_argument("--text-col", default="Text", help="Text column name.")
    p.add_argument("--label-col", default="label", help="Label column name.")
    p.add_argument("--test-size", type=float, default=0.2, help="Holdout size for evaluation/sweep.")
    p.add_argument("--random-state", type=int, default=42, help="Random seed.")
    p.add_argument("--out", default="reports/threshold_sweep.csv", help="Output CSV for threshold sweep.")
    p.add_argument("--metrics", default="reports/metrics.json", help="metrics.json to update/create.")
    p.add_argument(
        "--metric",
        default="f1_fake",
        choices=["f1_fake", "f1_macro", "accuracy"],
        help="Metric to maximize when choosing the threshold.",
    )
    p.add_argument("--steps", type=int, default=101, help="Number of thresholds between 0..1 to evaluate.")
    p.add_argument("--min-threshold", type=float, default=0.0, help="Minimum threshold (inclusive).")
    p.add_argument("--max-threshold", type=float, default=1.0, help="Maximum threshold (inclusive).")
    p.add_argument("--max-rows", type=int, default=0, help="Optional cap rows for quick runs (0 = no cap).")
    return p.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def load_model(path: Path) -> Tuple[Any, list[str]]:
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m = joblib.load(path)
    warn_msgs = [str(x.message) for x in w]
    return m, warn_msgs


def get_final_estimator(model: Any) -> Any:
    if hasattr(model, "named_steps"):
        try:
            return list(model.named_steps.values())[-1]
        except Exception:
            return model
    return model


def build_label_map(y: pd.Series, label_map_json: Dict[str, Any]) -> Tuple[np.ndarray, Dict[int, str]]:
    """
    Normalize labels to ints (ideally 0/1). Prefer artifacts/label_map.json if present.
    Supports:
      - label_map_json {"0":"Real","1":"Fake"} or {"Real":0,"Fake":1}
      - raw labels like Fake/Real
      - raw labels like 0/1 or "0"/"1"
    """
    if label_map_json:
        if all(str(k).strip().isdigit() for k in label_map_json.keys()):
            id_to_name = {int(k): str(v) for k, v in label_map_json.items()}
        else:
            id_to_name = {int(v): str(k) for k, v in label_map_json.items()}

        name_to_id: Dict[str, int] = {}
        for k, v in id_to_name.items():
            name_to_id[str(v)] = int(k)
            name_to_id[str(v).strip().lower()] = int(k)
        name_to_id["0"] = 0
        name_to_id["1"] = 1

        y_raw = y.astype(str).str.strip()
        mapped = []
        for val in y_raw.tolist():
            s = str(val).strip()
            if s.isdigit():
                mapped.append(int(s))
            else:
                mapped.append(name_to_id.get(s, name_to_id.get(s.lower(), -1)))
        y_int = np.asarray(mapped, dtype=int)
        if (y_int == -1).mean() < 0.05:
            return y_int, id_to_name

    y_raw = y.astype(str).str.strip()
    lowered = y_raw.str.lower()
    if set(lowered.unique()) <= {"fake", "real"}:
        id_to_name = {0: "Real", 1: "Fake"}
        y_int = (lowered == "fake").astype(int).to_numpy()
        return y_int, id_to_name

    try:
        y_int = y_raw.astype(int).to_numpy()
        uniq = sorted(set(y_int.tolist()))
        id_to_name = {int(u): str(u) for u in uniq}
        if uniq == [0, 1]:
            id_to_name = {0: "Real", 1: "Fake"}
        return y_int, id_to_name
    except Exception:
        codes, uniques = pd.factorize(y_raw)
        id_to_name = {int(i): str(name) for i, name in enumerate(uniques)}
        return codes.astype(int), id_to_name


def pos_index_from_estimator(estimator: Any, pos_label: int = 1) -> Optional[int]:
    classes = getattr(estimator, "classes_", None)
    if classes is None:
        return 1
    classes = np.asarray(classes)
    for i, c in enumerate(classes):
        if c == pos_label or str(c) == str(pos_label):
            return i
    if len(classes) == 2:
        return 1
    return None


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x, dtype=float)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    expx = np.exp(x[~pos])
    out[~pos] = expx / (1.0 + expx)
    return out


def score_proba_fake(model: Any, X: np.ndarray) -> Tuple[np.ndarray, str]:
    """
    Return a 0..1 score representing P(Fake) when possible.
    Prefer predict_proba; fallback to decision_function->sigmoid (best-effort, not calibrated).
    Returns: (scores, method)
    """
    est = get_final_estimator(model)

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        idx = pos_index_from_estimator(est, pos_label=1)
        if idx is None:
            idx = 1 if proba.shape[1] >= 2 else 0
        return proba[:, idx].astype(float), "predict_proba"

    if hasattr(model, "decision_function"):
        score = np.asarray(model.decision_function(X)).ravel().astype(float)
        return sigmoid(score), "decision_function_sigmoid"

    y_pred = np.asarray(model.predict(X)).astype(int).ravel()
    return y_pred.astype(float), "predict_hard_labels"


def sweep_thresholds(y_true: np.ndarray, p_fake: np.ndarray, thresholds: np.ndarray) -> pd.DataFrame:
    rows = []
    for t in thresholds:
        y_pred = (p_fake >= float(t)).astype(int)
        acc = float(accuracy_score(y_true, y_pred))
        f1m = float(f1_score(y_true, y_pred, average="macro"))
        prec, rec, f1s, sup = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)
        rows.append(
            {
                "threshold": float(t),
                "accuracy": acc,
                "f1_macro": f1m,
                "precision_real": float(prec[0]),
                "recall_real": float(rec[0]),
                "f1_real": float(f1s[0]),
                "precision_fake": float(prec[1]),
                "recall_fake": float(rec[1]),
                "f1_fake": float(f1s[1]),
                "support_real": int(sup[0]),
                "support_fake": int(sup[1]),
            }
        )
    return pd.DataFrame(rows)


def choose_best(df: pd.DataFrame, metric: str) -> pd.Series:
    order = [metric]
    for k in ("f1_fake", "f1_macro", "accuracy"):
        if k not in order:
            order.append(k)

    tmp = df.copy()
    for c in order:
        tmp[c] = pd.to_numeric(tmp[c], errors="coerce").fillna(-1.0)
    tmp = tmp.sort_values(order + ["threshold"], ascending=[False] * len(order) + [True], kind="mergesort")
    return tmp.iloc[0]


def main() -> None:
    args = parse_args()

    root = Path(__file__).resolve().parents[1]
    model_path = (root / args.model).resolve() if not Path(args.model).is_absolute() else Path(args.model)
    data_path = (root / args.data).resolve() if not Path(args.data).is_absolute() else Path(args.data)
    labelmap_path = (root / args.labelmap).resolve() if not Path(args.labelmap).is_absolute() else Path(args.labelmap)
    out_path = (root / args.out).resolve() if not Path(args.out).is_absolute() else Path(args.out)
    metrics_path = (root / args.metrics).resolve() if not Path(args.metrics).is_absolute() else Path(args.metrics)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    model, warn_msgs = load_model(model_path)
    if warn_msgs:
        print("WARNING: scikit-learn warnings detected while loading the model.")
        print("Recommendation: align scikit-learn versions (or retrain) to avoid invalid results.\n")
        for w in warn_msgs[:6]:
            print(f"- {w}")
        if len(warn_msgs) > 6:
            print(f"- ... ({len(warn_msgs) - 6} more)")
        print()

    df = pd.read_csv(data_path)
    if args.text_col not in df.columns:
        raise KeyError(f"Text column {args.text_col!r} not found. Found: {list(df.columns)}")
    if args.label_col not in df.columns:
        raise KeyError(f"Label column {args.label_col!r} not found. Found: {list(df.columns)}")

    if args.max_rows and args.max_rows > 0:
        df = df.head(args.max_rows).copy()

    label_map_json = load_json(labelmap_path)
    y_int, id_to_name = build_label_map(df[args.label_col], label_map_json)

    keep = y_int != -1
    if keep.mean() < 1.0:
        dropped = int((~keep).sum())
        print(f"Note: dropping {dropped:,} rows with unmapped labels (-1).")
        df = df.loc[keep].copy()
        y_int = y_int[keep]

    X = df[args.text_col].astype(str).fillna("").to_numpy(dtype=object)

    X_train, X_eval, y_train, y_eval = train_test_split(
        X,
        y_int,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y_int if len(np.unique(y_int)) > 1 else None,
    )

    # We do NOT refit here; we evaluate the already-trained model.
    p_fake, score_method = score_proba_fake(model, X_eval)

    thresholds = np.linspace(args.min_threshold, args.max_threshold, args.steps)
    sweep_df = sweep_thresholds(y_eval, p_fake, thresholds)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sweep_df.to_csv(out_path, index=False)

    best_row = choose_best(sweep_df, args.metric)
    chosen_threshold = float(best_row["threshold"])
    y_pred_best = (p_fake >= chosen_threshold).astype(int)

    metrics = load_json(metrics_path)
    metrics.update(
        {
            "stage": 3,
            "model_path": str(model_path),
            "dataset": str(data_path),
            "text_col": args.text_col,
            "label_col": args.label_col,
            "eval_split": {"test_size": args.test_size, "random_state": args.random_state},
            "threshold_sweep_path": str(out_path),
            "chosen_threshold": chosen_threshold,
            "threshold_metric": args.metric,
            "threshold_score_method": score_method,
            "accuracy_at_threshold": float(accuracy_score(y_eval, y_pred_best)),
            "f1_macro_at_threshold": float(f1_score(y_eval, y_pred_best, average="macro")),
            "f1_fake_at_threshold": float(f1_score(y_eval, y_pred_best, pos_label=1)),
            "confusion_matrix_at_threshold": confusion_matrix(y_eval, y_pred_best).tolist(),
            "classification_report_at_threshold": classification_report(y_eval, y_pred_best, output_dict=True),
            "label_map_int_to_name": {str(k): v for k, v in id_to_name.items()},
            "note": "Stage 3 adds threshold tuning over P(Fake); chosen_threshold maximizes the selected metric.",
        }
    )
    save_json(metrics_path, metrics)

    print("Stage 3 evaluation complete.")
    print(f"- threshold sweep: {out_path}")
    print(f"- metrics updated:  {metrics_path}")
    print(f"Chosen threshold: {chosen_threshold:.3f} (metric={args.metric})")


if __name__ == "__main__":
    main()
