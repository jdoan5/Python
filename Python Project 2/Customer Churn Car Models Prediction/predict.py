#!/usr/bin/env python3
"""
Predict on a CSV and KEEP the original columns (including 'churn' if present).
Appends:
  - prediction  : 0/1 label
  - probability : P(positive class)

Usage:
  python3 predict.py --file data.csv --model model.joblib --out _predictions_check.csv
  # optional custom threshold (instead of model.predict):
  python3 predict.py --file data.csv --model model.joblib --out _predictions_check.csv --threshold 0.5
"""

import argparse
from pathlib import Path
import sys
import pandas as pd
import numpy as np
from joblib import load

def main():
    ap = argparse.ArgumentParser(description="Run model inference and keep original columns.")
    ap.add_argument("--file", required=True, help="Input CSV with features (may include 'churn').")
    ap.add_argument("--model", default="model.joblib", help="Trained model/pipeline (joblib).")
    ap.add_argument("--out", required=True, help="Output CSV with predictions appended.")
    ap.add_argument("--threshold", type=float, default=None,
                    help="If set, use predict_proba >= threshold for labels. Otherwise use model.predict().")
    args = ap.parse_args()

    in_path = Path(args.file)
    model_path = Path(args.model)
    out_path = Path(args.out)

    if not in_path.exists():
        print(f"[ERROR] input file not found: {in_path}", file=sys.stderr)
        sys.exit(1)
    if not model_path.exists():
        print(f"[ERROR] model file not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    # Load data
    df = pd.read_csv(in_path)

    # Build feature table: drop columns we know are not features (safe if missing)
    non_feature_cols = ["churn", "prediction", "probability", "pred_label", "pred_proba_yes"]
    X = df.drop(columns=non_feature_cols, errors="ignore")

    # Load model/pipeline
    model = load(model_path)

    # Predict proba
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        # assume positive class is column 1
        p_pos = proba[:, 1]
        if args.threshold is not None:
            y_pred = (p_pos >= args.threshold).astype(int)
        else:
            # use the model's default classification if available; fallback to threshold=0.5
            if hasattr(model, "predict") and args.threshold is None:
                y_pred = model.predict(X)
                # Ensure ints 0/1 (some models return strings; coerce)
                try:
                    y_pred = pd.Series(y_pred).astype(int).values
                except Exception:
                    y_pred = (p_pos >= 0.5).astype(int)
            else:
                y_pred = (p_pos >= 0.5).astype(int)
    else:
        # No predict_proba: fall back to predict and set probability to NaN
        if not hasattr(model, "predict"):
            print("[ERROR] model has neither predict_proba nor predict.", file=sys.stderr)
            sys.exit(1)
        y_pred = model.predict(X)
        try:
            y_pred = pd.Series(y_pred).astype(int).values
        except Exception:
            # map typical yes/no, true/false cases to 1/0 if needed
            y_pred = pd.Series(y_pred).astype(str).str.lower().map(
                {"1":1,"0":0,"true":1,"false":0,"yes":1,"no":0}
            ).fillna(0).astype(int).values
        p_pos = np.full(shape=(len(df),), fill_value=np.nan)

    # Append to original data (KEEPS 'churn' or any other columns)
    out = df.copy()
    out["prediction"] = y_pred
    out["probability"] = p_pos

    out.to_csv(out_path, index=False)
    print(f"Wrote predictions → {out_path}")
    print("Columns now include:", list(out.columns))

if __name__ == "__main__":
    main()