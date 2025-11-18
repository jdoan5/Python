#!/usr/bin/env python3
"""
evaluate_predictions.py — compute classification metrics for your predictions

Works with either column naming scheme:
- Ground truth: 'churn' (default; override with --target)
- Predicted label: 'pred_label' OR 'prediction' (optional)
- Predicted probability (positive class): 'pred_proba_yes' OR 'probability' (optional)

If you don't have a predicted label column, this script will derive labels from the probability
using a decision threshold (default 0.5; change with --threshold).

Examples
--------
python3 evaluate_predictions.py --file _predictions_check.csv
python3 evaluate_predictions.py --file predictions.csv --threshold 0.6 --out eval_report.md
python3 evaluate_predictions.py --file predictions.csv --label-col prediction --proba-col probability

Outputs printed to stdout and, if --out is provided, to a Markdown report.
"""
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    precision_recall_fscore_support,
    roc_curve,
)

def find_col(df, *names):
    for n in names:
        if n in df.columns:
            return n
    return None

def to_binary(series):
    """Map common label encodings to {0,1}. Accepts 1/0, 'yes'/'no', 'true'/'false'."""
    def map_one(x):
        if pd.isna(x):
            return np.nan
        if isinstance(x, (int, np.integer, float, np.floating)):
            # treat any nonzero as 1
            return 1 if float(x) >= 1 else 0
        s = str(x).strip().lower()
        if s in {"1", "true", "yes", "y"}:
            return 1
        if s in {"0", "false", "no", "n"}:
            return 0
        # fallback: try to cast
        try:
            return 1 if float(s) >= 1 else 0
        except Exception:
            return np.nan
    return series.apply(map_one).astype("float")

def main():
    ap = argparse.ArgumentParser(description="Evaluate predictions vs. ground truth.")
    ap.add_argument("--file", default="predictions.csv", help="CSV containing ground truth + predictions.")
    ap.add_argument("--target", default="churn", help="Ground-truth column name (default: churn)")
    ap.add_argument("--label-col", help="Predicted label column name (optional). If missing, will derive from proba.")
    ap.add_argument("--proba-col", help="Predicted probability column name (optional). If missing, will try common names.")
    ap.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for converting proba->label (default: 0.5)")
    ap.add_argument("--out", help="Write a Markdown report to this path (optional)")
    args = ap.parse_args()

    path = Path(args.file)
    if not path.exists():
        print(f"[ERROR] file not found: {path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(path)

    # Resolve columns
    y_col = args.target
    if y_col not in df.columns:
        print(f"[ERROR] ground-truth column '{y_col}' not in file. Found: {list(df.columns)}", file=sys.stderr)
        sys.exit(1)

    label_col = args.label_col or find_col(df, "pred_label", "prediction")
    proba_col  = args.proba_col or find_col(df, "pred_proba_yes", "probability")

    if label_col is None and proba_col is None:
        print("[ERROR] Need either a label column or a probability column.\n"
              "Looked for 'pred_label'/'prediction' and 'pred_proba_yes'/'probability'.", file=sys.stderr)
        sys.exit(1)

    # Prepare y_true (0/1)
    y_true = to_binary(df[y_col]).values
    if np.isnan(y_true).any():
        bad = int(np.isnan(y_true).sum())
        print(f"[WARN] {bad} rows in ground truth could not be interpreted as 0/1 and will be dropped.")
    keep = ~np.isnan(y_true)

    # Prepare y_proba and y_pred
    y_proba = None
    if proba_col is not None:
        y_proba = df.loc[keep, proba_col].astype(float).values

    if label_col is not None:
        y_pred = to_binary(df.loc[keep, label_col]).astype(int).values
        # if labels exist but also proba exists, keep both; otherwise derive y_pred from proba
    else:
        # derive from probability
        y_pred = (y_proba >= args.threshold).astype(int)

    y_true = y_true[keep].astype(int)

    # Metrics
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    report = classification_report(y_true, y_pred, digits=3, zero_division=0)
    acc = (y_true == y_pred).mean()

    roc_auc = None
    ap_score = None
    if y_proba is not None:
        try:
            roc_auc = roc_auc_score(y_true, y_proba)
        except Exception:
            roc_auc = None
        try:
            ap_score = average_precision_score(y_true, y_proba)  # PR AUC
        except Exception:
            ap_score = None

    # Print
    print("\n=== Columns used ===")
    print(f"ground truth: {y_col}")
    print(f"pred label : {label_col if label_col else f'[derived from {proba_col} @ {args.threshold}]'}")
    print(f"pred proba : {proba_col if proba_col else 'N/A'}")

    print("\n=== Confusion matrix (rows=true, cols=pred) ===")
    print(pd.DataFrame(cm, index=["true_0","true_1"], columns=["pred_0","pred_1"]).to_string())

    print("\n=== Classification report ===")
    print(report)

    print("=== Summary ===")
    print(f"Accuracy: {acc:.4f}")
    if roc_auc is not None:
        print(f"ROC AUC : {roc_auc:.4f}")
    if ap_score is not None:
        print(f"PR AUC  : {ap_score:.4f}")

    # Markdown report
    if args.out:
        lines = []
        lines.append(f"# Evaluation Report\n")
        lines.append(f"- File: `{path.name}`\n")
        lines.append(f"- Ground truth: `{y_col}`\n")
        lines.append(f"- Pred label: `{label_col if label_col else f'derived from {proba_col} @ {args.threshold}'}`\n")
        lines.append(f"- Pred proba: `{proba_col if proba_col else 'N/A'}`\n")
        lines.append("\n## Confusion matrix (rows=true, cols=pred)\n")
        lines.append(pd.DataFrame(cm, index=["true_0","true_1"], columns=["pred_0","pred_1"]).to_markdown())
        lines.append("\n## Classification report\n")
        lines.append("```\n" + report + "\n```")
        lines.append("\n## Summary\n")
        lines.append("```\n" + "\n".join([
            f"Accuracy: {acc:.4f}",
            f"ROC AUC : {roc_auc:.4f}" if roc_auc is not None else "ROC AUC : N/A",
            f"PR AUC  : {ap_score:.4f}" if ap_score is not None else "PR AUC  : N/A",
        ]) + "\n```")
        out_path = Path(args.out)
        out_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"\nWrote Markdown report → {out_path}")

if __name__ == "__main__":
    main()
