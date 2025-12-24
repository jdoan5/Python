from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.config import CFG
from src.data_fetch import fetch_from_yahoo
from src.features import add_features, feature_columns
from src.train import time_split


def main(run_dir: str):
    run_path = Path(run_dir)
    model = joblib.load(run_path / "model.joblib")

    # Re-fetch data (same source + date range as training)
    raw = fetch_from_yahoo(CFG.symbol, start=CFG.start_date, end=CFG.end_date)
    df = add_features(raw)

    feats = feature_columns()

    # Drop rows where rolling features/target may be NaN
    needed_cols = feats + ["target_up"]
    df = df.dropna(subset=needed_cols).reset_index(drop=True)

    train_df, val_df, test_df = time_split(df)

    if len(test_df) == 0:
        raise RuntimeError(
            "Test split is empty. Adjust train_ratio/val_ratio or extend your date range."
        )

    X_test = test_df[feats]
    y_test = test_df["target_up"]

    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    cm = confusion_matrix(y_test, pred).tolist()

    out = {
        "symbol": CFG.symbol,
        "start_date": CFG.start_date,
        "end_date": CFG.end_date,
        "n_rows_after_features": int(len(df)),
        "split": {
            "train": int(len(train_df)),
            "val": int(len(val_df)),
            "test": int(len(test_df)),
        },
        "test_accuracy": float(acc),
        "confusion_matrix": cm,
        "classification_report": classification_report(y_test, pred, output_dict=True),
    }
    (run_path / "test_metrics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"TEST accuracy: {acc:.3f}")
    print("Confusion matrix:", cm)


if __name__ == "__main__":
    # Example: python -m src.evaluate runs/run_SPY_20250101_120000
    import sys

    if len(sys.argv) < 2:
        raise SystemExit("Usage: python -m src.evaluate <run_dir>")
    main(sys.argv[1])