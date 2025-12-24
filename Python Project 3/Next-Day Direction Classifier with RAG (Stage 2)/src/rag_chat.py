from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import CFG
from src.data_fetch import fetch_from_yahoo, save_raw
from src.features import add_features, feature_columns


def time_split(df: pd.DataFrame):
    """Time-based split (no shuffling)."""
    n = len(df)
    n_train = int(n * CFG.train_ratio)
    n_val = int(n * CFG.val_ratio)

    train = df.iloc[:n_train].copy()
    val = df.iloc[n_train : n_train + n_val].copy()
    test = df.iloc[n_train + n_val :].copy()
    return train, val, test


def baseline_predict_up_majority(y_train: pd.Series, y_eval: pd.Series):
    """Baseline: always predict the majority class from the training set."""
    majority = int(y_train.mean() >= 0.5)
    return [majority] * len(y_eval)


def save_processed(df: pd.DataFrame, symbol: str, stamp: str) -> str:
    """Save features+target dataset for reproducibility."""
    CFG.data_processed.mkdir(parents=True, exist_ok=True)
    out = CFG.data_processed / f"{symbol}_processed_{stamp}.csv"
    df.to_csv(out, index=False)
    return str(out)


def main():
    # One stamp for the entire run (raw, processed, runs/)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    symbol = CFG.symbol

    # Ensure dirs exist
    CFG.runs_dir.mkdir(parents=True, exist_ok=True)
    CFG.data_raw.mkdir(parents=True, exist_ok=True)
    CFG.data_processed.mkdir(parents=True, exist_ok=True)

    # 1) Fetch RAW (Yahoo) + save
    raw = fetch_from_yahoo(symbol, start=CFG.start_date, end=CFG.end_date)
    raw_path = save_raw(raw, symbol)

    # 2) Build features/target
    df = add_features(raw)

    feats = feature_columns()

    # Guard: drop any rows with missing feature/target
    needed = feats + ["target_up"]
    df = df.dropna(subset=needed).reset_index(drop=True)

    if len(df) < 200:
        raise SystemExit(
            f"Not enough rows after feature engineering: {len(df)}. "
            "Try an earlier start_date or a more liquid symbol."
        )

    processed_path = save_processed(df, symbol, stamp)

    # 3) Split
    train_df, val_df, test_df = time_split(df)

    X_train, y_train = train_df[feats], train_df["target_up"]
    X_val, y_val = val_df[feats], val_df["target_up"]
    X_test, y_test = test_df[feats], test_df["target_up"]

    # 4) Baseline
    base_val_pred = baseline_predict_up_majority(y_train, y_val)
    base_val_acc = accuracy_score(y_val, base_val_pred)

    # 5) Model
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, random_state=CFG.random_state)),
        ]
    )

    model.fit(X_train, y_train)

    val_pred = model.predict(X_val)
    val_acc = accuracy_score(y_val, val_pred)

    # (Optional quick test metric here; full test report in evaluate.py)
    test_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, test_pred)

    # 6) Save run artifacts
    run_dir = CFG.runs_dir / f"run_{symbol}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, run_dir / "model.joblib")

    meta = {
        "provider": getattr(CFG, "provider", "yahoo"),
        "symbol": symbol,
        "start_date": CFG.start_date,
        "end_date": CFG.end_date,
        "timestamp": stamp,
        "raw_path": raw_path,
        "processed_path": processed_path,
        "n_rows_raw": int(len(raw)),
        "n_rows_after_features": int(len(df)),
        "split": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
        "features": feats,
        "baseline_val_accuracy": float(base_val_acc),
        "model_val_accuracy": float(val_acc),
        "quick_test_accuracy": float(test_acc),
    }
    (run_dir / "metrics.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    report = []
    report.append(f"# Run Report — {symbol} — {stamp}\n\n")
    report.append("## Notes\n")
    report.append("- Time-based split (no shuffling)\n")
    report.append("- Local learning project; not investment advice.\n\n")
    report.append("## Data Artifacts\n")
    report.append(f"- Raw: `{raw_path}`\n")
    report.append(f"- Processed: `{processed_path}`\n\n")
    report.append("## Validation Metrics\n")
    report.append(f"- Baseline (majority) accuracy: **{base_val_acc:.3f}**\n")
    report.append(f"- Logistic Regression accuracy: **{val_acc:.3f}**\n")
    report.append(f"- Quick TEST accuracy: **{test_acc:.3f}**\n\n")
    report.append("## Classification Report (VAL)\n")
    report.append("```text\n")
    report.append(classification_report(y_val, val_pred))
    report.append("\n```\n")

    (run_dir / "report.md").write_text("".join(report), encoding="utf-8")

    print(f"Saved raw data to:       {raw_path}")
    print(f"Saved processed data to: {processed_path}")
    print(f"Saved run to:            {run_dir}")
    print(f"Baseline VAL acc: {base_val_acc:.3f}")
    print(f"Model    VAL acc: {val_acc:.3f}")
    print(f"Quick   TEST acc: {test_acc:.3f}")


if __name__ == "__main__":
    main()