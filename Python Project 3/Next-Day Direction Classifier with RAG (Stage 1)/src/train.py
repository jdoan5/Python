from __future__ import annotations

import json
from datetime import datetime

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
    n = len(df)
    n_train = int(n * CFG.train_ratio)
    n_val = int(n * CFG.val_ratio)
    train = df.iloc[:n_train].copy()
    val = df.iloc[n_train : n_train + n_val].copy()
    test = df.iloc[n_train + n_val :].copy()
    return train, val, test


def baseline_predict_up_majority(y_train: pd.Series, y_eval: pd.Series):
    majority = int(y_train.mean() >= 0.5)
    return [majority] * len(y_eval)


def _safe_name(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in s)


def _save_processed(df: pd.DataFrame, symbol: str, stamp: str) -> str:
    """
    Save the feature-engineered dataset used for training.
    """
    CFG.data_processed.mkdir(parents=True, exist_ok=True)
    out = CFG.data_processed / f"{_safe_name(symbol)}_{stamp}_processed.csv"
    df.to_csv(out, index=False)
    return str(out)


def main():
    CFG.runs_dir.mkdir(parents=True, exist_ok=True)

    # 1) Fetch RAW and save to data/raw
    raw = fetch_from_yahoo(CFG.symbol, start=CFG.start_date, end=CFG.end_date)
    raw_path = save_raw(raw, _safe_name(CFG.symbol))

    # 2) Features
    df = add_features(raw)

    feats = feature_columns()

    # Keep only rows with usable features/label
    needed_cols = feats + ["target_up"]
    df = df.dropna(subset=needed_cols).reset_index(drop=True)

    if len(df) < 50:
        raise RuntimeError(
            f"Not enough rows after feature engineering ({len(df)}). "
            "Try a longer date range in config.py."
        )

    # 3) Split
    train_df, val_df, test_df = time_split(df)

    if len(val_df) == 0 or len(test_df) == 0:
        raise RuntimeError(
            "Validation/test split resulted in empty set. "
            "Adjust train_ratio/val_ratio or use more data."
        )

    X_train, y_train = train_df[feats], train_df["target_up"]
    X_val, y_val = val_df[feats], val_df["target_up"]
    X_test, y_test = test_df[feats], test_df["target_up"]

    # 4) Baseline
    base_val = baseline_predict_up_majority(y_train, y_val)
    base_val_acc = accuracy_score(y_val, base_val)

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

    # 6) Save run artifacts
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = CFG.runs_dir / f"run_{_safe_name(CFG.symbol)}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save processed training frame snapshot
    processed_path = _save_processed(df, CFG.symbol, stamp)

    # Save model
    joblib.dump(model, run_dir / "model.joblib")

    meta = {
        "symbol": CFG.symbol,
        "start_date": CFG.start_date,
        "end_date": CFG.end_date,
        "timestamp": stamp,
        "raw_saved_to": raw_path,
        "processed_saved_to": processed_path,
        "n_rows_after_features": int(len(df)),
        "split": {
            "train": int(len(train_df)),
            "val": int(len(val_df)),
            "test": int(len(test_df)),
        },
        "features": feats,
        "baseline_val_accuracy": float(base_val_acc),
        "model_val_accuracy": float(val_acc),
    }
    (run_dir / "metrics.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    report = []
    report.append(f"# Run Report — {CFG.symbol} — {stamp}\n")
    report.append("## Notes\n")
    report.append("- Time-based split (no shuffling)\n")
    report.append("- Local learning project; not investment advice.\n")
    report.append("\n## Data Exports\n")
    report.append(f"- Raw data: `{raw_path}`\n")
    report.append(f"- Processed data: `{processed_path}`\n")
    report.append("\n## Validation Metrics\n")
    report.append(f"- Baseline (majority) accuracy: **{base_val_acc:.3f}**\n")
    report.append(f"- Logistic Regression accuracy: **{val_acc:.3f}**\n")
    report.append("\n## Classification Report (VAL)\n")
    report.append("```text\n")
    report.append(classification_report(y_val, val_pred))
    report.append("\n```\n")

    (run_dir / "report.md").write_text("".join(report), encoding="utf-8")

    print(f"Saved run to: {run_dir}")
    print(f"Raw saved to: {raw_path}")
    print(f"Processed saved to: {processed_path}")
    print(f"Baseline VAL acc: {base_val_acc:.3f}")
    print(f"Model    VAL acc: {val_acc:.3f}")


if __name__ == "__main__":
    main()