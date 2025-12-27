# src/train.py
from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import CFG
from src.data_fetch import fetch_from_yahoo, save_raw
from src.features import add_features, feature_columns


def _safe_name(s: str) -> str:
    # file-safe symbol (e.g., BRK.B -> BRK_B)
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in s)


def _json_safe(obj: Any) -> Any:
    """
    Recursively convert non-JSON-serializable objects (e.g., Path, numpy types)
    into JSON-safe primitives.
    """
    if isinstance(obj, Path):
        return str(obj)

    # numpy scalars
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)

    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]

    return obj


def time_split(df: pd.DataFrame, train_ratio: float, val_ratio: float):
    n = len(df)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train = df.iloc[:n_train].copy()
    val = df.iloc[n_train : n_train + n_val].copy()
    test = df.iloc[n_train + n_val :].copy()
    return train, val, test


def baseline_predict_majority(y_train: pd.Series, y_eval: pd.Series) -> np.ndarray:
    majority = int(y_train.mean() >= 0.5)  # if tie, choose 1 (Up)
    return np.full(shape=len(y_eval), fill_value=majority, dtype=int)


def _ensure_dirs():
    CFG.runs_dir.mkdir(parents=True, exist_ok=True)
    CFG.data_raw.mkdir(parents=True, exist_ok=True)
    CFG.data_processed.mkdir(parents=True, exist_ok=True)


def _model_candidates(random_state: int) -> List[Tuple[str, Any]]:
    """
    Stage 2: keep Stage 1 baseline (LogReg), plus a couple of stronger non-linear baselines.
    All are scikit-learn only (no extra deps).
    """
    return [
        (
            "logreg",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", LogisticRegression(max_iter=3000, random_state=random_state)),
                ]
            ),
        ),
        (
            "rf",
            RandomForestClassifier(
                n_estimators=400,
                random_state=random_state,
                n_jobs=-1,
                max_depth=None,
                min_samples_leaf=2,
            ),
        ),
        (
            "gboost",
            GradientBoostingClassifier(random_state=random_state),
        ),
    ]


def _score_model(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "f1_up": float(f1_score(y_true, y_pred, pos_label=1)),
        "f1_down": float(f1_score(y_true, y_pred, pos_label=0)),
    }


def _pick_best(results: Dict[str, Dict[str, float]]) -> str:
    """
    Rank by macro-F1 (more robust than accuracy for mild imbalance),
    then by accuracy as a tiebreaker.
    """
    items = list(results.items())
    items.sort(key=lambda kv: (kv[1]["macro_f1"], kv[1]["accuracy"]), reverse=True)
    return items[0][0]


def train_one_symbol(symbol: str) -> Path:
    _ensure_dirs()

    # 1) Fetch raw data and persist a snapshot (useful for reproducibility)
    raw = fetch_from_yahoo(symbol, start=CFG.start_date, end=CFG.end_date)
    raw_path = save_raw(raw, symbol)  # Path (recommended) or str

    # 2) Feature engineering
    df = add_features(raw)
    feats = feature_columns()

    needed_cols = feats + ["target_up"]
    df = df.dropna(subset=needed_cols).reset_index(drop=True)

    if len(df) < 200:
        raise RuntimeError(
            f"[{symbol}] Not enough rows after feature engineering ({len(df)}). "
            "Try a longer date range in config.py."
        )

    # Optional: save a processed snapshot used by the model
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    processed_path = CFG.data_processed / f"{symbol}_{stamp}_features.csv"
    df.to_csv(processed_path, index=False)

    # 3) Time split (no leakage)
    train_df, val_df, test_df = time_split(df, CFG.train_ratio, CFG.val_ratio)

    if len(val_df) == 0 or len(test_df) == 0:
        raise RuntimeError(
            f"[{symbol}] Validation/test split resulted in empty set. "
            "Adjust train_ratio/val_ratio or use more data."
        )

    X_train, y_train = train_df[feats], train_df["target_up"].astype(int)
    X_val, y_val = val_df[feats], val_df["target_up"].astype(int)
    X_test, y_test = test_df[feats], test_df["target_up"].astype(int)

    # 4) Baseline (majority class from TRAIN)
    base_val_pred = baseline_predict_majority(y_train, y_val)
    base_val_scores = _score_model(y_val, base_val_pred)

    # 5) Train candidates and compare on VAL
    candidates = _model_candidates(CFG.random_state)
    val_results: Dict[str, Dict[str, float]] = {}
    fitted: Dict[str, Any] = {}

    for name, model in candidates:
        model.fit(X_train, y_train)
        pred_val = model.predict(X_val)
        val_results[name] = _score_model(y_val, pred_val)
        fitted[name] = model

    best_name = _pick_best(val_results)
    best_model = fitted[best_name]

    # 6) Write run dir + artifacts
    run_dir = CFG.runs_dir / f"run_{_safe_name(symbol)}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_model, run_dir / "model.joblib")

    config_snapshot = asdict(CFG) if hasattr(CFG, "__dataclass_fields__") else None

    meta: Dict[str, Any] = {
        "stage": "stage2",
        "symbol": symbol,
        "start_date": CFG.start_date,
        "end_date": CFG.end_date,
        "timestamp": stamp,
        "raw_data_path": str(Path(raw_path).name),
        "processed_data_path": str(processed_path.name),
        "n_rows_after_features": int(len(df)),
        "split": {"train": int(len(train_df)), "val": int(len(val_df)), "test": int(len(test_df))},
        "features": feats,
        "baseline_val": base_val_scores,
        "candidates_val": val_results,
        "best_model": best_name,
        "notes": [
            "Time-based split (no shuffling).",
            "Stage 2 adds model comparison (LogReg + RF + GradientBoosting).",
            "Local learning project; not investment advice.",
        ],
        "config_snapshot": config_snapshot,
    }

    (run_dir / "metrics.json").write_text(
        json.dumps(_json_safe(meta), indent=2),
        encoding="utf-8",
    )

    # 7) Human-readable report.md (VAL)
    best_val_pred = best_model.predict(X_val)
    report_lines: List[str] = []
    report_lines.append(f"# Run Report — {symbol} — {stamp}\n\n")
    report_lines.append("## Notes\n")
    report_lines.append("- Time-based split (no shuffling)\n")
    report_lines.append("- Stage 2: model comparison; select best by **macro-F1**, then accuracy.\n")
    report_lines.append("- Local learning project; not investment advice.\n\n")

    report_lines.append("## Data Snapshots\n")
    report_lines.append(f"- Raw saved: `{raw_path}`\n")
    report_lines.append(f"- Processed saved: `{processed_path}`\n\n")

    report_lines.append("## Validation Results\n")
    report_lines.append(
        f"- Baseline (majority) — acc: **{base_val_scores['accuracy']:.3f}**, "
        f"macro-F1: **{base_val_scores['macro_f1']:.3f}**\n"
    )
    report_lines.append("\n### Candidate Models (VAL)\n")
    report_lines.append("| model | acc | macro-F1 | F1(up) | F1(down) |\n")
    report_lines.append("|---|---:|---:|---:|---:|\n")
    for name, s in sorted(val_results.items(), key=lambda kv: (kv[1]["macro_f1"], kv[1]["accuracy"]), reverse=True):
        report_lines.append(
            f"| `{name}` | {s['accuracy']:.3f} | {s['macro_f1']:.3f} | {s['f1_up']:.3f} | {s['f1_down']:.3f} |\n"
        )

    report_lines.append(f"\n**Selected best model:** `{best_name}`\n\n")
    report_lines.append("## Classification Report (VAL) — Best Model\n")
    report_lines.append("```text\n")
    report_lines.append(classification_report(y_val, best_val_pred))
    report_lines.append("```\n")

    (run_dir / "report.md").write_text("".join(report_lines), encoding="utf-8")

    print(f"[{symbol}] Saved run to: {run_dir}")
    print(f"[{symbol}] Baseline VAL acc: {base_val_scores['accuracy']:.3f} (macro-F1: {base_val_scores['macro_f1']:.3f})")
    best_scores = val_results[best_name]
    print(f"[{symbol}] Best({best_name}) VAL acc: {best_scores['accuracy']:.3f} (macro-F1: {best_scores['macro_f1']:.3f})")
    return run_dir


def main():
    if getattr(CFG, "symbols", None):
        symbols = [str(s).strip() for s in CFG.symbols if str(s).strip()]
    else:
        symbols = [CFG.symbol]

    for sym in symbols:
        train_one_symbol(sym)


if __name__ == "__main__":
    main()