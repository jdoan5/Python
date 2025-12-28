# customer_metrics/pipeline/train_model.py
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import joblib

from customer_metrics.config import (
    DATA_DIR,
    ARTIFACTS_DIR,
    MODEL_PATH,
    FEATURE_COLUMNS_JSON,
    FEATURE_COLUMNS,
)


RANDOM_SEED = 42
N_SAMPLES = 5000


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate_synthetic_customers(path: Path, n_samples: int = N_SAMPLES) -> None:
    """
    Generate a synthetic customer-level dataset with churn labels.
    """
    rng = np.random.default_rng(RANDOM_SEED)

    age = np.clip(rng.normal(loc=40, scale=12, size=n_samples).round().astype(int), 18, 80)
    tenure_months = rng.integers(0, 120, size=n_samples)  # 0–10 years
    avg_monthly_spend = np.clip(
        rng.normal(loc=120.0, scale=60.0, size=n_samples), 10.0, None
    )
    num_support_tickets = rng.poisson(lam=1.5, size=n_samples)
    is_premium = rng.integers(0, 2, size=n_samples)

    # Logistic-style churn probability
    z = (
        -2.0
        + 0.012 * (age - 40)
        - 0.015 * tenure_months          # longer tenure → less likely to churn
        - 0.004 * avg_monthly_spend      # higher spend → less likely to churn
        + 0.35 * num_support_tickets     # more tickets → more likely to churn
        - 0.9 * is_premium               # premium → less likely to churn
    )
    p_churn = _sigmoid(z)

    churn = rng.binomial(1, p_churn)

    df = pd.DataFrame(
        {
            "age": age,
            "tenure_months": tenure_months,
            "avg_monthly_spend": avg_monthly_spend,
            "num_support_tickets": num_support_tickets,
            "is_premium": is_premium,
            "churn": churn,
        }
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[pipeline] Wrote synthetic data to {path} (shape={df.shape})")


def train_and_save_model(data_path: Path) -> None:
    """
    Train a logistic regression model and persist the artifacts.
    """
    df = pd.read_csv(data_path)

    X = df[FEATURE_COLUMNS]
    y = df["churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("[pipeline] Classification report (test set):")
    print(classification_report(y_test, y_pred, digits=3))

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"[pipeline] Saved model to {MODEL_PATH}")

    meta = {"feature_columns": FEATURE_COLUMNS}
    with FEATURE_COLUMNS_JSON.open("w") as f:
        json.dump(meta, f, indent=2)
    print(f"[pipeline] Saved feature metadata to {FEATURE_COLUMNS_JSON}")


def run_pipeline() -> None:
    csv_path = DATA_DIR / "customers_synthetic.csv"
    generate_synthetic_customers(csv_path, n_samples=N_SAMPLES)
    train_and_save_model(csv_path)


if __name__ == "__main__":
    run_pipeline()