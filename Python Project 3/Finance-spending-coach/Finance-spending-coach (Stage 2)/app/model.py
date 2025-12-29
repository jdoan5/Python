# app/model.py
from __future__ import annotations

from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_COLUMNS, profile_to_features
from .schemas import SpendingProfile

# ---------------------------------------------------------------------------
# Paths & globals
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
ARTIFACT_DIR = DATA_DIR / "model_artifacts"
MONTHLY_PATH = DATA_DIR / "transactions_monthly.csv"
MODEL_PATH = ARTIFACT_DIR / "model.pkl"

_model: Pipeline | None = None


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_training_data() -> pd.DataFrame:
    """
    Load the aggregated monthly transactions used for training.

    Expected columns include:
      - income
      - housing
      - food
      - transport
      - shopping
      - entertainment
      - other
      - overspend_flag (0/1 target)

    savings_rate is optional and will be clamped in _row_to_profile.
    """
    if not MONTHLY_PATH.exists():
        raise FileNotFoundError(
            f"Missing {MONTHLY_PATH}. Run scripts/generate_fake_transactions.py first."
        )

    df = pd.read_csv(MONTHLY_PATH)

    required = {
        "income",
        "housing",
        "food",
        "transport",
        "shopping",
        "entertainment",
        "other",
        "overspend_flag",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"transactions_monthly.csv is missing columns: {missing}")

    return df


# ---------------------------------------------------------------------------
# Helpers for turning rows into profiles
# ---------------------------------------------------------------------------

def _row_to_profile(row: pd.Series) -> SpendingProfile:
    """
    Convert a training DataFrame row into a validated SpendingProfile.

    - Reads required numeric fields from the row
    - Handles missing savings_rate by defaulting to 0.0
    - Clamps savings_rate into [0.0, 1.0] to satisfy Pydantic constraints
    """
    income = float(row["income"])
    housing = float(row["housing"])
    food = float(row["food"])
    transport = float(row["transport"])
    shopping = float(row["shopping"])
    entertainment = float(row["entertainment"])
    other = float(row["other"])

    # Some generated rows can have slightly negative savings_rate based on noise,
    # so we clamp into [0.0, 1.0] before constructing the Pydantic model.
    raw_savings = float(row.get("savings_rate", 0.0))
    savings_rate = max(0.0, min(1.0, raw_savings))

    return SpendingProfile(
        income=income,
        housing=housing,
        food=food,
        transport=transport,
        shopping=shopping,
        entertainment=entertainment,
        other=other,
        savings_rate=savings_rate,
    )


# ---------------------------------------------------------------------------
# Training & model persistence
# ---------------------------------------------------------------------------

def train_and_save_model() -> Pipeline:
    """
    Train a simple overspending classifier and persist it to model_artifacts/model.pkl.

    Steps:
      1. Load monthly aggregated data.
      2. Convert each row → SpendingProfile (with clamped savings_rate).
      3. Convert profiles → feature matrix using profile_to_features.
      4. Train a LogisticRegression with a StandardScaler.
      5. Save the fitted pipeline to MODEL_PATH.
    """
    df = _load_training_data()

    # Derive features using the same logic as the API
    profiles = [_row_to_profile(row) for _, row in df.iterrows()]
    X = pd.concat([profile_to_features(p) for p in profiles], ignore_index=True)
    y = df["overspend_flag"].astype(int).values

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )
    pipeline.fit(X, y)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    return pipeline


def load_model() -> Pipeline:
    """
    Lazy-load the model, training it first if no valid artifact is present.

    On import:
      - If model.pkl exists and loads successfully, reuse it.
      - If loading fails (corrupt/incompatible), retrain and overwrite.
      - If no artifact exists, train from scratch.
    """
    global _model
    if _model is not None:
        return _model

    if MODEL_PATH.exists():
        try:
            _model = joblib.load(MODEL_PATH)
            return _model
        except Exception as exc:
            # Fallback: corrupt / incompatible artifact, retrain
            print(f"[model] Warning: failed to load {MODEL_PATH}: {exc}. Re-training model...")
            _model = train_and_save_model()
            return _model

    # No artifact – train from scratch
    _model = train_and_save_model()
    return _model


# ---------------------------------------------------------------------------
# Scoring + suggestions
# ---------------------------------------------------------------------------

def score_profile(profile: SpendingProfile) -> float:
    """
    Return overspend probability in [0, 1] for a single profile.
    """
    model = load_model()
    X = profile_to_features(profile)
    proba = model.predict_proba(X)[0, 1]
    return float(proba)


def risk_level_for_probability(p: float) -> str:
    """
    Map probability of overspending into a coarse risk bucket.
    """
    if p < 0.25:
        return "low"
    if p < 0.6:
        return "medium"
    return "high"


def suggestions_for_profile(p: float, profile: SpendingProfile) -> List[str]:
    """
    Generate a few simple rule-based suggestions given probability and profile.

    This is intentionally lightweight and explainable; it is not model-derived
    but uses simple heuristics on variable spending and savings_rate.
    """
    suggestions: List[str] = []

    income = max(profile.income, 1.0)
    variable_spend = (
        profile.food
        + profile.transport
        + profile.shopping
        + profile.entertainment
        + profile.other
    )
    variable_ratio = variable_spend / income

    if p < 0.25:
        suggestions.append("You are on track; keep saving consistently each month.")
        if profile.savings_rate < 0.15:
            suggestions.append("Consider nudging savings towards ~15–20% of income.")
    elif p < 0.6:
        suggestions.append("Your spending is a bit tight; review non-essential categories.")
        if variable_ratio > 0.4:
            suggestions.append("Try trimming 5–10% from shopping or entertainment.")
        if profile.savings_rate < 0.1:
            suggestions.append("Aim to build at least a 10% savings buffer.")
    else:
        suggestions.append("High risk of overspending this month.")
        suggestions.append(
            "Set a hard cap on discretionary spending (shopping / entertainment)."
        )
        if profile.savings_rate <= 0:
            suggestions.append("You are not saving at all—consider a no-spend week.")
        if variable_ratio > 0.5:
            suggestions.append(
                "More than half of your income is going to variable categories; "
                "look for recurring subscriptions or habits to cut."
            )

    return suggestions