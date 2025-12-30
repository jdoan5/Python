# app/features.py
from __future__ import annotations

from typing import Dict, Any

import pandas as pd

from .schemas import SpendingProfile

# Columns used for training / inference
FEATURE_COLUMNS = [
    "income",
    "housing",
    "food",
    "transport",
    "shopping",
    "entertainment",
    "other",
    "savings_rate",
    "variable_ratio",     # derived: variable spend / income
]


def profile_to_features(profile: SpendingProfile) -> pd.DataFrame:
    """
    Convert a SpendingProfile into a single-row DataFrame matching FEATURE_COLUMNS.
    """
    # Base numeric fields (coerce to float)
    base = {
        "income": float(profile.income),
        "housing": float(profile.housing),
        "food": float(profile.food),
        "transport": float(profile.transport),
        "shopping": float(profile.shopping),
        "entertainment": float(profile.entertainment),
        "other": float(profile.other),
        "savings_rate": float(profile.savings_rate),
    }

    income = max(base["income"], 1.0)
    variable_spend = (
        base["food"]
        + base["transport"]
        + base["shopping"]
        + base["entertainment"]
        + base["other"]
    )
    base["variable_ratio"] = variable_spend / income

    return pd.DataFrame([{col: base[col] for col in FEATURE_COLUMNS}])


def row_to_profile_dict(row: Dict[str, Any]) -> SpendingProfile:
    """
    Helper: turn a CSV row (from data/transactions_monthly.csv) into a SpendingProfile-like dict.
    Not used by the API directly, but handy for debugging / notebooks.
    """
    return SpendingProfile(
        income=float(row["income"]),
        housing=float(row["housing"]),
        food=float(row["food"]),
        transport=float(row["transport"]),
        shopping=float(row["shopping"]),
        entertainment=float(row["entertainment"]),
        other=float(row["other"]),
        savings_rate=float(row.get("savings_rate", 0.0)),
    )