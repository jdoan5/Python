import json
from typing import List

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from customer_metrics.config import MODEL_PATH, FEATURE_COLUMNS_JSON


class CustomerFeatures(BaseModel):
    age: int = Field(..., ge=0, le=120)
    tenure_months: int = Field(..., ge=0)
    avg_monthly_spend: float = Field(..., ge=0)
    num_support_tickets: int = Field(..., ge=0)
    is_premium: int = Field(..., ge=0, le=1)


class ScoreResponse(BaseModel):
    churn_probability: float
    churn_label: int
    feature_order: List[str]


app = FastAPI(title="Customer Metrics Pipeline & API")

_model = None
_feature_columns: List[str] = []


def load_artifacts() -> None:
    global _model, _feature_columns

    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model file not found at {MODEL_PATH}. "
            "Run the training pipeline first (python run_pipeline.py)."
        )

    _model = joblib.load(MODEL_PATH)

    try:
        with FEATURE_COLUMNS_JSON.open() as f:
            meta = json.load(f)
        _feature_columns = meta["feature_columns"]
    except Exception as exc:
        raise RuntimeError(f"Error loading feature metadata: {exc}") from exc


@app.on_event("startup")
def startup_event():
    load_artifacts()


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.post("/score_customer", response_model=ScoreResponse)
def score_customer(customer: CustomerFeatures):
    if _model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # Build a single-row DataFrame with the expected column order
    data = {col: getattr(customer, col) for col in _feature_columns}
    X = pd.DataFrame([data])

    proba = float(_model.predict_proba(X)[0, 1])
    label = int(proba >= 0.5)

    return ScoreResponse(
        churn_probability=round(proba, 4),
        churn_label=label,
        feature_order=_feature_columns,
    )