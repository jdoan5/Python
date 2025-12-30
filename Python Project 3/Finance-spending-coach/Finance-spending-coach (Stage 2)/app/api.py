# app/api.py
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from .model import (
    load_model,
    risk_level_for_probability,
    score_profile,
    suggestions_for_profile,
)
from .schemas import (
    SpendingProfile,
    ScoreResponse,
    CoachResponse,
    HealthResponse,
)
from .rag import coach_on_profile

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(
    title="Finance Spending Coach",
    version="2.0.0",
    description="Toy ML service that scores spending and (Stage 2) can add an AI coach.",
)

# Serve the front-end
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.on_event("startup")
def _warm_model() -> None:
    # Ensure model & artifacts are ready at startup
    load_model()


@app.get("/", response_class=HTMLResponse)
def read_index() -> str:
    index_path = STATIC_DIR / "index.html"
    return index_path.read_text(encoding="utf-8")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/score_profile", response_model=ScoreResponse)
def score_profile_endpoint(profile: SpendingProfile) -> ScoreResponse:
    proba = score_profile(profile)
    risk = risk_level_for_probability(proba)
    suggestions = suggestions_for_profile(proba, profile)
    return ScoreResponse(
        probability=proba,
        risk_level=risk,
        suggestions=suggestions,
    )


@app.post("/coach_profile", response_model=CoachResponse)
def coach_profile_endpoint(profile: SpendingProfile) -> CoachResponse:
    """
    Stage 2 endpoint: wraps the base model score with an AI coach explanation.
    """
    base = score_profile_endpoint(profile)  # reuse same logic
    return coach_on_profile(profile, base)