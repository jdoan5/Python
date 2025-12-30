# app/api.py
from __future__ import annotations

from pathlib import Path
from typing import List

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .model import (
    load_model,
    risk_level_for_probability,
    score_profile,
    suggestions_for_profile,
)
from .rag import ensure_kb_index, generate_coach_response
from .schemas import (
    CoachRequest,
    CoachResponse,
    HealthResponse,
    ScoreResponse,
    SpendingProfile,
)

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(
    title="Finance Spending Coach",
    version="0.3.0",
    description=(
        "Toy ML service that scores spending and (Stage 3) adds a KB-backed "
        "coach layer on top of the score."
    ),
)

# Serve index.html + static assets
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def read_index():
    return FileResponse(STATIC_DIR / "index.html")


@app.on_event("startup")
def _startup():
    # Warm model + KB index
    load_model()
    ensure_kb_index()


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/score_profile", response_model=ScoreResponse, tags=["model"])
def score_endpoint(profile: SpendingProfile) -> ScoreResponse:
    p = score_profile(profile)
    level = risk_level_for_probability(p)
    suggs = suggestions_for_profile(p, profile)
    return ScoreResponse(
        overspend_probability=p,
        risk_level=level,
        suggestions=suggs,
    )


@app.post("/coach_profile", response_model=CoachResponse, tags=["coach"])
def coach_endpoint(req: CoachRequest) -> CoachResponse:
    profile = req.profile
    question = req.question

    p = score_profile(profile)
    level = risk_level_for_probability(p)
    base_suggestions = suggestions_for_profile(p, profile)

    return generate_coach_response(
        profile=profile,
        probability=p,
        risk_level=level,
        base_suggestions=base_suggestions,
        question=question,
    )