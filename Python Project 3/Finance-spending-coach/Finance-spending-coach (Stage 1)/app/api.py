# app/api.py  (Stage 1)

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from .model import (
    load_model,
    score_profile,
    risk_level_for_probability,
    suggestions_for_profile,
)
from .schemas import SpendingProfile, ScoreResponse

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(
    title="Finance Spending Coach",
    version="0.1.0",
    description=(
        "Toy ML service that scores a monthly spending profile for overspending "
        "risk and returns friendly coaching suggestions."
    ),
)

# Serve index.html + any static assets
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.on_event("startup")
def _warm_model() -> None:
    # Load or train the model once at startup
    load_model()


# ---- UI entry point (not shown in Swagger) ----------------------------------


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def home() -> HTMLResponse:
    """
    Simple HTML UI that lets a user fill in their spending profile
    and calls /score_profile under the hood.
    """
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(html)


# ---- Meta / health ----------------------------------------------------------


@app.get("/health", tags=["meta"])
def health() -> dict:
    """Lightweight health check."""
    return {"status": "ok"}


# ---- Scoring endpoint -------------------------------------------------------


@app.post("/score_profile", response_model=ScoreResponse, tags=["scoring"])
def score_profile_endpoint(profile: SpendingProfile) -> ScoreResponse:
    """
    Score a monthly spending profile for overspending risk.
    """
    p = score_profile(profile)
    level = risk_level_for_probability(p)
    suggs = suggestions_for_profile(p, profile)

    return ScoreResponse(
        overspend_probability=p,
        risk_level=level,
        suggestions=suggs,
    )