# app/api.py
from fastapi import FastAPI
from pathlib import Path
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from .model import (
    load_model,
    risk_level_for_probability,
    score_profile,
    suggestions_for_profile,
)
from .schemas import HealthResponse, ScoreResponse, SpendingProfile

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(
    title="Finance Spending Coach",
    version="0.1.0",
    description="Toy ML service that scores a monthly spending profile "
                "for overspending risk and returns friendly coaching suggestions.",
)

# Serve static UI files
STATIC_DIR = BASE_DIR / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.on_event("startup")
def _warm_model() -> None:
    """
    Ensure the model is trained/loaded on startup so the first request is fast.
    """
    load_model()


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def home():
    """
    Simple HTML UI that lets a user fill in their spending profile
    and calls /score_profile under the hood.
    """
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(html)


@app.post("/score_profile", response_model=ScoreResponse, tags=["scoring"])
def score_spending_profile(profile: SpendingProfile) -> ScoreResponse:
    """
    Score a single month of spending.

    Example JSON body:

    {
      "income": 5000,
      "housing": 1800,
      "food": 600,
      "transport": 250,
      "shopping": 400,
      "entertainment": 300,
      "other": 200,
      "savings_rate": 0.1
    }
    """
    prob = score_profile(profile)
    level = risk_level_for_probability(prob)
    suggestions = suggestions_for_profile(prob, profile)

    message = f"Estimated overspending probability: {prob:.0%} ({level} risk)."

    return ScoreResponse(
        overspend_probability=prob,
        risk_level=level,
        message=message,
        suggestions=suggestions,
    )