# app/rag.py
from __future__ import annotations

from .schemas import SpendingProfile, ScoreResponse, CoachResponse


# In a true RAG setup, you'd load notes / examples / documents here and
# pass them + the profile into an LLM. For the portfolio version we keep
# it simple and template-based, but the API surface is the same.
def coach_on_profile(profile: SpendingProfile, base: ScoreResponse) -> CoachResponse:
    risk = base.risk_level
    prob = base.probability
    suggestions = base.suggestions

    if risk == "low":
        coach_message = (
            "Your spending pattern looks sustainable. "
            "Keep your current habits and review recurring expenses every few months. "
            "You could slowly increase your savings rate if it feels comfortable."
        )
    elif risk == "medium":
        coach_message = (
            "You are close to overspending. Focus on trimming a few non-essential "
            "categories (shopping, entertainment) and consider automating a fixed "
            "transfer into savings at the start of each month."
        )
    else:  # high
        coach_message = (
            "Your profile shows a high risk of overspending. Start by freezing big "
            "discretionary spend (shopping / entertainment) for a couple of weeks. "
            "Create simple caps per category and build a small emergency buffer "
            "before taking on new subscriptions or large purchases."
        )

    return CoachResponse(
        probability=prob,
        risk_level=risk,
        suggestions=suggestions,
        coach_message=coach_message,
    )