# app/agent.py
from __future__ import annotations

from typing import List

from .model import score_profile, risk_level_for_probability
from .rag import generate_coach_message
from .schemas import CoachChatRequest, CoachChatResponse


def run_agent(request: CoachChatRequest) -> CoachChatResponse:
    """
    Very small "agent": it scores the profile, calls RAG for grounded tips,
    and wraps everything in a single response object.
    """
    proba = score_profile(request.profile)
    risk_level = risk_level_for_probability(proba)

    coach_message, sources = generate_coach_message(
        profile=request.profile,
        risk_level=risk_level,
        probability=proba,
        question=request.question,
        top_k=4,
    )

    return CoachChatResponse(
        answer=coach_message,
        probability=proba,
        risk_level=risk_level,
        sources=sources,
    )