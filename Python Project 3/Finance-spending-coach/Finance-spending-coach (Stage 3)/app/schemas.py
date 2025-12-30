# app/schemas.py
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class SpendingProfile(BaseModel):
    income: float = Field(..., ge=0)
    housing: float = Field(..., ge=0)
    food: float = Field(..., ge=0)
    transport: float = Field(..., ge=0)
    shopping: float = Field(..., ge=0)
    entertainment: float = Field(..., ge=0)
    other: float = Field(..., ge=0)
    # 0–1 (e.g. 0.10 == 10%)
    savings_rate: float = Field(0.0, ge=0.0, le=1.0)


class ScoreResponse(BaseModel):
    overspend_probability: float = Field(..., ge=0.0, le=1.0)
    risk_level: str
    # For the simple endpoint we just use `suggestions`.
    suggestions: Optional[List[str]] = None
    # For the coach endpoint we may also set `model_suggestions`.
    model_suggestions: Optional[List[str]] = None


class CoachRequest(BaseModel):
    profile: SpendingProfile
    question: Optional[str] = None


class CoachResponse(BaseModel):
    overspend_probability: float = Field(..., ge=0.0, le=1.0)
    risk_level: str

    # Numeric-model suggestions we computed from the profile
    model_suggestions: List[str] = Field(default_factory=list)

    # Human-readable answer text (multiple paragraphs)
    answer: str

    # NEW: grounded bullets from the local KB (what the UI shows
    # under "Grounded tips from KB:")
    kb_tips: List[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str = "ok"