# app/schemas.py
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, confloat


# ---------------------------------------------------------------------------
# Core input schema
# ---------------------------------------------------------------------------

class SpendingProfile(BaseModel):
    """
    Monthly spending profile.

    NOTE: savings_rate is a fraction in [0, 1] (e.g. 0.15 == 15% of income),
    not a percentage. The front-end UI converts from 0–100% to a fraction
    before sending.
    """

    income: float = Field(..., ge=0, description="Monthly income after tax.")
    housing: float = Field(..., ge=0, description="Housing / rent / mortgage.")
    food: float = Field(..., ge=0, description="Food & groceries.")
    transport: float = Field(..., ge=0, description="Transport, gas, public transit.")
    shopping: float = Field(..., ge=0, description="Shopping & discretionary spend.")
    entertainment: float = Field(..., ge=0, description="Entertainment, eating out, etc.")
    other: float = Field(
        0.0, ge=0, description="Other variable expenses not covered above."
    )
    savings_rate: Optional[confloat(ge=0, le=1)] = Field(
        None,
        description=(
            "Savings rate as a fraction in [0, 1]. "
            "For example, 0.15 == 15% of income saved."
        ),
    )


# ---------------------------------------------------------------------------
# Stage 1 – model-only response
# ---------------------------------------------------------------------------

class ScoreResponse(BaseModel):
    probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probability of overspending this month.",
    )
    risk_level: Literal["low", "medium", "high"] = Field(
        ...,
        description="Discrete risk bucket derived from the probability.",
    )
    suggestions: List[str] = Field(
        default_factory=list,
        description="Simple rule-based suggestions based on the profile.",
    )


# ---------------------------------------------------------------------------
# Stage 2 – coach / agent response
# ---------------------------------------------------------------------------

class CoachResponse(BaseModel):
    probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probability of overspending this month.",
    )
    risk_level: Literal["low", "medium", "high"] = Field(
        ...,
        description="Discrete risk bucket derived from the probability.",
    )
    suggestions: List[str] = Field(
        default_factory=list,
        description="Bullet-point suggestions (same as ScoreResponse).",
    )
    coach_message: str = Field(
        ...,
        description="Human-readable explanation / coaching summary (Stage 2).",
    )


# ---------------------------------------------------------------------------
# Meta
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str = Field(
        "ok",
        description="Simple health status string.",
    )