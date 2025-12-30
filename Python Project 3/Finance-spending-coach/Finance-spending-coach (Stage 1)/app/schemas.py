# app/schemas.py  (Stage 1)

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class SpendingProfile(BaseModel):
    """
    Simple monthly spending profile.

    savings_rate is stored as a FRACTION (0–1), e.g.
    0.10 == 10% of income saved.
    """

    income: float = Field(..., ge=0, description="Monthly take-home income.")
    housing: float = Field(..., ge=0)
    food: float = Field(..., ge=0, description="Food & groceries.")
    transport: float = Field(..., ge=0)
    shopping: float = Field(..., ge=0)
    entertainment: float = Field(..., ge=0)
    other: float = Field(..., ge=0)
    savings_rate: float = Field(
        ...,
        ge=0.0,
        le=0.8,
        description="Fraction of income saved (0–1). Example: 0.10 == 10%.",
    )


class ScoreResponse(BaseModel):
    """
    Response for /score_profile in Stage 1.

    message is OPTIONAL so api.py does not have to set it.
    """

    overspend_probability: float = Field(..., ge=0.0, le=1.0)
    risk_level: str
    suggestions: List[str]
    message: Optional[str] = None  # <— make this optional with a default