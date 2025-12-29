# app/schemas.py
from typing import List
from pydantic import BaseModel, Field, confloat


class SpendingProfile(BaseModel):
    """
    One month of spending for a single person/household.

    These fields roughly match the columns in data/transactions_monthly.csv.
    """
    income: float = Field(..., gt=0, description="Net monthly income")
    housing: float = Field(..., ge=0, description="Rent/mortgage + utilities")
    food: float = Field(..., ge=0, description="Groceries + eating out")
    transport: float = Field(..., ge=0, description="Public transport, fuel, etc.")
    shopping: float = Field(..., ge=0, description="Shopping / retail")
    entertainment: float = Field(..., ge=0, description="Entertainment & leisure")
    other: float = Field(..., ge=0, description="Other variable spending")

    savings_rate: confloat(ge=0, le=1) = Field(
        ...,
        description="Savings as a fraction of income, e.g. 0.2 for 20%. "
                    "The backend will sanity-check this against income and spending.",
    )


class ScoreResponse(BaseModel):
    overspend_probability: float = Field(..., ge=0, le=1)
    risk_level: str
    message: str
    suggestions: List[str]


class HealthResponse(BaseModel):
    status: str