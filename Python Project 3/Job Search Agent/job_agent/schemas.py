"""Pydantic schemas — the structured-output contracts between pipeline stages.

Every LLM call in the pipeline returns one of these models via
`client.messages.parse(...)`, so downstream code never string-parses model
output. This is the "structured output" pattern interviewers ask about.
"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class Requirement(BaseModel):
    """One requirement extracted from a job posting."""

    text: str = Field(description="The requirement as stated in the posting, condensed.")
    kind: Literal["must_have", "nice_to_have", "responsibility"]
    keywords: List[str] = Field(
        default_factory=list,
        description="Short search keywords for matching against the candidate's experience.",
    )


class JobRequirements(BaseModel):
    """Stage 1 output: what the posting is actually asking for."""

    company: str = Field(description="Company name, or 'Unknown' if not stated.")
    role: str = Field(description="Job title, or 'Unknown' if not stated.")
    seniority: Literal["entry", "mid", "senior", "staff+", "unknown"] = Field(
        description="Infer from years/scope; use 'unknown' if unstated."
    )
    requirements: List[Requirement]
    red_flags: List[str] = Field(
        default_factory=list,
        description="Vague scope, unrealistic stacks, missing salary — anything worth noting.",
    )


class EvidenceMatch(BaseModel):
    """One requirement matched against retrieved experience evidence."""

    requirement: str = Field(description="The requirement text being matched.")
    evidence_ids: List[str] = Field(
        default_factory=list,
        description="IDs of inventory entries that support this requirement.",
    )
    strength: Literal["strong", "partial", "none"] = Field(
        description="Judged only from retrieved evidence."
    )
    note: str = Field(
        default="",
        description="One sentence on why the evidence does or does not support the requirement.",
    )


class TailoredBullet(BaseModel):
    """A resume bullet drafted for this specific posting."""

    text: str = Field(description="The drafted bullet, <= 30 words, starts with a strong verb.")
    evidence_ids: List[str] = Field(
        description="Inventory entries this bullet is grounded in. Never empty — "
        "a bullet with no evidence is fabrication."
    )
    targets_requirement: str = Field(
        description="The posting requirement this bullet speaks to."
    )


class TailoredDraft(BaseModel):
    """Stage 3 output: the full draft awaiting human approval."""

    fit_score: int = Field(ge=1, le=10, description="Honest 1-10 fit against must-haves.")
    fit_rationale: str = Field(description="2-3 sentences justifying the score.")
    matches: List[EvidenceMatch]
    bullets: List[TailoredBullet]
    gaps: List[str] = Field(
        default_factory=list,
        description="Must-have requirements with no supporting evidence. Be honest.",
    )
    cover_note: Optional[str] = Field(
        default=None,
        description="Optional 3-4 sentence opener referencing specific evidence.",
    )
