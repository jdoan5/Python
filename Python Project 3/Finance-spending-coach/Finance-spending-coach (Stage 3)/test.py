# app/rag.py
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from .schemas import SpendingProfile, CoachResponse

BASE_DIR = Path(__file__).resolve().parent
KB_DIR = BASE_DIR / "kb"


@dataclass
class KBEntry:
    path: Path
    title: str
    tips: List[str]
    tokens: set[str]


_kb_index: List[KBEntry] | None = None


# ------------------------- KB loading & parsing ---------------------------------


def _split_into_tips(text: str) -> List[str]:
    """
    Extract short, tip-like lines from a markdown file.

    - Prefer bullet lines starting with '-' or '*'
    - Fall back to the first 2–3 sentences if no bullets are found
    """
    tips: List[str] = []

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        m = re.match(r"^[-*]\s+(.*)", stripped)
        if m:
            tips.append(m.group(1).strip())
            continue

        # numbered list, e.g. "1. Build an emergency fund"
        m = re.match(r"^\d+\.\s+(.*)", stripped)
        if m:
            tips.append(m.group(1).strip())

    if not tips:
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        tips = [s.strip() for s in sentences[:3] if s.strip()]

    return tips


def _tokenize(text: str) -> set[str]:
    # Simple word tokenizer, ignore very short tokens
    tokens = re.findall(r"[a-zA-Z]{3,}", text.lower())
    return set(tokens)


def _load_kb() -> List[KBEntry]:
    entries: List[KBEntry] = []

    if not KB_DIR.exists():
        return entries

    for path in sorted(KB_DIR.glob("*.md")):
        raw = path.read_text(encoding="utf-8", errors="ignore")

        # Title: first non-empty heading, otherwise file stem
        title = path.stem.replace("_", " ")
        for line in raw.splitlines():
            line = line.strip()
            if line.startswith("#"):
                candidate = line.lstrip("#").strip()
                if candidate:
                    title = candidate
                break

        tips = _split_into_tips(raw)
        tokens = _tokenize(raw)
        entries.append(KBEntry(path=path, title=title, tips=tips, tokens=tokens))

    return entries


def ensure_kb_index() -> None:
    """Load KB index into memory once."""
    global _kb_index
    if _kb_index is None:
        _kb_index = _load_kb()


# ------------------------------ Retrieval ---------------------------------------


def _score_query(query: str, entry: KBEntry) -> float:
    """
    Very small, dependency-free similarity score:
    scaled overlap between query tokens and document tokens.
    """
    if not query:
        return 0.0

    q_tokens = _tokenize(query)
    if not q_tokens:
        return 0.0

    overlap = q_tokens & entry.tokens
    if not overlap:
        return 0.0

    return float(len(overlap) / math.sqrt(len(entry.tokens) * len(q_tokens)))


def _retrieve_kb_tips(
    profile: SpendingProfile, question: str | None, top_k: int = 3
) -> List[str]:
    """
    Retrieve a few short tips from the KB, based on the profile and question.
    """
    ensure_kb_index()
    if not _kb_index:
        return []

    query_parts = [
        f"income {profile.income}",
        f"housing {profile.housing}",
        f"food {profile.food}",
        f"transport {profile.transport}",
        f"shopping {profile.shopping}",
        f"entertainment {profile.entertainment}",
        f"savings rate {profile.savings_rate * 100:.0f} percent",
    ]
    if question:
        query_parts.append(question)

    query = " ".join(query_parts)

    scored: List[Tuple[float, KBEntry]] = [
        (_score_query(query, entry), entry) for entry in (_kb_index or [])
    ]
    scored = [item for item in scored if item[0] > 0]
    scored.sort(key=lambda x: x[0], reverse=True)

    tips: List[str] = []
    for _, entry in scored[:top_k]:
        for tip in entry.tips:
            tips.append(tip)
            if len(tips) >= 5:
                break
        if len(tips) >= 5:
            break

    return tips


# -------------------------- Coach response builder ------------------------------


def generate_coach_response(
    profile: SpendingProfile,
    probability: float,
    risk_level: str,
    base_suggestions: List[str],
    question: str | None = None,
) -> CoachResponse:
    """
    Build a friendly response + KB tips.

    The `answer` field is structured with double newlines between logical
    paragraphs so the front end can split on `\\n\\n` and render each paragraph
    separately. In particular, **"You asked ..."** is always its own paragraph.
    """
    p_pct = probability * 100.0

    # Paragraph 1: summary of the profile and numeric score
    profile_summary = (
        f"For this profile (income={profile.income:.1f}, housing={profile.housing:.1f}, "
        f"food={profile.food:.1f}, transport={profile.transport:.1f}, "
        f"shopping={profile.shopping:.1f}, entertainment={profile.entertainment:.1f}, "
        f"other={profile.other:.1f}, savings_rate={profile.savings_rate:.2f}), "
        f"the estimated overspending probability is about {p_pct:.1f}% "
        f"({risk_level.upper()} risk)."
    )

    # Paragraph 2: short interpretation of the risk level
    if risk_level == "low":
        risk_comment = (
            "Overall, this suggests low risk of overspending. Nice work – keep your habits steady."
        )
    elif risk_level == "medium":
        risk_comment = (
            "Overall, this suggests a moderate risk of overspending. Tightening one or two "
            "non-essential categories this month could help."
        )
    else:
        risk_comment = (
            "Overall, this suggests high risk of overspending. It may be worth tightening "
            "non-essential categories and watching cash flow more closely this month."
        )

    # Paragraph 3: echo the user's question (if any)
    question_text = f'You asked: "{question}".' if question else ""

    # Paragraph 4: model-based suggestions, rendered as bullet lines (the front end
    # keeps the line breaks).
    if base_suggestions:
        model_sugg_text = "Model-based suggestions:\n- " + "\n- ".join(base_suggestions)
    else:
        model_sugg_text = ""

    # KB tips (only listed in the bullet list, not inline in the paragraph)
    kb_tips = _retrieve_kb_tips(profile, question)

    kb_intro = ""
    if kb_tips:
        kb_intro = (
            "Grounded tips from the budgeting knowledge base are listed below. "
            "They are general best practices, not personalized financial advice."
        )

    paragraphs = [
        profile_summary,
        risk_comment,
        question_text,
        model_sugg_text,
        kb_intro,
    ]
    answer = "\n\n".join(p for p in paragraphs if p)

    return CoachResponse(
        overspend_probability=probability,
        risk_level=risk_level,
        model_suggestions=base_suggestions,
        answer=answer,
        kb_tips=kb_tips,
    )