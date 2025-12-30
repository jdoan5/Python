# app/rag.py
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .schemas import CoachResponse, SpendingProfile

BASE_DIR = Path(__file__).resolve().parent
KB_DIR = BASE_DIR / "kb"

# In-memory KB index
_vectorizer: TfidfVectorizer | None = None
_doc_texts: List[str] = []
_doc_titles: List[str] = []
_doc_matrix = None  # scipy.sparse matrix or None


def ensure_kb_index() -> None:
    """
    Load all Markdown files from app/kb and build a simple TF-IDF index.

    This is called once on startup from api.py (_startup), and can also be
    called lazily from generate_coach_response.
    """
    global _vectorizer, _doc_texts, _doc_titles, _doc_matrix

    if _vectorizer is not None:
        return

    kb_files = sorted(KB_DIR.glob("*.md"))
    texts: List[str] = []
    titles: List[str] = []

    for path in kb_files:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        texts.append(text)
        titles.append(path.name)

    if not texts:
        # No KB files – just initialize empty state
        _vectorizer = TfidfVectorizer()
        _doc_texts = []
        _doc_titles = []
        _doc_matrix = None
        return

    _vectorizer = TfidfVectorizer(stop_words="english")
    _doc_matrix = _vectorizer.fit_transform(texts)
    _doc_texts = texts
    _doc_titles = titles


def _extract_bullet_tips(md_text: str, max_tips: int = 5) -> List[str]:
    """
    Very simple Markdown bullet extractor:
    - take lines starting with '- ' or '* '
    - strip the marker and whitespace
    - de-duplicate
    """
    tips: List[str] = []
    for line in md_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("- ") or stripped.startswith("* "):
            tip = stripped[2:].strip()
            if tip and tip not in tips:
                tips.append(tip)
        if len(tips) >= max_tips:
            break
    return tips


def _retrieve_kb_tips(question: str, top_k_docs: int = 3, max_tips: int = 5) -> List[str]:
    """
    Retrieve a few grounded tips from the local KB based on the question
    using cosine similarity over TF-IDF vectors.

    Returns a flat list of tip strings.
    """
    if (
        not question
        or _vectorizer is None
        or _doc_matrix is None
        or _doc_matrix.shape[0] == 0
    ):
        return []

    q_vec = _vectorizer.transform([question])
    scores = cosine_similarity(q_vec, _doc_matrix)[0]

    ranked_indices = np.argsort(scores)[::-1]

    tips: List[str] = []
    for idx in ranked_indices[:top_k_docs]:
        if scores[idx] <= 0.05:  # ignore almost-zero matches
            break

        doc_text = _doc_texts[idx]
        doc_tips = _extract_bullet_tips(doc_text, max_tips=max_tips)

        for tip in doc_tips:
            if tip not in tips:
                tips.append(tip)
            if len(tips) >= max_tips:
                break

        if len(tips) >= max_tips:
            break

    return tips


def generate_coach_response(
    *,
    profile: SpendingProfile,
    probability: float,
    risk_level: str,
    base_suggestions: List[str],
    question: str | None,
) -> CoachResponse:
    """
    Build the full coach response:

    - Narrative explanation (answer) – multiple paragraphs separated by blank lines
    - Echo of model-based suggestions (already shown above in the UI)
    - List of grounded tips from the local KB (kb_tips)
    """
    ensure_kb_index()

    # ----- Narrative explanation -----
    paragraphs: List[str] = []

    # 1) Profile + risk summary
    paragraphs.append(
        (
            "For this profile (income={income:.1f}, housing={housing:.1f}, "
            "food={food:.1f}, transport={transport:.1f}, shopping={shopping:.1f}, "
            "entertainment={entertainment:.1f}, other={other:.1f}, "
            "savings_rate={savings_rate:.2f}), the estimated overspending "
            "probability is about {prob:.1%} ({risk} risk)."
        ).format(
            income=profile.income,
            housing=profile.housing,
            food=profile.food,
            transport=profile.transport,
            shopping=profile.shopping,
            entertainment=profile.entertainment,
            other=profile.other,
            savings_rate=profile.savings_rate,
            prob=probability,
            risk=risk_level.upper(),
        )
    )

    if risk_level == "low":
        paragraphs.append(
            "Overall, this suggests low risk of overspending. Nice work – keep your habits steady."
        )
    elif risk_level == "medium":
        paragraphs.append(
            "Overall, this suggests a moderate risk of overspending. Tightening non-essential categories this month could help."
        )
    else:
        paragraphs.append(
            "Overall, this suggests a high risk of overspending. You may want to tighten non-essential categories and watch variable spending closely."
        )

    # 2) User question
    if question:
        paragraphs.append(f'You asked: "{question}".')

    # 3) Where to find model-based suggestions
    paragraphs.append(
        "Model-based suggestions (from the numeric profile) are listed above in the result panel."
    )

    # ----- KB retrieval -----
    kb_tips = _retrieve_kb_tips(question or "", top_k_docs=3, max_tips=5)

    if kb_tips:
        paragraphs.append(
            "Grounded tips from the budgeting knowledge base related to your question are listed below."
        )
    else:
        # Fallback: generic tips if retrieval yields nothing.
        kb_tips = [
            "Build an emergency fund (aim for at least one month of essential expenses).",
            "Cap variable categories like dining out, shopping, and entertainment.",
            "Make sure essentials (rent, utilities, groceries, insurance, transport) are protected first.",
        ]
        paragraphs.append(
            "No strong matches were found in the budgeting knowledge base for this exact question, "
            "but the general budgeting guidelines below still apply."
        )

    answer = "\n\n".join(paragraphs)

    # ----- Assemble response -----
    return CoachResponse(
        overspend_probability=probability,
        risk_level=risk_level,
        model_suggestions=base_suggestions,
        answer=answer,
        kb_tips=kb_tips,
    )