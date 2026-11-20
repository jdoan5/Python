"""Retrieval metrics, computed in sections: the unit the gold labels use.

Two families, because they answer different questions.

Ranking quality: success@5, recall@k, MRR@10, nDCG@10. A chunk ranking is
first turned into a ranking of sections. Walking down the chunks, each section
a chunk retrieves joins the list (most-overlapped first) unless it is already
there, and a chunk that retrieves no section at all, such as a lone heading,
takes a slot as a miss. Counting in sections puts every chunking strategy in
the same units. A strategy that cuts finer doesn't burn slots on more of a
section it already found, and a window that spans three sections fills three
slots, not one.

Answer delivery at a fixed context size: success, recall and coverage within B
characters. Ranking metrics can't see how much text a configuration needs:
five 70-character paragraphs and five 1,200-character windows are both
"top 5". A prompt has a fixed size, so these metrics pack each chunk ranking
into the same budget, the longest prefix of the ranking that fits.

Success is yes/no: did at least one answering section arrive. Recall is a
share: of a question's answering sections, how many arrived. Most questions
have more than one answering section, so the two differ. Coverage measures how
much of their text arrived, in non-whitespace characters. Mean coverage
averages over a question's answering sections, and best coverage takes the
most completely delivered one.

Grades: 2 = answers the question, 1 = related, 0 = not relevant. Success,
recall, MRR and coverage count only grade-2 sections; nDCG uses both positive
grades with gain 2^grade - 1. A section nobody judged scores as not relevant.
`raglab golden pool` checks that every section any configuration ranks in its
top 10 is judged; sections deeper in a budget may not be.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

from raglab.sections import SectionIndex, SectionRef

Span = tuple[str, int, int]  # (doc_id, start, end) of a retrieved chunk

K_VALUES = (1, 3, 5, 10)
RANK_DEPTH = 10  # sections; MRR and nDCG are computed at 10
# Room for 1, 2, 4 and 8 full-size chunks: Stage 1 caps fixed and heading chunks at 1,200 chars.
BUDGETS = (1200, 2400, 4800, 9600)
HEADLINE_BUDGET = 4800  # ~1,200 tokens. Chosen before any results were seen.
GAIN = {1: 1.0, 2: 3.0}

METRICS: tuple[str, ...] = (
    *(f"recall@{k}" for k in K_VALUES),
    "success@5",
    "mrr@10",
    "ndcg@10",
    *(f"budget_recall@{b}" for b in BUDGETS),
    *(f"budget_success@{b}" for b in BUDGETS),
    *(f"coverage@{b}" for b in BUDGETS),
    *(f"best_coverage@{b}" for b in BUDGETS),
)
BINARY_METRICS = frozenset({"success@5", *(f"budget_success@{b}" for b in BUDGETS)})
HEADLINE_METRICS: tuple[str, ...] = (
    "recall@5", "mrr@10", "ndcg@10", f"budget_recall@{HEADLINE_BUDGET}", f"coverage@{HEADLINE_BUDGET}",
)
LABELS = {
    **{f"recall@{k}": f"Recall@{k}" for k in K_VALUES},
    "success@5": "Success@5",
    "mrr@10": "MRR@10",
    "ndcg@10": "nDCG@10",
    **{f"budget_recall@{b}": f"Recall in {b / 1000:g}k chars" for b in BUDGETS},
    **{f"budget_success@{b}": f"Success in {b / 1000:g}k chars" for b in BUDGETS},
    **{f"coverage@{b}": f"Mean coverage in {b / 1000:g}k chars" for b in BUDGETS},
    **{f"best_coverage@{b}": f"Best coverage in {b / 1000:g}k chars" for b in BUDGETS},
}


def section_ranking(spans: Sequence[Span], index: SectionIndex, depth: int = RANK_DEPTH) -> list[str | None]:
    """The chunk ranking as a ranking of section locations; None marks a chunk that retrieved nothing."""
    ranking: list[str | None] = []
    seen: set[str] = set()
    for doc_id, start, end in spans:
        if len(ranking) >= depth:
            break
        retrieved = index.retrieved_by(doc_id, start, end)
        if not retrieved:
            ranking.append(None)
            continue
        for ref, _ in retrieved:
            if ref.location not in seen:
                seen.add(ref.location)
                ranking.append(ref.location)
    return ranking[:depth]


def recall_at(ranking: Sequence[str | None], primary: frozenset[str], k: int) -> float:
    if not primary:
        raise ValueError("a question needs at least one answering (grade 2) section")
    return len(primary.intersection(ranking[:k])) / len(primary)


def reciprocal_rank(ranking: Sequence[str | None], primary: frozenset[str], k: int = RANK_DEPTH) -> float:
    for position, location in enumerate(ranking[:k], start=1):
        if location in primary:
            return 1.0 / position
    return 0.0


def ndcg_at(ranking: Sequence[str | None], judgments: Mapping[str, int], k: int = RANK_DEPTH) -> float:
    gains = [GAIN.get(judgments.get(loc, 0), 0.0) if loc is not None else 0.0 for loc in ranking[:k]]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    ideal = sorted((GAIN[g] for g in judgments.values() if g in GAIN), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def pack(spans: Sequence[Span], budget: int) -> list[Span]:
    """The longest prefix of the ranking that fits in `budget` characters.

    Nothing is truncated and nothing is skipped: a chunk that doesn't fit ends
    the context, as it would in a prompt with a hard limit. Overlapping chunks
    each pay for their own text, because the prompt would carry it twice.
    """
    packed: list[Span] = []
    used = 0
    for span in spans:
        size = span[2] - span[1]
        if used + size > budget:
            break
        packed.append(span)
        used += size
    return packed


def delivered(packed: Sequence[Span], index: SectionIndex) -> set[str]:
    """Every section the packed context retrieves, by the same hit rule as rankings."""
    return {ref.location for doc_id, start, end in packed for ref, _ in index.retrieved_by(doc_id, start, end)}


def budget_recall(packed: Sequence[Span], primary: frozenset[str], index: SectionIndex) -> float:
    if not primary:
        raise ValueError("a question needs at least one answering (grade 2) section")
    return len(primary & delivered(packed, index)) / len(primary)


def coverage(packed: Sequence[Span], refs: Sequence[SectionRef], index: SectionIndex) -> float:
    """Mean, over a question's answering sections, of the share of each one's text in the context."""
    fractions = coverage_fractions(packed, refs, index)
    return sum(fractions) / len(fractions)


def coverage_fractions(packed: Sequence[Span], refs: Sequence[SectionRef], index: SectionIndex) -> list[float]:
    """For each answering section, the share of its text that is inside the packed context.

    Counted in non-whitespace characters, so a chunker that drops the blank
    lines between paragraphs can still reach 1.0 by retrieving every paragraph.
    """
    if not refs:
        raise ValueError("a question needs at least one answering (grade 2) section")
    fractions = []
    for ref in refs:
        pieces = sorted(
            (max(start, ref.start), min(end, ref.end))
            for doc_id, start, end in packed
            if doc_id == ref.doc_id and start < ref.end and end > ref.start
        )
        covered, reach = 0, ref.start
        for lo, hi in pieces:  # union of intervals, so overlapping chunks aren't counted twice
            lo = max(lo, reach)
            if hi > lo:
                covered += index.ink(ref.doc_id, lo, hi)
                reach = hi
        fractions.append(covered / max(index.ink(ref.doc_id, ref.start, ref.end), 1))
    return fractions


def score(
    spans: Sequence[Span], primary: frozenset[str], judgments: Mapping[str, int], index: SectionIndex
) -> tuple[dict[str, float], list[str | None]]:
    """Every metric for one question under one configuration, plus its section ranking."""
    refs = []
    for location in sorted(primary):
        ref = index.get(location)
        if ref is None:
            raise ValueError(f"answering section {location!r} is not in the corpus")
        refs.append(ref)
    ranking = section_ranking(spans, index)
    values: dict[str, float] = {}
    for k in K_VALUES:
        values[f"recall@{k}"] = recall_at(ranking, primary, k)
    values["success@5"] = float(any(loc in primary for loc in ranking[:5]))
    values["mrr@10"] = reciprocal_rank(ranking, primary)
    values["ndcg@10"] = ndcg_at(ranking, judgments)
    for budget in BUDGETS:
        packed = pack(spans, budget)
        arrived = primary & delivered(packed, index)
        values[f"budget_recall@{budget}"] = len(arrived) / len(primary)
        values[f"budget_success@{budget}"] = float(bool(arrived))
        fractions = coverage_fractions(packed, refs, index)
        values[f"coverage@{budget}"] = sum(fractions) / len(fractions)
        values[f"best_coverage@{budget}"] = max(fractions)
    return {m: values[m] for m in METRICS}, ranking
