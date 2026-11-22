"""The hand-graded check: which questions and answers the developer grades (PLAN.md, "The hand-graded check").

The draw is made at freeze point 3, before any golden judge call, and prints
only question ids. Strata are what code and the sufficiency audit can see:
whether S says hhr's window answers the question (yes or partly) or not, crossed
with question style. The 12 are shared in proportion to the 98 by largest
remainders, first between the two S classes (about 10 and 2), then between
styles within each class. Within a stratum questions go in sha256 order of
"human:" + id.

A question whose two run-1 answers both abstain (the code flag) is replaced by
the next in its stratum, since it holds nothing to grade. A question where only
one arm abstains is kept, and only its other answer is graded: that answer
alone decides the question's paired difference, which is what the bias check
needs to see. A short stratum is filled from the largest.

Grading order goes round the strata, largest first, so a sitting that stops
early still covers both S classes.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from raglab import params


@dataclass(frozen=True)
class Candidate:
    id: str
    style: str
    answerable: bool  # S said hhr's window answers it (yes or partly)


@dataclass(frozen=True)
class Draw:
    order: tuple[str, ...]                   # questions, in grading order
    grade: Mapping[str, tuple[str, ...]]     # question -> the arms whose answers are graded


def _order(prefix: str, qid: str) -> str:
    return hashlib.sha256(f"{prefix}:{qid}".encode("utf-8")).hexdigest()


def allocate(sizes: dict, n: int) -> dict:
    """Shares of n in proportion to sizes, by largest remainders (ties by key, for a fixed result)."""
    total = sum(sizes.values())
    if total == 0 or n <= 0:
        return {k: 0 for k in sizes}
    quotas = {k: n * v / total for k, v in sizes.items()}
    out = {k: int(q) for k, q in quotas.items()}
    for k in sorted(quotas, key=lambda k: (-(quotas[k] - out[k]), str(k)))[: n - sum(out.values())]:
        out[k] += 1
    return out


def draw(candidates: Sequence[Candidate], abstained: Mapping[str, Iterable[str]], arms: Sequence[str],
         n: int | None = None) -> Draw:
    """Which questions to grade, in what order, and which of each question's answers.

    `abstained` maps a question to the arms whose run-1 answer is a code-flagged full abstention.
    """
    n = params.HAND_GRADED["questions"] if n is None else n
    gone = {q: set(a) for q, a in abstained.items()}
    strata: dict[tuple[bool, str], list[str]] = defaultdict(list)
    for c in candidates:
        strata[(c.answerable, c.style)].append(c.id)
    for members in strata.values():
        members.sort(key=lambda q: _order("human", q))
    # Shares in two steps, so rounding can't erase a class: first answerable or not, then style within each.
    classes = allocate({a: sum(len(v) for k, v in strata.items() if k[0] == a) for a in {k[0] for k in strata}}, n)
    shares: dict = {}
    for a, m in classes.items():
        shares |= allocate({k: len(v) for k, v in strata.items() if k[0] == a}, m)
    chosen: dict[tuple[bool, str], list[str]] = {}
    rest, short = {}, 0
    for k, members in strata.items():
        eligible = [q for q in members if gone.get(q, set()) != set(arms)]  # both arms abstained: nothing to grade
        chosen[k] = eligible[: shares[k]]
        short += max(0, shares[k] - len(eligible))
        rest[k] = eligible[shares[k]:]
    for k in sorted(rest, key=lambda k: (-len(strata[k]), str(k))):
        take, rest[k] = rest[k][:short], rest[k][short:]
        chosen[k] += take
        short -= len(take)
        if short == 0:
            break
    # Round-robin over strata, largest first; within a stratum, sha256 order.
    queues = [list(chosen[k]) for k in sorted(chosen, key=lambda k: (-len(strata[k]), str(k))) if chosen[k]]
    order: list[str] = []
    while any(queues):
        for queue in queues:
            if queue:
                order.append(queue.pop(0))
    grade = {q: tuple(a for a in arms if a not in gone.get(q, set())) for q in order}
    return Draw(tuple(order), grade)


def sitting_one_size(minutes_per_question: float, sitting_minutes: float = 60.0) -> int:
    """How many questions sitting 1 grades: as many as the dry run's pace says sitting 2 can re-cover in about an
    hour, at least the floor of 8 and at most the 12 drawn (PLAN.md, "Time")."""
    if minutes_per_question <= 0:
        raise ValueError("the pace must be positive")
    fit = int(sitting_minutes // minutes_per_question)
    return max(params.HAND_GRADED["floor"], min(params.HAND_GRADED["questions"], fit))
