"""The hand-grading draw: proportional strata, sha256 order, abstentions handled, round-robin grading order."""

from __future__ import annotations

from raglab.human import Candidate, allocate, draw

ARMS = ("hhr-4.8k-med", "base-4.8k-med")


def golden_like() -> list[Candidate]:
    # 98 questions: 84 whose hhr window answers, 14 that don't (12 paraphrase, 2 scenario), as in Stage 3's windows
    out = [Candidate(f"q{i:03d}", ("keyword", "paraphrase", "scenario")[i % 3], True) for i in range(98)]
    no_answer = [c for c in out if c.style == "paraphrase"][:12] + [c for c in out if c.style == "scenario"][:2]
    ids = {c.id for c in no_answer}
    return [Candidate(c.id, c.style, c.id not in ids) for c in out]


def test_shares_are_proportional_and_sum_to_n() -> None:
    assert allocate({"a": 84, "b": 14}, 12) == {"a": 10, "b": 2}
    assert sum(allocate({"x": 1, "y": 1, "z": 1}, 2).values()) == 2


def test_the_draw_is_fixed_proportional_and_mixes_strata_from_the_start() -> None:
    cands = golden_like()
    first = draw(cands, {}, ARMS)
    assert first == draw(list(reversed(cands)), {}, ARMS)  # a rule, not an accident of order
    assert len(first.order) == 12 and len(set(first.order)) == 12
    by_id = {c.id: c for c in cands}
    assert sum(not by_id[q].answerable for q in first.order) == 2
    assert not all(by_id[q].answerable for q in first.order[:8])  # round-robin: an 8-question floor covers S-no
    assert all(first.grade[q] == ARMS for q in first.order)


def test_one_arm_abstaining_keeps_the_question_and_both_abstaining_replaces_it() -> None:
    cands = golden_like()
    first = draw(cands, {}, ARMS)
    one, both = first.order[0], first.order[1]
    d = draw(cands, {one: ["base-4.8k-med"], both: list(ARMS)}, ARMS)
    assert one in d.order and d.grade[one] == ("hhr-4.8k-med",)  # only the answer that answered is graded
    assert both not in d.order and len(d.order) == 12


def test_a_short_stratum_is_filled_from_the_largest() -> None:
    cands = golden_like()
    no_answer = [c.id for c in cands if not c.answerable]
    d = draw(cands, {q: list(ARMS) for q in no_answer}, ARMS)  # every no-answer question: both arms abstained
    assert len(d.order) == 12 and not set(d.order) & set(no_answer)
