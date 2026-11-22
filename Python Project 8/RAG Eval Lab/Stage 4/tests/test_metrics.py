from __future__ import annotations

import math

import pytest

from conftest import make_doc
from raglab.metrics import (
    BUDGETS, HEADLINE_METRICS, METRICS, budget_recall, coverage, ndcg_at, pack, reciprocal_rank, recall_at, score,
    section_ranking,
)
from raglab.sections import SectionIndex

LONG = "word " * 60  # 300 chars of body: longer than MIN_OVERLAP_CHARS
PAGE = f"""\
# Page {{ #page }}

{LONG}

## A {{ #a }}

{LONG}

## B {{ #b }}

{LONG}

## C {{ #c }}

{LONG}
"""


@pytest.fixture
def doc():
    return make_doc(PAGE, "p")


@pytest.fixture
def index(doc):
    return SectionIndex([doc])


def body(index, location):
    ref = index.get(location)
    return ("p", ref.start, ref.end)


def heading_line(doc, title):
    start = doc.text.index(f"## {title}")
    return ("p", start, doc.text.index("\n", start))


# --- section ranking -------------------------------------------------------------------------


def test_ranking_lists_sections_in_order_and_skips_repeats(index):
    a, b = body(index, "p#a"), body(index, "p#b")
    a_again = ("p", a[1] + 10, a[2])
    assert section_ranking([a, a_again, b], index) == ["p#a", "p#b"]


def test_a_chunk_that_retrieves_nothing_takes_a_slot_as_a_miss(doc, index):
    assert section_ranking([heading_line(doc, "A"), body(index, "p#b")], index) == [None, "p#b"]


def test_a_window_over_two_sections_fills_two_slots_biggest_overlap_first(index):
    a, b = index.get("p#a"), index.get("p#b")
    window = ("p", a.end - 150, b.start + 250)  # 150 chars of A, 250 of B
    assert section_ranking([window], index) == ["p#b", "p#a"]


def test_ranking_stops_at_depth(index):
    spans = [body(index, f"p#{x}") for x in "abc"]
    assert section_ranking(spans, index, depth=2) == ["p#a", "p#b"]


# --- rank metrics, by hand -----------------------------------------------------------------------


def test_recall_counts_answering_sections_found_within_k():
    ranking = [None, "x", "a", "b"]
    assert recall_at(ranking, frozenset({"a", "b"}), 2) == 0.0
    assert recall_at(ranking, frozenset({"a", "b"}), 3) == 0.5
    assert recall_at(ranking, frozenset({"a", "b"}), 4) == 1.0


def test_reciprocal_rank_is_one_over_the_first_answers_position():
    assert reciprocal_rank([None, "x", "a"], frozenset({"a"})) == pytest.approx(1 / 3)
    assert reciprocal_rank(["x"] * 10 + ["a"], frozenset({"a"})) == 0.0  # beyond @10


def test_ndcg_matches_a_hand_computation():
    # gains: grade 2 -> 3, grade 1 -> 1. Ranking: miss, A(2), unjudged, B(1).
    judgments = {"a": 2, "b": 1, "z": 0}
    dcg = 3 / math.log2(3) + 1 / math.log2(5)
    idcg = 3 / math.log2(2) + 1 / math.log2(3)
    assert ndcg_at([None, "a", "x", "b"], judgments) == pytest.approx(dcg / idcg)
    assert ndcg_at(["a", "b"], judgments) == pytest.approx(1.0)
    assert ndcg_at(["z", "x"], judgments) == 0.0


def test_empty_answers_are_a_programming_error():
    with pytest.raises(ValueError):
        recall_at(["a"], frozenset(), 1)


# --- budget metrics --------------------------------------------------------------------------------


def test_pack_takes_the_longest_prefix_that_fits_and_never_skips_ahead():
    spans = [("p", 0, 600), ("p", 0, 500), ("p", 0, 100)]
    assert pack(spans, 1100) == spans[:2]
    assert pack(spans, 1099) == spans[:1]  # the 100-char chunk would fit, but it ranks after one that doesn't
    assert pack(spans, 599) == []


def test_budget_recall_uses_the_same_hit_rule_as_rankings(doc, index):
    packed = [heading_line(doc, "A"), body(index, "p#b")]
    assert budget_recall(packed, frozenset({"p#a", "p#b"}), index) == 0.5


def test_coverage_counts_each_character_once(index):
    a = index.get("p#a")
    half = (a.start + a.end) // 2
    overlapping = [("p", a.start, half), ("p", a.start + 10, half), ("p", half, half + 30)]
    expected = index.ink("p", a.start, half + 30) / index.ink("p", a.start, a.end)
    assert coverage(overlapping, [a], index) == pytest.approx(expected)
    assert coverage([("other", a.start, a.end)], [a], index) == 0.0
    assert coverage([body(index, "p#a")], [a, index.get("p#b")], index) == pytest.approx(0.5)


def test_coverage_ignores_the_blank_lines_a_chunker_drops():
    doc = make_doc("# T { #t }\n\nFirst paragraph here.\n\nSecond paragraph here.\n", "d")
    index = SectionIndex([doc])
    ref = index.get("d#t")
    paragraphs = [("d", doc.text.index(p), doc.text.index(p) + len(p))
                  for p in ("First paragraph here.", "Second paragraph here.")]
    assert coverage(paragraphs, [ref], index) == pytest.approx(1.0)  # every paragraph: all of the text


# --- the whole score -------------------------------------------------------------------------------


def test_a_perfect_ranking_scores_one_everywhere(index):
    values, ranking = score([body(index, "p#a")], frozenset({"p#a"}), {"p#a": 2}, index)
    assert ranking == ["p#a"]
    assert set(values) == set(METRICS) and set(HEADLINE_METRICS) <= set(values)
    assert all(v == pytest.approx(1.0) for v in values.values())


def test_a_ranking_that_finds_nothing_scores_zero_everywhere(doc, index):
    values, _ = score([heading_line(doc, "A")], frozenset({"p#a"}), {"p#a": 2}, index)
    assert all(v == 0.0 for v in values.values())


def test_scores_stay_between_zero_and_one_for_windows_over_many_sections(index):
    page, c = index.get("p#page"), index.get("p#c")
    windows = [("p", start, start + 1200) for start in range(page.start, c.end, 200)]
    judgments = {"p#a": 2, "p#b": 1, "p#c": 2}
    values, _ = score(windows, frozenset({"p#a", "p#c"}), judgments, index)
    assert all(0.0 <= v <= 1.0 for v in values.values()), values


def test_budget_metrics_reward_fitting_the_answer_in_a_small_context(index):
    page, a, c = index.get("p#page"), index.get("p#a"), index.get("p#c")
    small = [("p", a.start, a.end)]              # 300 chars: fits every budget
    padded = [("p", page.start, c.end)]          # the whole page: too big for the smallest budget
    assert BUDGETS[0] < c.end - page.start <= BUDGETS[1]
    v_small, _ = score(small, frozenset({"p#a"}), {"p#a": 2}, index)
    v_padded, _ = score(padded, frozenset({"p#a"}), {"p#a": 2}, index)
    for budget in BUDGETS:
        assert v_small[f"budget_recall@{budget}"] == 1.0
        assert v_padded[f"budget_recall@{budget}"] == (0.0 if budget == BUDGETS[0] else 1.0)
    assert v_small["recall@5"] == v_padded["recall@5"] == 1.0  # top-5 recall can't tell them apart


def test_an_answer_outside_the_corpus_is_reported_not_scored(index):
    with pytest.raises(ValueError, match="not in the corpus"):
        score([], frozenset({"p#nope"}), {"p#nope": 2}, index)


def test_success_is_yes_no_while_recall_is_a_share(index):
    # Two answering sections, only one retrieved: recall says half, success says yes.
    values, _ = score([body(index, "p#a")], frozenset({"p#a", "p#c"}), {"p#a": 2, "p#c": 2}, index)
    assert values["recall@5"] == 0.5 and values["success@5"] == 1.0
    for budget in BUDGETS:
        assert values[f"budget_recall@{budget}"] == 0.5 and values[f"budget_success@{budget}"] == 1.0
    nothing, _ = score([body(index, "p#b")], frozenset({"p#a"}), {"p#a": 2}, index)
    assert nothing["success@5"] == 0.0 and nothing[f"budget_success@{BUDGETS[-1]}"] == 0.0


def test_budget_recall_ignores_a_clipped_edge(doc, index):
    # 50 chars of A's body plus all of B's: by the hit rule, B arrives and A doesn't.
    a, b = index.get("p#a"), index.get("p#b")
    window = ("p", a.end - 50, b.end)
    assert budget_recall([window], frozenset({"p#a"}), index) == 0.0
    assert budget_recall([window], frozenset({"p#b"}), index) == 1.0


def test_ndcg_ideal_comes_from_the_labels_not_from_what_was_retrieved():
    judgments = {"a": 2, "b": 1}
    assert ndcg_at(["b"], judgments) == pytest.approx(1 / (3 + 1 / math.log2(3)))
    many = {f"s{i}": 1 for i in range(12)}  # more answers than slots: the ideal is cut at 10
    ideal = sum(1 / math.log2(i + 2) for i in range(10))
    assert ndcg_at(["s0"], many) == pytest.approx(1 / ideal)


def test_best_coverage_is_the_most_completely_delivered_answer(index):
    # Two answering sections; one arrives whole, the other not at all.
    values, _ = score([body(index, "p#a")], frozenset({"p#a", "p#c"}), {"p#a": 2, "p#c": 2}, index)
    for budget in BUDGETS:
        assert values[f"coverage@{budget}"] == pytest.approx(0.5)
        assert values[f"best_coverage@{budget}"] == pytest.approx(1.0)


def test_answers_only_ndcg_ignores_related_sections():
    from raglab.metrics import ANSWER_GAIN, ndcg_at

    judgments = {"a#1": 2, "b#1": 1, "c#1": 1}
    related_first = ["b#1", "c#1", "a#1"]
    assert ndcg_at(related_first, judgments) > ndcg_at(related_first, judgments, gain=ANSWER_GAIN)
    assert ndcg_at(["a#1", "b#1"], judgments, gain=ANSWER_GAIN) == 1.0
    assert ndcg_at(["b#1", "c#1"], judgments, gain=ANSWER_GAIN) == 0.0  # no answer at all: no credit
    assert ndcg_at(["b#1", "a#1"], judgments, gain=ANSWER_GAIN) == pytest.approx(1 / math.log2(3))
