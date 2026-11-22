"""The pool analyses' rules (analysis/pool_effects.py), on made-up rankings: they never touch the real labels here."""

from __future__ import annotations

import sys
from pathlib import Path
from types import MappingProxyType

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))

import pool_effects as pe  # noqa: E402
from raglab.evaluate import QuestionResult  # noqa: E402
from raglab.golden import GoldenSet, Question  # noqa: E402


def q(qid, target, judgments, style="keyword"):
    return Question(qid, f"text {qid}", style, target, MappingProxyType(judgments))


def gs(questions):
    return GoldenSet(1, "0.1.0", "f" * 64, tuple(questions), "0" * 64, Path("v1.yaml"), {})


def test_families_follow_the_retriever_whatever_the_variant():
    assert pe.family_of("heading+hybrid+rerank [depth 10]") == "hybrid+rerank"
    assert pe.family_of("heading+dense [no instruction]") == "dense"
    assert pe.family_of("baseline (RAG Mini Chat)") == "tfidf"
    assert pe.family_of("heading+bm25+rerank") == "bm25+rerank"


def test_only_judgments_a_single_new_family_surfaced_are_dropped():
    provenance = {"pool": {"items": [
        {"question": "q1", "location": "a#1", "top10": ["heading+dense", "fixed+dense (no breadcrumb)"], "window": []},
        {"question": "q1", "location": "a#2", "top10": ["heading+dense"], "window": ["heading+hybrid"]},
        {"question": "q2", "location": "b#1", "top10": [], "window": ["heading+hybrid+rerank [depth 20]"]},
        {"question": "q2", "location": "b#2", "top10": ["heading+bm25"], "window": []},
    ]}}
    dropped = pe.single_family_judgments(provenance)
    assert dropped["dense"] == {("q1", "a#1")} and dropped["hybrid+rerank"] == {("q2", "b#1")}
    assert dropped["hybrid"] == set() and dropped["bm25+rerank"] == set()


def test_kendall_tau_counts_concordant_and_discordant_pairs():
    assert pe.kendall_tau([1, 2, 3], [1, 2, 3]) == 1.0
    assert pe.kendall_tau([1, 2, 3], [3, 2, 1]) == -1.0
    assert pe.kendall_tau([1, 2, 3, 4], [1, 3, 2, 4]) == pytest.approx(4 / 6)


def test_dropping_judgments_leaves_them_unjudged():
    out = pe.without([q("q1", "a#t", {"a#t": 2, "a#1": 1})], {("q1", "a#1")})
    assert dict(out[0].judgments) == {"a#t": 2}


def test_unreached_answers_and_questions_are_set_by_the_lexical_top_tens():
    questions = [q("q1", "a#t", {"a#t": 2, "a#2": 2}), q("q2", "b#t", {"b#t": 2})]
    results = {"heading+bm25": [QuestionResult({}, ("a#t", "x#1"), 0), QuestionResult({}, ("x#2",), 0)],
               "heading+tfidf": [QuestionResult({}, ("x#3",), 0), QuestionResult({}, ("x#4",), 0)],
               "heading+dense": [QuestionResult({}, ("a#2",), 0), QuestionResult({}, ("x#5", "b#t"), 0)]}
    answers, unanswered = pe.unreached(gs(questions), results)
    assert answers == {("q1", "a#2"), ("q2", "b#t")} and unanswered == {"q2"}
    counted = pe.reach(gs(questions), results, answers, unanswered)
    dense = counted["configurations"]["heading+dense"]
    assert dense["answers_in_top10"] == 2 and dense["questions_answered_in_top10"] == 1
    assert "heading+bm25" not in counted["configurations"]  # only new configurations are counted


def test_target_page_recall_skips_the_target_itself():
    questions = [q("q1", "p#t", {"p#t": 2, "p#a": 2, "p#b": 2, "z#c": 2}), q("q2", "r#t", {"r#t": 2})]
    results = {"x": [QuestionResult({}, ("p#t", "p#a"), 0), QuestionResult({}, ("r#t",), 0)]}
    out = pe.target_page_recall(questions, results)
    assert out["questions"] == 1 and out["answers"] == 2 and out["recall@10"]["x"] == 0.5


def test_leaving_the_source_out_moves_the_sections_below_it_up():
    question = q("q1", "p#t", {"p#t": 2, "p#a": 2})
    assert pe.source_left_out_mrr(question, ["p#t", "x#1", "p#a"]) == pytest.approx(1 / 2)
    assert pe.source_left_out_mrr(question, ["x#1", "p#t"] + [f"y#{i}" for i in range(8)] + ["p#a"]) == pytest.approx(1 / 10)
    assert pe.source_left_out_mrr(q("q2", "r#t", {"r#t": 2}), ["r#t"]) is None  # no other answer: not asked


def test_a_nudge_moves_each_value_by_exactly_one_float32_unit_in_the_last_place():
    import numpy as np

    import perturbation

    values = np.array([0.7, -3.25, 5.5, 1e-3])
    moved = perturbation.nudge(values, np.random.default_rng(1))
    ulp = np.spacing(np.abs(values).astype(np.float32)).astype(np.float64)
    assert np.allclose(np.abs(moved - values), ulp, rtol=0, atol=0)
    many = perturbation.nudge(np.full(1000, 0.5), np.random.default_rng(2))
    assert 400 < np.sum(many > 0.5) < 600  # up and down equally often
    names = [c.name for c in perturbation.chain_configs()]
    assert names == ["heading+bm25", "heading+hybrid+rerank", "heading+hybrid", "heading+dense", "heading+bm25+rerank"]


def test_the_bench_helpers_time_without_caching():
    import time

    from raglab import bench

    clock = {}

    class Slow:
        settings = {"model": "slow"}

        def embed(self, texts, instruction):
            time.sleep(0.01)
            return texts

    timed = bench._Timed(Slow(), "embed", clock, "question embedding")
    timed.embed(["a"], "")
    timed.embed(["a"], "")  # the same text again: computed again, never cached
    assert clock["question embedding"] >= 0.02
    stats = bench._percentiles([0.001, 0.002, 0.003, 0.004])
    assert stats["n"] == 4 and stats["p50_ms"] == pytest.approx(2.5)
    assert {"chip", "cores", "ram_gb", "power", "os"} <= set(bench.machine())
