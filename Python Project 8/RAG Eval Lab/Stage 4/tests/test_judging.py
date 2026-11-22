"""Judge request lists: one line per judge and answer, retries listed beside, no collisions across runs."""

from __future__ import annotations

import pytest

from raglab.blocks import window_documents
from raglab.judge import Nugget, ReferenceSection
from raglab.judging import check_unique, judge_requests
from raglab.sections import SectionIndex
from raglab.units import parse_reply
from tests.conftest import make_doc
from tests.test_units import PAGE, cite, message

MAX = {"support": 6000, "correctness": 6000}
KEY = [Nugget("N1", "Raise HTTPException with a status code", True)]
REF = [ReferenceSection("tutorial/errors#raise-one", "Errors > Raise one", "Raise it with a status code:")]


@pytest.fixture
def docs():
    doc = make_doc(PAGE, "tutorial/errors")
    return window_documents([(doc.doc_id, 0, len(doc.text.rstrip()))], {doc.doc_id: doc}, SectionIndex([doc]))


def answer_of(docs, text="Raise `HTTPException` with a status code.", cited=True):
    return parse_reply(message([{"type": "text", "text": text, "citations": [cite(docs, 0, 3, 5)] if cited else None}]),
                       docs)


def test_each_judge_gets_a_primary_a_retry_and_a_truncation_line(docs):
    lines = judge_requests("r" + "a" * 56 + "s00", "q001", "How?", answer_of(docs), docs, KEY, REF, "core", MAX)
    roles = [(jl.judge, jl.role) for jl, _ in lines]
    assert roles == [("F", "primary"), ("F", "retry"), ("F", "truncation"),
                     ("C", "primary"), ("C", "retry"), ("C", "truncation")]
    (prim, pb), (retry, rb), (trunc, tb) = lines[:3]
    assert retry.line.request_key == prim.line.request_key and retry.line.sample == prim.line.sample + 1
    assert tb["max_tokens"] == 12000 and trunc.line.request_key != prim.line.request_key
    check_unique([jl for jl, _ in lines])


def test_two_runs_with_identical_answers_are_judged_separately(docs):
    a = answer_of(docs)
    run1 = judge_requests("r" + "a" * 56 + "s00", "q001", "How?", a, docs, KEY, REF, "core", MAX)
    run2 = judge_requests("r" + "a" * 56 + "s01", "q001", "How?", a, docs, KEY, REF, "link1-run2", MAX)
    check_unique([jl for jl, _ in run1 + run2])  # same bodies, different answers judged: no collision


def test_abstentions_and_unanswerable_questions_get_no_judge_they_dont_need(docs):
    abst = answer_of(docs, "The provided documentation does not answer this question.", cited=False)
    assert judge_requests("r" + "b" * 56 + "s00", "q002", "How?", abst, docs, KEY, REF, "core", MAX) == []
    no_key = judge_requests("r" + "c" * 56 + "s00", "p04", "How?", answer_of(docs), docs, None, None, "core", MAX)
    assert {jl.judge for jl, _ in no_key} == {"F"}
    closed = judge_requests("r" + "d" * 56 + "s00", "q001", "How?", answer_of(docs), [], KEY, REF, "closed-book", MAX,
                            support=False)
    assert {jl.judge for jl, _ in closed} == {"C"}


def test_a_judge_list_round_trips_without_bodies(tmp_path) -> None:
    from raglab.judging import JudgeLine, read_judge_list, write_judge_list
    from raglab.plan4 import line_for

    body = {"model": "claude-opus-5-5", "max_tokens": 10, "messages": [{"role": "user", "content": "secret doc text"}]}
    lines = [JudgeLine(line_for("F:rx", "p01", 0, body), "F", "rx", "primary", "pilot-medium"),
             JudgeLine(line_for("F:rx:swap-lexical:U1", "p01", 0, body), "F", "rx", "primary", "pilot-swaps-low",
                       "swap-lexical:U1")]
    path = tmp_path / "judge-requests.jsonl"
    write_judge_list(path, lines)
    assert read_judge_list(path) == lines
    assert "secret doc text" not in path.read_text()
