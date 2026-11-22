"""The pilot's judge request list: what is listed, in what order, with which samples. No API."""

from __future__ import annotations

from collections import Counter

import pytest

from raglab.blocks import window_documents
from raglab.generate import generation_body
from raglab.judgeplan import (
    PILOT_ITEMS, Stored, graded_questions, pilot_candidates, pilot_judge_list, reference_sections, stored_answers,
)
from raglab.llm import fake_message
from raglab.plan4 import line_for
from raglab.sections import SectionIndex
from raglab.sets import AskedQuestion
from raglab.store import Record
from raglab.windows import CONDITIONS
from tests.conftest import make_doc
from tests.test_planted import GRADES, PAGE, TARGET


@pytest.fixture
def world():
    doc = make_doc(PAGE, "tutorial/errors")
    index = SectionIndex([doc])
    by_id = {doc.doc_id: doc}
    ref = index.get(TARGET)
    docs = window_documents([(doc.doc_id, ref.start, ref.end)], by_id, index)
    questions = [AskedQuestion("p01", "How do I return a 404?", "keyword", TARGET, frozenset({TARGET})),
                 AskedQuestion("p02", "How do I rate-limit?", "keyword", None, frozenset())]
    keys = {"p01": {"nuggets": [{"id": "N1", "text": "Raise HTTPException with status_code=404", "essential": True},
                                {"id": "N2", "text": "detail becomes the body", "essential": False}],
                    "reference": [TARGET]}}
    return doc, index, by_id, docs, questions, keys


def reply(docs) -> dict:
    cite = {"type": "content_block_location", "document_index": 0, "start_block_index": 0, "end_block_index": 1,
            "cited_text": docs[0].blocks[0].text}
    return fake_message(content=[{"type": "text", "citations": [cite],
                                  "text": "Raise `HTTPException` with `status_code=404` to return a not-found error."}])


def build(world):
    doc, index, by_id, docs, questions, keys = world
    built = {}
    runs = [("hhr-4.8k-med", 0), ("hhr-4.8k-med", 1), ("base-4.8k-med", 0), ("closed-med", 0),
            ("hhr-4.8k-low", 0), ("smoke:hhr-4.8k-med", 0)]
    records = []
    for q in questions:
        for name, sample in runs:
            if name.startswith("smoke") and q.id != "p01":
                continue
            cond = CONDITIONS[name.split(":")[-1]]
            shown = [] if cond.closed else docs
            body = generation_body(cond, q.text, shown, 1000, "answer", "closed")
            line = line_for(name, q.id, sample, body)
            built[line.custom_id] = (line, body, shown)
            msg = fake_message("I can't say.") if cond.closed else reply(docs)
            records.append(Record(line.custom_id, line.request_key, name, line.body_sha256, "pilot", "succeeded",
                                  msg, None, None, None, "2026-11-21T00:00:00+00:00"))
    return stored_answers(records, built)


def test_the_pilot_list_follows_the_plans_order_and_samples(world) -> None:
    doc, index, by_id, docs, questions, keys = world
    stored = build(world)
    graded = graded_questions(questions, keys, by_id, index)
    grades = {"p01": GRADES}
    cands = pilot_candidates(stored, questions, grades, by_id, index)
    assert {c.kind for c in cands} == {"swap-lexical", "swap-topical"}
    confirmed = {f"{c.answered}:{c.name}": True for c in cands}
    lines = [jl for jl, _ in pilot_judge_list(stored, questions, graded, cands, confirmed)]
    items = list(dict.fromkeys(jl.item for jl in lines))
    assert items == [i for i in PILOT_ITEMS if i in items] and "smoke" == items[0]
    primaries = [jl for jl in lines if jl.role == "primary"]
    smoke = Counter(jl.judge for jl in primaries if jl.item == "smoke")
    assert smoke == {"F": 2, "C": 1}
    assert {jl.line.sample for jl in primaries if jl.item == "smoke" and jl.judge == "F"} == {0, 2}
    assert {jl.line.sample for jl in primaries if jl.item == "pilot-noise"} == {2}
    # Every primary has its retry (the next sample) and its truncation re-run listed beside it.
    for jl in primaries:
        twins = [x for x in lines if x.line.condition == jl.line.condition and x.item == jl.item]
        roles = Counter(x.role for x in twins if x.line.sample in (jl.line.sample, jl.line.sample + 1))
        assert roles["retry"] >= 1 and roles["truncation"] >= 1
    # A question no section answers gets F but never C; closed-book answers get C only.
    assert not [jl for jl in lines if jl.judge == "C" and stored[jl.answered].line.question == "p02"]
    closed = [jl for jl in primaries if jl.item == "pilot-closed"]
    assert closed and {jl.judge for jl in closed} == {"C"}
    swaps = [jl for jl in primaries if jl.item.startswith("pilot-swaps")]
    assert swaps and {jl.judge for jl in swaps} == {"F"} and all(jl.planted for jl in swaps)
    assert len({jl.line.custom_id for jl in lines}) == len(lines)


def test_unconfirmed_swaps_are_never_listed(world) -> None:
    doc, index, by_id, docs, questions, keys = world
    stored = build(world)
    graded = graded_questions(questions, keys, by_id, index)
    cands = pilot_candidates(stored, questions, {"p01": GRADES}, by_id, index)
    lines = [jl for jl, _ in pilot_judge_list(stored, questions, graded, cands, {})]
    assert not [jl for jl in lines if jl.item.startswith("pilot-swaps")]


def test_a_reference_section_is_shown_whole(world) -> None:
    doc, index, by_id, *_ = world
    (sec,) = reference_sections([TARGET], by_id, index)
    assert sec.text.startswith("## Raise one") and "Use `HTTPException` in a dependency too." in sec.text
    with pytest.raises(ValueError, match="doesn't have"):
        reference_sections(["tutorial/errors#nope"], by_id, index)


def test_a_stored_answer_must_match_its_rebuilt_request(world) -> None:
    doc, index, by_id, docs, questions, keys = world
    body = generation_body(CONDITIONS["hhr-4.8k-med"], "q?", docs, 1000, "answer", "closed")
    line = line_for("hhr-4.8k-med", "p01", 0, body)
    rec = Record(line.custom_id, line.request_key, "hhr-4.8k-med", "0" * 64, "pilot", "succeeded", reply(docs),
                 None, None, None, "t")
    with pytest.raises(ValueError, match="isn't the one"):
        stored_answers([rec], {line.custom_id: (line, body, docs)})
    assert isinstance(next(iter(stored_answers([Record(**{**rec.to_json(), "body_sha256": line.body_sha256})],
                                               {line.custom_id: (line, body, docs)}).values())), Stored)


def test_the_golden_draws_follow_the_plans_counts() -> None:
    from raglab import params
    from raglab.judgeplan import golden_needed, masked_draw

    needed = golden_needed()
    assert sum(v for (k, _), v in needed.items() if k.startswith("swap")) == params.INSTRUMENT_CHECK["swaps"]
    assert sum(v for (k, _), v in needed.items() if k.startswith("plant")) == params.INSTRUMENT_CHECK["plants"]
    answers = {f"r{i:02d}": (f"q{i}", "no" if i < 30 else "partly", i % 10 == 0) for i in range(60)}
    picked = masked_draw(answers)
    assert len(picked) == params.MASKED_JUDGED and not any(answers[c][2] for c in picked)
    assert all(answers[c][1] == "no" for c in picked[:27])  # every usable "no" window first


def test_sitting_one_grades_as_many_as_the_pace_allows_within_8_to_12() -> None:
    from raglab.human import sitting_one_size

    assert sitting_one_size(10) == 8 and sitting_one_size(6) == 10 and sitting_one_size(2) == 12


def test_the_smoke_judge_falls_back_and_refuses_when_nothing_can_be_judged(world) -> None:
    from raglab.judgeplan import pilot_judge_list

    doc, index, by_id, docs, questions, keys = world
    stored = build(world)
    graded = graded_questions(questions, keys, by_id, index)
    no_smoke = {cid: s for cid, s in stored.items() if not s.line.condition.startswith("smoke")}
    with pytest.raises(ValueError, match="no smoke-test answer can be judged"):
        pilot_judge_list(no_smoke, questions, graded, [], {})
