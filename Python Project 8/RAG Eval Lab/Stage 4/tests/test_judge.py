"""The judges: what each sees, what counts as invalid, how quotes and verdicts become SC."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from raglab.blocks import window_documents
from raglab.judge import (
    InvalidVerdict, Nugget, ReferenceSection, check_correctness, check_support, correctness_body, outcome,
    quote_holds, read_output, render_support, support_body,
)
from raglab.sections import SectionIndex
from raglab.units import parse_reply
from tests.conftest import make_doc
from tests.test_units import PAGE, cite, message

STAGE = Path(__file__).resolve().parents[1]


@pytest.fixture
def answered():
    doc = make_doc(PAGE, "tutorial/errors")
    index = SectionIndex([doc])
    docs = window_documents([(doc.doc_id, 0, len(doc.text.rstrip()))], {doc.doc_id: doc}, index)
    reply = message([
        {"type": "text", "text": "Raise `HTTPException` with a status code.", "citations": [cite(docs, 0, 3, 5)]},
        {"type": "text", "text": " The client gets a JSON body.", "citations": [cite(docs, 0, 5, 6)]},
        {"type": "text", "text": " It also logs the error.", "citations": None},
    ])
    return parse_reply(reply, docs), docs


KEY = [Nugget("N1", "Raise HTTPException with a status code", True), Nugget("N2", "The body is JSON", False)]
REFERENCE = [ReferenceSection("tutorial/errors#raise-one", "Errors > Raise one",
                              "Raise it with a status code:\n\nThe client gets a JSON body.")]


def f_output(**changes):
    units = [
        {"id": "U1", "claim": True, "support": "supported", "quotes": [{"excerpt": "U1.E1", "quote": "Raise it with a status code:"}], "reason": "stated"},
        {"id": "U2", "claim": True, "support": "supported", "quotes": [{"excerpt": "U2.E1", "quote": "The client gets a JSON body."}], "reason": "stated"},
        {"id": "U3", "claim": True, "support": "uncited", "quotes": [], "reason": "no citation"},
    ]
    for uid, fields in changes.items():
        next(u for u in units if u["id"] == uid).update(fields)
    return {"units": units}


def c_output(n1="yes", contradiction=None, abstention="none"):
    return {"nuggets": [{"id": "N1", "conveyed": n1, "answer_quote": "Raise"}, {"id": "N2", "conveyed": "no", "answer_quote": ""}],
            "contradiction": contradiction or {"present": False, "reference_quote": "", "answer_quote": ""},
            "abstention": abstention, "resolves": "yes", "reason": "fine"}


def test_quote_holds_is_stage_3_s_rule():
    sys.path.insert(0, str(STAGE / "analysis"))
    import v2_labels

    for quote, text in (("a  b", "x a b y"), ("", "anything"), ("a b", "a\nb"), ("c", "a b")):
        assert quote_holds(quote, text) == v2_labels.quote_holds(quote, text)


def test_the_support_judge_sees_units_and_their_excerpts_only(answered):
    answer, docs = answered
    text = render_support("How do I return a 404?", answer, docs)
    assert "U1: Raise `HTTPException` with a status code." in text
    assert "U1.E1: Raise it with a status code:" in text and "(no citations)" in text
    assert "Errors > Raise one" not in text  # titles locate an excerpt; F never sees them
    body = support_body("q", answer, docs, 8000)
    assert body["model"] == "claude-opus-5-5" and body["output_config"]["effort"] == "medium"
    assert body["output_config"]["format"]["type"] == "json_schema"
    (block,) = body["messages"][0]["content"]
    assert block["type"] == "text" and "citations" not in block  # 2026-10-21 refuses cited user text
    for key in ("temperature", "fallbacks", "tool_choice", "thinking"):
        assert key not in body


def test_support_needs_a_quote_that_holds(answered):
    answer, docs = answered
    v = check_support(f_output(), answer, docs)
    assert [u.supported for u in v.units] == [True, True, False]
    assert v.support_rate == 2 / 3 and not v.all_supported and v.any_unbacked  # U3 is an uncited claim
    made_up = check_support(f_output(U1={"quotes": [{"excerpt": "U1.E1", "quote": "Raise it twice"}]}), answer, docs)
    assert not made_up.units[0].supported and made_up.units[0].quotes_failed == 1
    framing = check_support(f_output(U3={"claim": False, "support": "n/a"}), answer, docs)
    assert framing.all_supported and framing.support_rate == 1.0


def test_invalid_support_output_is_refused(answered):
    answer, docs = answered
    with pytest.raises(InvalidVerdict, match="unit ids"):
        check_support({"units": f_output()["units"][:2]}, answer, docs)
    with pytest.raises(InvalidVerdict, match="not one of its own"):
        check_support(f_output(U1={"quotes": [{"excerpt": "U2.E1", "quote": "The client"}]}), answer, docs)
    with pytest.raises(InvalidVerdict, match="no quote"):
        check_support(f_output(U2={"quotes": []}), answer, docs)
    with pytest.raises(InvalidVerdict, match="a cited claim marked uncited"):
        check_support(f_output(U1={"support": "uncited", "quotes": []}), answer, docs)
    with pytest.raises(InvalidVerdict, match="JSON"):
        read_output({"content": [{"type": "text", "text": "not json"}]})
    assert read_output({"content": [{"type": "text", "text": json.dumps(f_output())}]}) == f_output()


def test_correctness_counts_essential_nuggets_and_only_contradictions_that_hold():
    assert check_correctness(c_output(), KEY, REFERENCE).correct  # N2 is supporting, so "no" is fine
    assert not check_correctness(c_output(n1="partly"), KEY, REFERENCE).correct
    real = {"present": True, "reference_quote": "The client gets a JSON body.", "answer_quote": "XML"}
    invented = {"present": True, "reference_quote": "The client gets XML.", "answer_quote": "XML"}
    assert not check_correctness(c_output(contradiction=real), KEY, REFERENCE).correct
    v = check_correctness(c_output(contradiction=invented), KEY, REFERENCE)
    assert v.correct and v.contradiction_claimed and v.warnings
    with pytest.raises(InvalidVerdict):
        check_correctness({**c_output(), "nuggets": c_output()["nuggets"][:1]}, KEY, REFERENCE)
    body = correctness_body("q", KEY, REFERENCE, "Raise it.", 8000)
    text = body["messages"][0]["content"][0]["text"]
    assert "N1 (essential)" in text and "R1: Errors > Raise one" in text and "<answer>\nRaise it.\n</answer>" in text


def test_sc_needs_every_claim_supported_and_every_essential_nugget(answered):
    answer, docs = answered
    support_ok = check_support(f_output(U3={"claim": False, "support": "n/a"}), answer, docs)
    support_bad = check_support(f_output(), answer, docs)
    right, wrong = check_correctness(c_output(), KEY, REFERENCE), check_correctness(c_output(n1="no"), KEY, REFERENCE)
    assert outcome(answer, support_ok, right).sc == 1 and outcome(answer, support_ok, right).penalized == 1
    o = outcome(answer, support_bad, right)
    assert (o.sc, o.correct, o.penalized) == (0, 1, -1)
    assert outcome(answer, support_ok, wrong).sc == 0
    assert outcome(answer, None, right).sc == 0  # unjudged support scores 0
    full = check_correctness(c_output(abstention="full"), KEY, REFERENCE)
    a = outcome(answer, support_ok, full)
    assert a.abstained and (a.sc, a.correct, a.penalized) == (0, 0, 0)


def test_s3_is_0_for_a_partial_claim_alone_and_for_an_unjudged_answer(answered):
    answer, docs = answered
    right = check_correctness(c_output(), KEY, REFERENCE)
    partial_only = check_support(f_output(U3={"claim": False, "support": "n/a"},
                                          U2={"support": "partial"}), answer, docs)
    o = outcome(answer, partial_only, right)
    assert (o.sc, o.penalized) == (0, 0)  # not SC, but nothing unbacked either
    flawed = check_support(f_output(), answer, docs)  # an unsupported claim
    assert outcome(answer, flawed, right).penalized == -1
    assert outcome(answer, flawed, None).penalized == 0  # C unjudged: S3 is unjudged, scored 0
    wrong = check_correctness(c_output(contradiction={"present": True, "reference_quote": "The client gets a JSON body.",
                                                      "answer_quote": "XML"}), KEY, REFERENCE)
    assert outcome(answer, None, wrong).penalized == 0  # F unjudged


def test_judge_requests_name_the_answer_they_judge():
    from raglab.judge import judge_condition
    from raglab.store import request_key

    body = {"model": "claude-opus-5-5", "messages": [{"role": "user", "content": "the same abstention"}]}
    run1, run2 = "r" + "1" * 56 + "s00", "r" + "1" * 56 + "s01"  # two runs, byte-identical answers
    assert request_key(judge_condition("C", run1), body) != request_key(judge_condition("C", run2), body)
    assert judge_condition("F", run1, "swap-3") == f"F:{run1}:swap-3"
    with pytest.raises(ValueError):
        judge_condition("J", run1)


def test_a_heading_line_is_framing_whatever_the_judge_says(answered):
    from raglab.units import parse_reply

    answer, docs = answered
    reply = {"content": [{"type": "text", "text": "## Raising errors\nRaise `HTTPException` with a status code.",
                          "citations": None}], "stop_reason": "end_turn"}
    a = parse_reply(reply, docs)
    out = {"units": [{"id": "U1", "claim": True, "support": "uncited", "quotes": [], "reason": "heading"},
                     {"id": "U2", "claim": True, "support": "uncited", "quotes": [], "reason": "no citation"}]}
    v = check_support(out, a, docs)
    assert not v.units[0].claim and v.units[1].claim and any("heading" in w for w in v.warnings)
