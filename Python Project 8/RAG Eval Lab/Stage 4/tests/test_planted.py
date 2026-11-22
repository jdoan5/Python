"""Near-miss swaps and true-support plants: which units, which blocks, the draw and the gate. No API."""

from __future__ import annotations

import pytest

from raglab.blocks import window_documents
from raglab.judge import excerpts, support_body
from raglab.planted import (
    Candidate, baseline_pieces, candidates_for, draw, gate, lexical_near_miss, literals, names, planted_excerpts,
    topical_near_miss,
)
from raglab.sections import SectionIndex
from raglab.units import parse_reply
from tests.conftest import make_doc

PAGE = """\
# Errors { #errors }

Intro text about errors.

## Raise one { #raise-one }

Raise `HTTPException` with `status_code=404` to return a not-found error.

The `detail` you pass becomes the JSON body.

```python
raise HTTPException(status_code=404, detail="Item not found")

# the client sees {"detail": "Item not found"}
```

Use `HTTPException` in a dependency too.

## Custom headers { #custom-headers }

Pass `headers` to `HTTPException` to add response headers.

## Handlers { #handlers }

Install a handler with `app.exception_handler` for your own exception classes.
"""

GRADES = {"tutorial/errors#raise-one": 2, "tutorial/errors#custom-headers": 2, "tutorial/errors#handlers": 1,
          "tutorial/errors#errors": 0}
TARGET = "tutorial/errors#raise-one"


@pytest.fixture
def world():
    doc = make_doc(PAGE, "tutorial/errors")
    index = SectionIndex([doc])
    corpus = {doc.doc_id: doc}
    ref = index.get(TARGET)
    docs = window_documents([(doc.doc_id, ref.start, ref.end)], corpus, index)
    return doc, index, corpus, docs


def cite(docs, start, end) -> dict:
    return {"type": "content_block_location", "document_index": 0, "start_block_index": start,
            "end_block_index": end, "cited_text": "".join(b.text for b in docs[0].blocks[start:end])}


def answer_of(docs, text, start, end):
    return parse_reply({"stop_reason": "end_turn",
                        "content": [{"type": "text", "text": text, "citations": [cite(docs, start, end)]}]}, docs)


def test_names_and_literals_are_read_from_the_unit() -> None:
    text = 'Raise `HTTPException` with `status_code=404` and detail "Item not found".'
    assert names(text) == ["HTTPException", "status_code=404"]
    found = literals(text)
    assert "404" in found and "Item not found" in found
    assert lexical_near_miss("Use HTTPException in a dependency too.", ["HTTPException"], ["404"])
    assert not lexical_near_miss("HTTPException(status_code=404)", ["HTTPException"], ["404"])
    assert not lexical_near_miss("Status 4040 is not 404.", ["HTTPException"], ["404"])  # no shared name
    assert topical_near_miss("Install a handler.", ["HTTPException"])
    assert not topical_near_miss("Pass headers to HTTPException.", ["HTTPException"])
    assert "doesn't" not in "".join(literals("It doesn't fail, it's fine."))  # apostrophes aren't quotes


def test_a_lexical_swap_comes_from_the_rest_of_the_same_section(world) -> None:
    doc, index, corpus, docs = world
    ans = answer_of(docs, "Raise `HTTPException` with `status_code=404` to return a not-found error.", 0, 1)
    found = candidates_for("swap-lexical", "hhr-4.8k-med", "q1", "r" + "a" * 56 + "s00", ans, docs, GRADES,
                           TARGET, corpus, index)
    assert len(found) == 1
    c = found[0]
    assert c.section == TARGET and c.texts == ("Use `HTTPException` in a dependency too.",)
    assert c.unit == "U1" and c.claim.startswith("Raise")


def test_a_topical_swap_shares_no_name_and_falls_back_to_a_lower_graded_section(world) -> None:
    doc, index, corpus, docs = world
    ans = answer_of(docs, "Raise `HTTPException` with `status_code=404` to return a not-found error.", 0, 1)
    c = candidates_for("swap-topical", "hhr-4.8k-med", "q1", "r" + "a" * 56 + "s00", ans, docs, GRADES, TARGET,
                       corpus, index)[0]
    assert c.section == TARGET and c.texts == ("The `detail` you pass becomes the JSON body.",)
    # With `detail` among its names, nothing else in the section is free of them: the swap comes from a
    # grade-0 or grade-1 section on the same page.
    ans = answer_of(docs, "Raise `HTTPException`; its `detail` becomes the body.", 0, 2)
    c = candidates_for("swap-topical", "hhr-4.8k-med", "q1", "r" + "a" * 56 + "s00", ans, docs, GRADES, TARGET,
                       corpus, index)[0]
    assert c.section in ("tutorial/errors#handlers", "tutorial/errors#errors")
    assert not any(n in c.texts[0] for n in ("HTTPException", "detail"))


def test_a_cross_section_plant_takes_the_other_grade_2_block_sharing_most_names(world) -> None:
    doc, index, corpus, docs = world
    ans = answer_of(docs, "Raise `HTTPException` with `status_code=404` to return a not-found error.", 0, 1)
    c = candidates_for("plant-cross", "base-4.8k-med", "q1", "r" + "b" * 56 + "s00", ans, docs, GRADES, TARGET,
                       corpus, index)[0]
    assert c.section == "tutorial/errors#custom-headers" and "HTTPException" in c.texts[0]


def test_a_shape_plant_recuts_the_same_characters_at_blank_lines(world) -> None:
    doc, index, corpus, docs = world
    code_block = next(n for n, b in enumerate(docs[0].blocks) if b.is_code)
    ans = answer_of(docs, "The `detail` you pass becomes the JSON body.", code_block - 1, code_block + 1)
    c = candidates_for("plant-shape", "hhr-4.8k-med", "q1", "r" + "c" * 56 + "s00", ans, docs, GRADES, TARGET,
                       corpus, index)[0]
    original = excerpts(ans, docs)["U1.E1"]
    assert len(c.texts) == 3  # the paragraph, then the code fence cut at its blank line
    assert "".join("".join(t.split()) for t in c.texts) == "".join(original.split())
    assert baseline_pieces("a\n\n b \n\nc", 10) == [(10, 11), (14, 15), (18, 19)]


def test_the_planted_request_changes_only_that_units_excerpts(world) -> None:
    doc, index, corpus, docs = world
    text = "Raise `HTTPException` with `status_code=404` to return a not-found error. The `detail` you pass becomes the JSON body."
    reply = {"stop_reason": "end_turn", "content": [
        {"type": "text", "text": text[:text.index(" The")], "citations": [cite(docs, 0, 1)]},
        {"type": "text", "text": text[text.index(" The"):], "citations": [cite(docs, 1, 2)]}]}
    ans = parse_reply(reply, docs)
    c = candidates_for("swap-lexical", "hhr-4.8k-med", "q1", "r" + "a" * 56 + "s00", ans, docs, GRADES, TARGET,
                       corpus, index)
    swap = next(x for x in c if x.unit == "U1")
    plain = excerpts(ans, docs)
    planted = excerpts(ans, docs, planted_excerpts(swap))
    assert planted["U2.E1"] == plain["U2.E1"] and planted["U1.E1"] != plain["U1.E1"]
    body = support_body("q?", ans, docs, 1000, planted=planted_excerpts(swap))
    assert "Use `HTTPException` in a dependency too." in body["messages"][0]["content"][0]["text"]
    with pytest.raises(ValueError, match="doesn't have"):
        excerpts(ans, docs, {"U9": ["x"]})


def make(kind, arm, question, order, unit="U1") -> Candidate:
    return Candidate(kind, arm, question, f"r{question}", unit, "claim", ("t",), (("d", 0, 1),), None, order)


def test_the_draw_keeps_one_per_question_per_kind_and_arm_in_hash_order() -> None:
    cands = [make("swap-lexical", "hhr", "q1", "b"), make("swap-lexical", "hhr", "q1", "a", "U2"),
             make("swap-lexical", "hhr", "q2", "c"), make("swap-lexical", "hhr", "q3", "d"),
             make("swap-lexical", "base", "q1", "e")]
    got = draw(cands, per_kind=2)
    assert [(c.arm, c.question, c.unit) for c in got] == [("base", "q1", "U1"), ("hhr", "q1", "U2"),
                                                          ("hhr", "q2", "U1")]


def test_the_gate_takes_the_first_confirmed_candidates_whose_original_was_supported() -> None:
    a, b, c = make("swap-lexical", "hhr", "q1", "a"), make("swap-lexical", "hhr", "q2", "b"), \
        make("swap-lexical", "hhr", "q3", "c")
    shape = make("plant-shape", "hhr", "q4", "d")
    confirmed = {f"{a.answered}:{a.name}": True, f"{b.answered}:{b.name}": False, f"{c.answered}:{c.name}": True}
    supported = {(x.answered, x.unit): True for x in (a, b, c, shape)}
    supported[(a.answered, a.unit)] = False
    got = gate([a, b, c, shape], confirmed, supported, {("swap-lexical", "hhr"): 1, ("plant-shape", "hhr"): 1})
    assert got == [shape, c]  # a's original wasn't supported, b's block wasn't confirmed; shapes need no worksheet


def test_only_lexical_swaps_need_a_backticked_name(world) -> None:
    doc, index, corpus, docs = world
    ans = answer_of(docs, "Raise the error and add response headers to it.", 0, 1)  # no backticked name
    grab = lambda kind: candidates_for(kind, "hhr-4.8k-med", "q1", "r" + "d" * 56 + "s00", ans, docs, GRADES,  # noqa: E731
                                       TARGET, corpus, index)
    assert grab("swap-lexical") == []
    assert grab("swap-topical")
    (cross,) = grab("plant-cross")  # chosen by shared words: "response", "headers"
    assert cross.section == "tutorial/errors#custom-headers"


def test_a_shape_plant_recuts_every_multi_block_citation_in_the_answer(world) -> None:
    doc, index, corpus, docs = world
    code_block = next(n for n, b in enumerate(docs[0].blocks) if b.is_code)
    first = "The `detail` you pass becomes the JSON body."
    second = " Raise `HTTPException` with `status_code=404` to return a not-found error."
    reply = {"stop_reason": "end_turn", "content": [
        {"type": "text", "text": first, "citations": [cite(docs, code_block - 1, code_block + 1)]},
        {"type": "text", "text": second, "citations": [cite(docs, 0, 2)]}]}
    ans = parse_reply(reply, docs)
    (c1, c2) = sorted(candidates_for("plant-shape", "hhr-4.8k-med", "q1", "r" + "e" * 56 + "s00", ans, docs, GRADES,
                                     TARGET, corpus, index), key=lambda c: c.unit)
    assert c1.unit == "U1" and [u for u, _ in c1.others] == ["U2"]
    planted = planted_excerpts(c1)
    assert set(planted) == {"U1", "U2"} and len(planted["U2"]) == 2  # U2's two-block citation, re-cut
    assert all(c.context and not any(t in c.context for t in planted[c.unit]) for c in (c1,))


def test_a_shape_plant_keeps_a_one_block_citation_as_it_was(world) -> None:
    doc, index, corpus, docs = world
    code_block = next(n for n, b in enumerate(docs[0].blocks) if b.is_code)
    text = "The `detail` you pass becomes the JSON body."
    reply = {"stop_reason": "end_turn", "content": [{"type": "text", "text": text, "citations": [
        cite(docs, code_block - 1, code_block + 1), cite(docs, code_block, code_block + 1)]}]}
    ans = parse_reply(reply, docs)
    (c,) = candidates_for("plant-shape", "hhr-4.8k-med", "q1", "r" + "f" * 56 + "s00", ans, docs, GRADES, TARGET,
                          corpus, index)
    # The two-block citation is re-cut (paragraph, and the fence split at its blank line); the one-block citation
    # of the same fence stays one excerpt, whole.
    assert len(c.texts) == 3 + 1 and c.texts[-1] == docs[0].blocks[code_block].text
