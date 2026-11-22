"""Replies cut into units, citations checked and merged, abstentions and statuses; requests and windows."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from raglab import params
from raglab.blocks import window_documents
from raglab.generate import check_generation_body, generation_body
from raglab.sections import SectionIndex
from raglab.units import code_gate, is_abstention, looks_like_abstention, parse_reply
from raglab.windows import CONDITIONS, Windows, parse_condition
from tests.conftest import make_doc

PAGE = """\
# Errors { #errors }

Use `HTTPException` to return errors.

## Raise one { #raise-one }

Raise it with a status code:

```python
raise HTTPException(status_code=404, detail="Item not found")
```

The client gets a JSON body.
"""


@pytest.fixture
def setup():
    doc = make_doc(PAGE, "tutorial/errors")
    index = SectionIndex([doc])
    docs = window_documents([(doc.doc_id, 0, len(doc.text.rstrip()))], {doc.doc_id: doc}, index)
    return doc, index, docs


def cite(docs, d: int, start: int, end: int) -> dict:
    return {"type": "content_block_location", "document_index": d, "start_block_index": start,
            "end_block_index": end, "document_title": docs[d].title,
            "cited_text": "".join(b.text for b in docs[d].blocks[start:end])}


def message(content, stop="end_turn", **extra) -> dict:
    return {"content": content, "stop_reason": stop, **extra}


def test_a_sentence_split_across_api_blocks_is_one_unit_with_every_citation(setup) -> None:
    _, _, docs = setup
    reply = message([
        {"type": "thinking", "thinking": "", "signature": "x"},
        {"type": "text", "text": "Raise `HTTPException` with ", "citations": None},
        {"type": "text", "text": "a status code", "citations": [cite(docs, 0, 3, 4)]},
        {"type": "text", "text": " and the client gets JSON.", "citations": [cite(docs, 0, 5, 6)]},
    ])
    a = parse_reply(reply, docs)
    assert a.status == "ok" and a.text == "Raise `HTTPException` with a status code and the client gets JSON."
    (u,) = a.units
    assert len(docs[0].blocks) == 6
    assert [(c.start_block, c.end_block) for c in u.citations] == [(3, 4), (5, 6)]
    assert (a.citations_total, a.citations_failed) == (2, 0)


def test_code_fences_are_units_and_sentences_are_cut_with_care(setup) -> None:
    _, _, docs = setup
    text = ("Use `app.get()` for routes, e.g. `@app.get(\"/\")`. Then raise it.\n"
            "1. Import it.\n"
            "```python\nraise HTTPException(status_code=404)\n```\n"
            "That's all. \"Done.\" Next step.")
    a = parse_reply(message([{"type": "text", "text": text, "citations": [cite(docs, 0, 5, 6)]}]), docs)
    texts = [u.text for u in a.units]
    assert texts == ["Use `app.get()` for routes, e.g. `@app.get(\"/\")`.", "Then raise it.", "1. Import it.",
                     "```python\nraise HTTPException(status_code=404)\n```", "That's all.", "\"Done.\"", "Next step."]
    assert [u.is_code for u in a.units] == [False, False, False, True, False, False, False]
    assert all(u.citations for u in a.units)  # one cited block covers the whole reply


def test_a_citation_that_doesnt_match_its_blocks_is_dropped_and_counted(setup) -> None:
    _, _, docs = setup
    bad = cite(docs, 0, 1, 2) | {"cited_text": "Something the page never says."}
    out_of_range = cite(docs, 0, 0, 1) | {"document_index": 7}
    a = parse_reply(message([{"type": "text", "text": "Use it.", "citations": [bad, out_of_range]}]), docs)
    assert (a.citations_total, a.citations_failed) == (2, 2) and not a.units[0].citations
    assert len(a.failures) == 2


def test_status_and_the_abstention_flag(setup) -> None:
    _, _, docs = setup
    s = params.ABSTENTION_SENTENCE
    assert parse_reply(message([{"type": "text", "text": s}]), docs).abstained
    assert is_abstention("  the provided documentation does not answer this question ")
    assert not is_abstention(s + " But here is a guess.")
    assert looks_like_abstention("The documents don't mention WebSockets at all.")
    assert not looks_like_abstention("Use `HTTPException` to return a 404.")
    refused = parse_reply(message([], stop="refusal", stop_details={"category": "cyber"}), docs)
    assert (refused.status, refused.refusal_category) == ("refused", "cyber")
    assert parse_reply(message([{"type": "text", "text": "Raise"}], stop="max_tokens"), docs).status == "truncated"
    assert parse_reply(message([{"type": "thinking", "thinking": ""}]), docs).status == "empty"


def test_the_code_gate_wants_every_substantial_unit_cited(setup) -> None:
    _, _, docs = setup
    long_uncited = "This sentence is long enough to need a citation of its own here."
    a = parse_reply(message([{"type": "text", "text": "Short. " + long_uncited}]), docs)
    assert not code_gate(a)
    b = parse_reply(message([{"type": "text", "text": "## Heading line that is long enough to count\nShort."}]), docs)
    assert code_gate(b)


# ---------------------------------------------------------------------------------------------------- requests


def test_every_condition_parses_and_runs_match_the_plan() -> None:
    assert set(CONDITIONS) == set(params.CONDITIONS)
    hhr, off, closed, notitle = (parse_condition(n) for n in
                                 ("hhr-4.8k-med", "hhr-4.8k-off", "closed-med", "base-4.8k-med-notitle"))
    assert (hhr.pipeline, hhr.budget, hhr.effort, hhr.thinking, hhr.titles, hhr.runs) == \
        ("hhr", 4800, "medium", "adaptive", True, 3)
    assert (off.effort, off.thinking) == ("medium", "disabled")
    assert closed.closed and closed.budget is None
    assert not notitle.titles and notitle.pipeline == "base"
    assert parse_condition("hhr-9.6k-med").budget == 9600 and parse_condition("hhr-2.4k-med").budget == 2400
    with pytest.raises(ValueError):
        parse_condition("sonnet-4.8k-med")


def test_generation_bodies_send_exactly_what_the_plan_says(setup) -> None:
    _, _, docs = setup
    body = generation_body(CONDITIONS["hhr-4.8k-med"], "How do I return a 404?", docs, 16000, "ANSWER", "CLOSED")
    assert body["model"] == "claude-haiku-5-5" and body["system"] == "ANSWER"
    assert body["thinking"] == {"type": "adaptive"} and body["output_config"] == {"effort": "medium"}
    content = body["messages"][0]["content"]
    assert content[-1] == {"type": "text", "text": "How do I return a 404?"}
    assert all(c["citations"] == {"enabled": True} and "title" in c for c in content[:-1])
    untitled = generation_body(CONDITIONS["base-4.8k-med-notitle"], "q", docs, 16000, "ANSWER", "CLOSED")
    assert all("title" not in c for c in untitled["messages"][0]["content"][:-1])
    closed = generation_body(CONDITIONS["closed-med"], "q", [], 8000, "ANSWER", "CLOSED")
    assert closed["system"] == "CLOSED" and closed["messages"][0]["content"] == [{"type": "text", "text": "q"}]
    off = generation_body(CONDITIONS["hhr-4.8k-off"], "q", docs, 8000, "ANSWER", "CLOSED")
    assert off["thinking"] == {"type": "disabled"}
    for key in ("temperature", "top_p", "top_k", "fallbacks"):
        with pytest.raises(ValueError, match="never sent"):
            check_generation_body(body | {key: 1})
    cited_user_text = {**body, "messages": [{"role": "user", "content": [
        {"type": "text", "text": "x", "citations": [{"type": "char_location"}]}]}]}
    with pytest.raises(ValueError, match="2026-10-21"):
        check_generation_body(cited_user_text)
    with pytest.raises(ValueError, match="high or below"):
        check_generation_body(body | {"thinking": {"type": "disabled"}, "output_config": {"effort": "xhigh"}})


# ---------------------------------------------------------------------------------------------------- windows


class FakeRetriever:
    def __init__(self, spans):
        self.chunks = spans

    def search(self, query, k):
        return [SimpleNamespace(chunk=SimpleNamespace(doc_id=d, start=s, end=e)) for d, s, e in self.chunks][:k]


def test_masked_and_oracle_windows_follow_the_plan(setup) -> None:
    doc, index, _ = setup
    t = doc.text
    intro = (doc.doc_id, t.index("Use `HTTP"), t.index("Use `HTTP") + len("Use `HTTPException` to return errors."))
    raise_ = (doc.doc_id, t.index("## Raise one"), len(t.rstrip()))
    ranked = [intro, raise_]
    from raglab.windows import PIPELINES
    w = Windows([doc], index, retrievers={PIPELINES["hhr"].name: FakeRetriever(ranked)})
    answering = frozenset({"tutorial/errors#raise-one"})
    assert w.window(CONDITIONS["hhr-4.8k-med"], "q") == ranked
    assert w.window(CONDITIONS["masked-4.8k-med"], "q", answering) == [intro]
    assert w.window(CONDITIONS["masked-4.8k-med"], "q", frozenset({"elsewhere#x"})) is None  # nothing to mask
    assert w.window(CONDITIONS["oracle-4.8k-med"], "q", answering, target="tutorial/errors#raise-one") == [raise_]
    assert w.window(CONDITIONS["closed-med"], "q") == []


def test_the_splitter_keeps_abbreviations_in_brackets_and_splits_before_inline_code(setup) -> None:
    _, _, docs = setup

    def units(text):
        return [u.text for u in parse_reply(message([{"type": "text", "text": text, "citations": None}]), docs).units]

    assert units("Use a framework (e.g. FastAPI) here.") == ["Use a framework (e.g. FastAPI) here."]
    assert units("Add a dependency. `Depends(get_db)` injects the session.") == \
        ["Add a dependency.", "`Depends(get_db)` injects the session."]
    assert units("Install it with pip. `pip install fastapi` does that.") == \
        ["Install it with pip.", "`pip install fastapi` does that."]
    assert units("Call `app.get(path)`. Then run it.") == ["Call `app.get(path)`.", "Then run it."]
    assert units("Use `x. Y` as is.") == ["Use `x. Y` as is."]  # never cut inside inline code


def test_a_list_number_after_a_bullet_or_emphasis_never_ends_a_unit(setup) -> None:
    _, _, docs = setup

    def units(text):
        return [u.text for u in parse_reply(message([{"type": "text", "text": text, "citations": None}]), docs).units]

    assert units("- 1. Install FastAPI first.") == ["- 1. Install FastAPI first."]
    assert units("**2.** Then run the server.") == ["**2.** Then run the server."]


def test_a_list_number_after_a_sentence_on_the_same_line_numbers_what_follows(setup) -> None:
    _, _, docs = setup

    def units(text):
        return [u.text for u in parse_reply(message([{"type": "text", "text": text, "citations": None}]), docs).units]

    assert units("1. First. 2. Second.") == ["1. First.", "2. Second."]
    assert units("Do this. 3. Then that.") == ["Do this.", "3. Then that."]
