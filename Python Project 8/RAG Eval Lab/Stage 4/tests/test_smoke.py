"""The smoke test's checks: the join rule, usage fields, and its pass/fail verdict. No API."""

from __future__ import annotations

import pytest

from raglab.blocks import Citation, window_documents
from raglab.sections import SectionIndex
from raglab.smoke import answer_check, join_rule, verdict
from tests.conftest import make_doc

PAGE = """\
# Errors { #errors }

Use `HTTPException` to return errors.

It carries a status code.

```python
raise HTTPException(status_code=404)
```
"""


@pytest.fixture
def docs():
    doc = make_doc(PAGE, "tutorial/errors")
    return window_documents([(doc.doc_id, 0, len(doc.text.rstrip()))], {doc.doc_id: doc}, SectionIndex([doc]))


def citation(docs, start, end, sep="") -> Citation:
    return Citation(0, start, end, sep.join(b.text for b in docs[0].blocks[start:end]))


def test_the_join_rule_is_the_one_that_rebuilds_cited_text_exactly(docs) -> None:
    assert len(docs[0].blocks) >= 3
    assert join_rule(citation(docs, 1, 3), docs) == "concatenated"
    assert join_rule(citation(docs, 1, 3, "\n"), docs) == "newline"
    stripped = Citation(0, 1, 3, "\n\n".join(b.text.rstrip() for b in docs[0].blocks[1:3]))
    assert join_rule(stripped, docs) in ("blank line", "blank line (stripped)")
    loose = Citation(0, 1, 3, "  ".join(" ".join(b.text.split()) for b in docs[0].blocks[1:3]))
    assert join_rule(loose, docs) == "whitespace only"
    assert join_rule(Citation(0, 1, 2, "something else"), docs) == "no match"


def message(docs, cites, usage=None) -> dict:
    raw = [{"type": "content_block_location", "document_index": c.document_index, "start_block_index": c.start_block,
            "end_block_index": c.end_block, "cited_text": c.cited_text, "document_title": None} for c in cites]
    return {"model": "claude-haiku-5-5", "stop_reason": "end_turn",
            "usage": usage or {"input_tokens": 900, "output_tokens": 80, "inference_geo": "global",
                               "service_tier": "standard"},
            "content": [{"type": "text", "text": "Raise HTTPException with a status code.", "citations": raw}]}


def test_a_clean_answer_passes_and_names_its_join_rule(docs) -> None:
    check = answer_check(message(docs, [citation(docs, 1, 3)]), docs)
    assert check["citations"] == 1 and check["citations_failed"] == 0
    assert check["join_rules"] == {"concatenated": 1}
    assert check["unpriced_usage_fields"] == [] and check["inference_geo"] == "global"
    v = verdict([check])
    assert v["passed"] and v["join_rule"] == "concatenated" and v["inference_geo"] == ["global"]


def test_a_broken_citation_or_an_unknown_usage_field_fails_the_smoke_test(docs) -> None:
    bad = answer_check(message(docs, [Citation(0, 1, 2, "not the block's text")]), docs)
    assert bad["citations_failed"] == 1
    assert not verdict([bad])["passed"]
    odd = answer_check(message(docs, [citation(docs, 1, 2)],
                               {"input_tokens": 900, "output_tokens": 80, "web_fetch_requests": 1}), docs)
    v = verdict([odd])
    assert not v["passed"] and "web_fetch_requests" in v["problems"][0]


def test_two_join_rules_leave_the_rule_unrecorded(docs) -> None:
    a = answer_check(message(docs, [citation(docs, 1, 3)]), docs)
    b = answer_check(message(docs, [citation(docs, 1, 3, "\n")]), docs)
    v = verdict([a, b])
    assert v["passed"] and v["join_rule"] is None and v["join_rules"] == {"concatenated": 1, "newline": 1}


def test_us_only_inference_fails_the_smoke_test_with_what_to_do(docs) -> None:
    usage = {"input_tokens": 900, "output_tokens": 80, "inference_geo": "us", "service_tier": "standard"}
    v = verdict([answer_check(message(docs, [citation(docs, 1, 3)], usage), docs)])
    assert not v["passed"] and "global" in v["problems"][0] and v["inference_geo"] == ["us"]


def test_a_one_block_citation_never_decides_the_join_rule(docs) -> None:
    one = answer_check(message(docs, [citation(docs, 1, 2)]), docs)
    assert one["join_rules"] == {} and one["single_block_citations"] == 1
    two = answer_check(message(docs, [citation(docs, 1, 3, "\n")]), docs)
    v = verdict([one, two])
    assert v["join_rule"] == "newline" and v["single_block_citations"] == 1
