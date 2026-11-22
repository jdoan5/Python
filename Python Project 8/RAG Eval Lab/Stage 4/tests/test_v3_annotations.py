"""The v3 annotation rules applied by code: pool and S batches, references, answer keys."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))

import v3_annotations as v3  # noqa: E402
from raglab.sections import SectionIndex  # noqa: E402
from tests.conftest import make_doc  # noqa: E402

PAGE = """\
# Errors { #errors }

Use `HTTPException` to return errors.

## Raise one { #raise-one }

Raise it with a status code, such as 404.

## Handlers { #handlers }

Add a handler with `@app.exception_handler` and raise it with a status code.
"""


@pytest.fixture
def corpus():
    doc = make_doc(PAGE, "tutorial/errors")
    return {doc.doc_id: doc}, SectionIndex([doc])


def test_worksheets_never_go_inside_the_project(tmp_path: Path) -> None:
    with pytest.raises(v3.AnnotationError, match="outside the project"):
        v3.outside_project(v3.STAGE / "pilot" / "sheets")
    assert v3.outside_project(tmp_path) == tmp_path.resolve()


def test_pool_and_s_batches_must_be_complete_and_valid() -> None:
    expected = {("p01", "a#x"), ("p01", "a#y")}
    assert v3.check_pool_batch(expected, [{"question": "p01", "location": "a#x", "grade": 2},
                                          {"question": "p01", "location": "a#y", "grade": 0}]) == []
    bad = v3.check_pool_batch(expected, [{"question": "p01", "location": "a#x", "grade": True},
                                         {"question": "p01", "location": "a#z", "grade": 1}])
    assert any("bad grade" in p for p in bad) and any("unexpected" in p for p in bad) and any("missing" in p for p in bad)
    blocks = {"p01|aa": ["Raise it with a status code.", "The client gets JSON."], "p02|bb": ["Unrelated."]}
    ok = [{"id": "p01|aa", "answers": "yes", "quote": "Raise it with a  status code."}, {"id": "p02|bb", "answers": "no"}]
    assert v3.check_s_batch(["p01|aa", "p02|bb"], ok, blocks) == []
    made_up = [{"id": "p01|aa", "answers": "partly", "quote": "Raise it twice."}, {"id": "p02|bb", "answers": "no"}]
    assert v3.check_s_batch(["p01|aa", "p02|bb"], made_up, blocks)[0].startswith("p01|aa: the quote isn't inside")
    spanning = [{"id": "p01|aa", "answers": "yes", "quote": "status code. The client"}, {"id": "p02|bb", "answers": "no"}]
    assert v3.check_s_batch(["p01|aa", "p02|bb"], spanning, blocks)  # across two blocks: refused


def test_s_evidence_leaves_out_heading_lines(corpus) -> None:
    from raglab.blocks import window_documents

    by_id, index = corpus
    doc = by_id["tutorial/errors"]
    wdocs = window_documents([(doc.doc_id, 0, len(doc.text))], by_id, index)
    texts = v3.evidence_blocks(wdocs)
    headings = [b.text for d in wdocs for b in d.blocks if b.section is None]
    assert headings and not set(headings) & set(texts)  # a heading-only block is never evidence
    answers = [{"id": "x", "answers": "yes", "quote": headings[0].strip()}]
    assert v3.check_s_batch(["x"], answers, {"x": texts})


def test_the_reference_keeps_the_target_and_the_best_delivered_sections_and_shows_them_in_document_order(corpus):
    by_id, index = corpus
    labels = {"tutorial/errors#raise-one": 2, "tutorial/errors#handlers": 2, "tutorial/errors": 1}
    delivered = {"tutorial/errors#handlers": (2, 0.9)}
    ref, removed = v3.reference_for(labels, "tutorial/errors#raise-one", delivered, by_id, index)
    assert [s["location"] for s in ref] == ["tutorial/errors#raise-one", "tutorial/errors#handlers"]  # document order
    assert ref[1]["text"].startswith("## Handlers") and ref[1]["heading"] == "Errors > Handlers" and removed == []


def test_under_the_cap_the_target_survives_and_fragment_touches_go_first(corpus, monkeypatch):
    by_id, index = corpus
    labels = {"tutorial/errors#raise-one": 2, "tutorial/errors#handlers": 2, "tutorial/errors#errors": 2}
    sizes = {s["location"]: len(s["text"]) for s in v3.reference_for(labels, None, {}, by_id, index)[0]}
    # A cap with room for the target and one more section only.
    monkeypatch.setattr(v3.params, "REFERENCE_CAP_CHARS", sizes["tutorial/errors#raise-one"]
                        + sizes["tutorial/errors#handlers"])
    # #errors is delivered by more windows (2) but only by 3% fragments; #handlers by one window, 80% of it.
    delivered = {"tutorial/errors#errors": (2, 0.03), "tutorial/errors#handlers": (1, 0.8)}
    ref, removed = v3.reference_for(labels, "tutorial/errors#raise-one", delivered, by_id, index)
    kept = [s["location"] for s in ref]
    assert "tutorial/errors#raise-one" in kept  # the target is never dropped
    # A bigger share outranks more windows: #handlers is kept; #errors, touched only by fragments, is reported.
    assert kept == ["tutorial/errors#raise-one", "tutorial/errors#handlers"]
    assert removed == [{"location": "tutorial/errors#errors", "windows": 2, "largest_share": 0.03}]


def test_an_essential_nugget_must_be_quoted_from_every_reference_section(corpus) -> None:
    by_id, index = corpus
    ref, _ = v3.reference_for({"tutorial/errors#raise-one": 2, "tutorial/errors#handlers": 2}, None, {}, by_id, index)
    good = {"nuggets": [
        {"text": "Raise HTTPException with a status code", "essential": True, "alternatives": [],
         "quotes": [{"section": "R1", "quote": "Raise it with a status code"}, {"section": "R2", "quote": "raise it with a status code"}]},
        {"text": "A handler uses @app.exception_handler", "essential": False, "alternatives": [],
         "quotes": [{"section": "R2", "quote": "Add a handler with `@app.exception_handler`"}]}]}
    assert v3.check_key(good, ref) == []
    one_section = {"nuggets": [good["nuggets"][1] | {"essential": True}, good["nuggets"][1]]}
    assert any("isn't quoted from every section" in p for p in v3.check_key(one_section, ref))
    invented = {"nuggets": [good["nuggets"][0], {**good["nuggets"][1], "quotes": [{"section": "R2", "quote": "use middleware"}]}]}
    assert any("isn't in R2" in p for p in v3.check_key(invented, ref))
    too_few = {"nuggets": [good["nuggets"][0]]}
    assert any("1 nuggets" in p for p in v3.check_key(too_few, ref))


def test_the_swap_worksheet_is_blind_and_its_answers_are_checked() -> None:
    import json

    from raglab.planted import Candidate

    check_swaps, swap_entries = v3.check_swaps, v3.swap_entries

    def cand(kind, unit):
        return Candidate(kind, "hhr-4.8k-med", "p01", "r" + "a" * 56 + "s00", unit, f"claim {unit}",
                         (f"passage {unit}",), (("d", 0, 1),), None, unit)

    cands = [cand("swap-lexical", "U1"), cand("plant-cross", "U2"), cand("plant-shape", "U3")]
    entries, key_of = swap_entries(cands)
    assert len(entries) == 2  # shape plants need no confirmation
    text = json.dumps(entries)
    assert "swap" not in text and "plant" not in text and "hhr" not in text  # nothing but claim and passage
    assert set(key_of) == {e["id"] for e in entries}
    good = [{"id": e["id"], "states": "no", "quote": "", "others": "no"} for e in entries]
    assert all("other_passages" in e for e in entries)
    assert check_swaps(entries, good) == []
    bad = [{"id": entries[0]["id"], "states": "yes", "quote": "not in it", "others": "no"}, good[1]]
    assert "isn't in the passage" in check_swaps(entries, bad)[0]
    assert "ids don't match" in check_swaps(entries, good[:1])[0]
    assert "others must be" in check_swaps(entries, [{**good[0], "others": "maybe"}, good[1]])[0]


def test_the_key_fallback_trims_from_the_end_of_the_priority_order_never_the_target(corpus) -> None:
    by_id, index = corpus
    labels = {"tutorial/errors#raise-one": 2, "tutorial/errors#handlers": 2, "tutorial/errors#errors": 2}
    delivered = {"tutorial/errors#handlers": (2, 0.9)}
    full, _ = v3.reference_for(labels, "tutorial/errors#raise-one", delivered, by_id, index)
    assert len(full) == 3
    kept, removed = v3.reference_for(labels, "tutorial/errors#raise-one", delivered, by_id, index, trim=1)
    assert [s["location"] for s in kept] == ["tutorial/errors#raise-one", "tutorial/errors#handlers"]
    assert removed == [{"location": "tutorial/errors#errors", "trimmed": True}]  # undelivered: lowest priority
    with pytest.raises(v3.AnnotationError, match="besides the target"):
        v3.reference_for(labels, "tutorial/errors#raise-one", delivered, by_id, index, trim=3)
