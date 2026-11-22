"""Sealed stores carry no documentation text and rebuild exactly from the corpus."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from raglab.blocks import window_documents
from raglab.sections import SectionIndex
from raglab.seal import (
    check_unsealed_citation, read_sealed, seal_message, seal_quote, seal_record, unlisted_files, unseal_quote,
    write_sealed,
)
from raglab.store import Record
from tests.conftest import make_doc
from tests.test_units import PAGE, cite


def test_a_message_keeps_its_answer_but_not_the_docs_it_cites(tmp_path: Path) -> None:
    doc = make_doc(PAGE, "tutorial/errors")
    docs = window_documents([(doc.doc_id, 0, len(doc.text.rstrip()))], {doc.doc_id: doc}, SectionIndex([doc]))
    c = cite(docs, 0, 3, 5)
    message = {"id": "msg_1", "model": "claude-haiku-5-5", "stop_reason": "end_turn", "usage": {"input_tokens": 1},
               "content": [{"type": "thinking", "thinking": "", "signature": "opaque-bytes"},
                           {"type": "text", "text": "Raise it with a status code.", "citations": [c]}],
               "container": {"id": "not-allowlisted"}}
    sealed = seal_message(message)
    blob = json.dumps(sealed)
    assert "Raise it with a status code:" not in blob  # the cited block's text is gone
    assert docs[0].title not in blob and "opaque-bytes" not in blob and "container" not in sealed
    assert sealed["content"][1]["text"] == "Raise it with a status code."  # the answer stays
    blocks_text = "".join(b.text for b in docs[0].blocks[3:5])
    assert check_unsealed_citation(sealed["content"][1]["citations"][0], blocks_text)
    assert not check_unsealed_citation(sealed["content"][1]["citations"][0], "different text")


def test_quotes_that_hold_become_offsets_and_rebuild_exactly() -> None:
    source = "Raise it with a status code:\n\n```python\nraise HTTPException(status_code=404)\n```"
    held = seal_quote("raise   HTTPException(status_code=404)", source)
    assert held["held"] and "HTTPException" not in json.dumps(held)
    assert unseal_quote(held, source) == "raise HTTPException(status_code=404)"
    invented = seal_quote("It returns XML.", source)
    assert invented["held"] is False and "XML" not in json.dumps(invented) and invented["len"] == len("It returns XML.")
    assert unseal_quote(invented, source) is None
    with pytest.raises(ValueError, match="doesn't match"):
        unseal_quote(held, source.replace("404", "405"))


def test_sealed_files_are_deterministic_and_checked(tmp_path: Path) -> None:
    rec = Record(custom_id="r" + "a" * 56 + "s00", request_key="a" * 64, condition="hhr-4.8k-med",
                 body_sha256="b" * 64, phase="generation", outcome="succeeded",
                 response={"content": [{"type": "text", "text": "x"}], "model": "claude-haiku-5-5"}, error=None,
                 batch_id="msgbatch_1", request_id=None, received_at="2026-11-21T00:00:00+00:00")
    rows = [seal_record(rec, seal_message)]
    first = write_sealed(rows, tmp_path / "a" / "generation.jsonl.gz")
    second = write_sealed(rows, tmp_path / "b" / "generation.jsonl.gz")
    assert (tmp_path / "a" / "generation.jsonl.gz").read_bytes() == (tmp_path / "b" / "generation.jsonl.gz").read_bytes()
    assert first == second and first["records"] == 1
    assert read_sealed(tmp_path / "a" / "generation.jsonl.gz", first)[0]["condition"] == "hhr-4.8k-med"
    with pytest.raises(ValueError, match="doesn't match"):
        read_sealed(tmp_path / "a" / "generation.jsonl.gz", {**first, "jsonl_sha256": "0" * 64})
    (tmp_path / "a" / "generation 2.jsonl.gz").write_bytes(b"icloud copy")
    assert unlisted_files(tmp_path / "a", ["generation.jsonl.gz"]) == ["generation 2.jsonl.gz"]
