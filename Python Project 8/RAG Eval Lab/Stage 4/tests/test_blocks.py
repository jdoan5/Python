"""Windows as citable documents, and citations mapped back to sections."""

from __future__ import annotations

import pytest

from raglab.blocks import Citation, CitationError, citation_intact, cited_blocks, squash, window_documents
from raglab.sections import SectionIndex
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
def doc():
    return make_doc(PAGE, "tutorial/errors")


def test_a_chunk_becomes_one_document_of_paragraph_and_code_blocks(doc) -> None:
    index = SectionIndex([doc])
    start = doc.text.index("## Raise one")
    (wd,) = window_documents([(doc.doc_id, start, len(doc.text.rstrip()))], {doc.doc_id: doc}, index)
    texts = [b.text for b in wd.blocks]
    assert texts[0] == "## Raise one" and texts[1] == "Raise it with a status code:"
    assert texts[2].startswith("```python") and texts[2].endswith("```") and wd.blocks[2].is_code
    assert wd.blocks[0].section is None  # a heading line is no section's body
    assert {b.section for b in wd.blocks[1:]} == {"tutorial/errors#raise-one"}
    assert wd.title == "Errors > Raise one"
    for b in wd.blocks:
        assert doc.text[b.start:b.end] == b.text
    api = wd.api_document()
    assert api["citations"] == {"enabled": True} and api["source"]["type"] == "content"
    assert [c["text"] for c in api["source"]["content"]] == texts


def test_citations_map_to_blocks_and_must_match_their_text(doc) -> None:
    index = SectionIndex([doc])
    docs = window_documents([(doc.doc_id, 0, len(doc.text.rstrip()))], {doc.doc_id: doc}, index)
    blocks = docs[0].blocks
    whole = Citation(0, 1, 3, blocks[1].text + "\n\n" + blocks[2].text)
    assert [b.text for b in cited_blocks(whole, docs)] == [blocks[1].text, blocks[2].text]
    assert citation_intact(whole, docs)
    assert not citation_intact(Citation(0, 1, 2, "Something the page never says."), docs)
    with pytest.raises(CitationError):
        cited_blocks(Citation(1, 0, 1, ""), docs)
    with pytest.raises(CitationError):
        cited_blocks(Citation(0, 2, 2, ""), docs)  # an empty range
    with pytest.raises(CitationError):
        Citation.from_api({"type": "char_location", "start_char_index": 0})
    c = Citation.from_api({"type": "content_block_location", "document_index": 0, "start_block_index": 1,
                           "end_block_index": 2, "cited_text": blocks[1].text, "document_title": "x"})
    assert c == Citation(0, 1, 2, blocks[1].text)


def test_whitespace_never_decides_a_match() -> None:
    assert squash(" a\n\tb  c ") == "abc"
