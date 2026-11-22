from __future__ import annotations

import pytest

from conftest import make_doc
from raglab.corpus import Document
from raglab.chunking import (
    CHUNKERS,
    FixedChunker,
    HeadingChunker,
    ParagraphChunker,
    chunk_corpus,
    chunker_config,
    get_chunker,
)

LONG_SECTION = "\n\n".join(f"Paragraph {i} " + "word " * 40 for i in range(12))  # ~2.9k chars

RICH = f"""\
Preamble before any heading.

# Guide {{ #guide }}

Intro paragraph one.

Intro paragraph two with `inline code`.

## Empty Parent {{ #empty-parent }}

### Child {{ #child }}

Child body.

```python
# this is a comment, not a heading

def f():
    return 1
```

## Long {{ #long }}

{LONG_SECTION}
"""


@pytest.fixture
def rich_doc():
    return make_doc(RICH, doc_id="guide")


# ---------------------------------------------------------------- invariants (every strategy)

@pytest.mark.parametrize("name", sorted(CHUNKERS))
def test_every_chunk_is_the_exact_trimmed_span_it_claims(rich_doc, name):
    chunks = get_chunker(name).chunk(rich_doc)
    assert chunks
    for c in chunks:
        assert rich_doc.text[c.start:c.end] == c.text
        assert c.text and c.text == c.text.strip()


RICH_HEADINGS = [  # (heading line, anchor, path) — located by text, independent of Document.sections
    ("# Guide", "guide", ("Guide",)),
    ("## Empty Parent", "empty-parent", ("Guide", "Empty Parent")),
    ("### Child", "child", ("Guide", "Empty Parent", "Child")),
    ("## Long", "long", ("Guide", "Long")),
]


def expected_section(text: str, start: int) -> tuple[str, tuple[str, ...]]:
    anchor, path = "", ("Guide",)  # the preamble, before the first heading
    for line, a, p in RICH_HEADINGS:
        if text.index("\n" + line + "\n") + 1 <= start:
            anchor, path = a, p
    return anchor, path


@pytest.mark.parametrize("name", sorted(CHUNKERS))
def test_chunks_carry_the_section_they_start_in(rich_doc, name):
    chunks = get_chunker(name).chunk(rich_doc)
    for c in chunks:
        assert (c.anchor, c.heading_path) == expected_section(rich_doc.text, c.start)
    # Not vacuous: chunks start in more than one section, so a lookup that always
    # returned the first section would fail the check above.
    assert len({c.anchor for c in chunks}) >= 2


@pytest.mark.parametrize("name", sorted(CHUNKERS))
def test_chunk_ids_are_unique_and_namespaced(rich_doc, name):
    chunks = get_chunker(name).chunk(rich_doc)
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    assert all(c.chunk_id.startswith(f"guide::{name}-") for c in chunks)


def test_location_is_doc_and_anchor_and_preamble_has_no_anchor(rich_doc):
    chunks = HeadingChunker().chunk(rich_doc)
    assert chunks[0].location == "guide"  # the preamble
    assert chunks[1].location == "guide#guide"


def test_index_text_prefixes_the_breadcrumb_only_when_asked(rich_doc):
    child = next(c for c in HeadingChunker().chunk(rich_doc) if c.anchor == "child")
    assert child.index_text(with_context=True).startswith("Guide > Empty Parent > Child\n")
    assert child.index_text(with_context=False) == child.text


# ---------------------------------------------------------------- paragraph (the baseline)

def test_paragraph_chunks_match_rag_mini_chat_exactly(rich_doc):
    # RAG Mini Chat: [c.strip() for c in text.split("\n\n") if c.strip()]
    expected = [p.strip() for p in rich_doc.text.split("\n\n") if p.strip()]
    assert [c.text for c in ParagraphChunker().chunk(rich_doc)] == expected


def test_paragraph_chunking_splits_code_at_blank_lines(rich_doc):
    # The baseline's known weakness, kept on purpose: a function is cut from its comment.
    texts = [c.text for c in ParagraphChunker().chunk(rich_doc)]
    assert "```python\n# this is a comment, not a heading" in texts


# ---------------------------------------------------------------- fixed windows

def test_fixed_windows_respect_size_overlap_and_word_boundaries(rich_doc):
    chunker = FixedChunker(size=300, overlap=60)
    chunks = chunker.chunk(rich_doc)
    text = rich_doc.text
    assert all(len(c.text) <= 300 for c in chunks)
    assert all(nxt.start < prev.end for prev, nxt in zip(chunks, chunks[1:]))  # overlap
    assert all(c.start == 0 or text[c.start - 1].isspace() for c in chunks)  # no mid-word starts


def test_fixed_windows_cover_every_non_whitespace_character(rich_doc):
    covered = set()
    for c in FixedChunker(size=300, overlap=60).chunk(rich_doc):
        covered.update(range(c.start, c.end))
    assert all(i in covered for i, ch in enumerate(rich_doc.text) if not ch.isspace())


@pytest.mark.parametrize("kwargs", [{"size": 50}, {"size": 300, "overlap": 150}, {"size": 300, "overlap": -1}])
def test_fixed_rejects_unworkable_parameters(kwargs):
    with pytest.raises(ValueError):
        FixedChunker(**kwargs)


# ---------------------------------------------------------------- heading-aware

def test_small_sections_become_exactly_one_chunk_each(rich_doc):
    chunks = HeadingChunker().chunk(rich_doc)
    by_anchor = {c.anchor: c for c in chunks if c.anchor in ("guide", "child")}
    assert by_anchor["guide"].text.startswith("# Guide\n\nIntro paragraph one.")
    assert by_anchor["guide"].text.endswith("`inline code`.")


def test_a_section_with_only_its_heading_produces_no_chunk(rich_doc):
    assert "empty-parent" not in {c.anchor for c in HeadingChunker().chunk(rich_doc)}


def test_code_with_comments_and_blank_lines_stays_with_its_section(rich_doc):
    child = [c for c in HeadingChunker().chunk(rich_doc) if c.anchor == "child"]
    assert len(child) == 1
    assert "# this is a comment, not a heading\n\ndef f():\n    return 1\n```" in child[0].text


def test_long_sections_are_packed_by_whole_paragraphs(rich_doc):
    chunker = HeadingChunker(max_chars=1200)
    long_chunks = [c for c in chunker.chunk(rich_doc) if c.anchor == "long"]
    assert len(long_chunks) > 1
    assert all(len(c.text) <= 1200 for c in long_chunks)
    # Each piece after the first begins exactly at a paragraph boundary.
    assert all(c.text.startswith("Paragraph ") for c in long_chunks[1:])


def test_a_single_block_too_big_to_fit_falls_back_to_windows():
    code = "```python\n" + "\n".join(f"value_{i} = {i}" for i in range(400)) + "\n```"
    doc = make_doc(f"# Big {{ #big }}\n\n{code}\n", doc_id="big")
    chunks = HeadingChunker(max_chars=500, overlap=50).chunk(doc)
    assert len(chunks) > 1 and all(len(c.text) <= 500 for c in chunks)


@pytest.mark.parametrize("kwargs", [{"max_chars": 50}, {"max_chars": 400, "overlap": 200}])
def test_heading_rejects_unworkable_parameters(kwargs):
    with pytest.raises(ValueError):
        HeadingChunker(**kwargs)


def uncovered(doc, chunks) -> list[int]:
    """Non-whitespace offsets of sections with a body that no chunk contains."""
    covered = set()
    for c in chunks:
        covered.update(range(c.start, c.end))
    missing = []
    for s in doc.sections:
        if s.level and "\n" not in doc.text[s.start:s.end].strip():
            continue  # a heading with no body: deliberately no chunk
        missing += [i for i in range(s.start, s.end) if not doc.text[i].isspace() and i not in covered]
    return missing


def test_heading_chunks_lose_no_content(rich_doc):
    assert uncovered(rich_doc, HeadingChunker().chunk(rich_doc)) == []
    assert uncovered(rich_doc, HeadingChunker(max_chars=300, overlap=60).chunk(rich_doc)) == []


def test_a_heading_never_becomes_a_chunk_on_its_own():
    # Review finding: a heading too big to merge with its first block used to be emitted alone.
    doc = make_doc("## Setup { #setup }\n\n" + "lorem ipsum " * 100, doc_id="s")
    chunks = HeadingChunker().chunk(doc)
    assert all(c.text != "## Setup" for c in chunks)
    assert chunks[0].text.startswith("## Setup\n\nlorem")


def test_fixed_windows_break_at_any_whitespace_before_cutting_a_token():
    # Review finding: with no space in the window's last 20%, the cut fell mid-token
    # and the next window started there — no overlap, mid-word start.
    text = "word " * 190 + "x" * 300 + " tail" * 100
    chunks = FixedChunker().chunk(Document("d", "d", text, ()))
    assert all(c.start == 0 or text[c.start - 1].isspace() for c in chunks)
    assert all(nxt.start < prev.end for prev, nxt in zip(chunks, chunks[1:]))


# ---------------------------------------------------------------- registry

def test_get_chunker_names_the_choices_on_a_typo():
    with pytest.raises(ValueError, match="fixed"):
        get_chunker("semantic")


def test_chunker_config_records_every_parameter_for_run_manifests():
    assert chunker_config(FixedChunker()) == {"name": "fixed", "size": 1200, "overlap": 200}
    assert chunker_config(ParagraphChunker()) == {"name": "paragraph"}


# ---------------------------------------------------------------- the real corpus

@pytest.mark.corpus
@pytest.mark.parametrize("name", sorted(CHUNKERS))
def test_invariants_hold_across_the_real_corpus(real_docs, name):
    by_id = {d.doc_id: d for d in real_docs}
    chunks = chunk_corpus(real_docs, get_chunker(name))
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    for c in chunks:
        assert by_id[c.doc_id].text[c.start:c.end] == c.text
        assert c.text == c.text.strip()
    if name != "paragraph":
        assert max(len(c.text) for c in chunks) <= 1200


@pytest.mark.corpus
def test_heading_chunks_lose_no_content_on_the_real_corpus(real_docs):
    chunker = HeadingChunker()
    assert {d.doc_id: n for d in real_docs if (n := len(uncovered(d, chunker.chunk(d))))} == {}


@pytest.mark.corpus
def test_paragraph_baseline_fidelity_holds_on_the_real_corpus(real_docs):
    for doc in real_docs:
        expected = [p.strip() for p in doc.text.split("\n\n") if p.strip()]
        assert [c.text for c in ParagraphChunker().chunk(doc)] == expected, doc.doc_id
