from __future__ import annotations

import pytest

from conftest import make_doc
from raglab.chunking import chunk_corpus, get_chunker
from raglab.sections import MIN_OVERLAP_CHARS, SectionIndex, body_span, overlap_retrieves

PAGE = """\
# Guide { #guide }

Intro paragraph about the guide.

## Empty { #empty }

## Alpha { #alpha }

Alpha body line one.

Alpha body line two.

## Beta { #beta }

Beta body.
"""


@pytest.fixture
def doc():
    return make_doc(PAGE, "guide")


@pytest.fixture
def index(doc):
    return SectionIndex([doc])


def text_of(doc, ref):
    return doc.text[ref.start:ref.end]


def test_body_excludes_the_heading_line_and_surrounding_whitespace(doc, index):
    alpha = index.get("guide#alpha")
    assert text_of(doc, alpha) == "Alpha body line one.\n\nAlpha body line two."
    assert not text_of(doc, alpha).startswith("#")


def test_a_heading_with_nothing_under_it_has_no_body(doc, index):
    assert "guide#empty" not in index
    assert "guide#empty" in index.empty
    empty = next(s for s in doc.sections if s.anchor == "empty")
    assert body_span(doc.text, empty) is None


def test_text_before_the_first_heading_is_all_body():
    doc = make_doc("Preamble text.\n\n# Title { #t }\n\nBody.\n", "p")
    index = SectionIndex([doc])
    assert doc.text[index.get("p").start:index.get("p").end] == "Preamble text."


@pytest.mark.parametrize(
    ("overlap", "body", "span_body", "expected"),
    [
        (MIN_OVERLAP_CHARS, 500, 1200, True),       # enough of a long section
        (MIN_OVERLAP_CHARS - 1, 500, 1200, False),  # a window clipping a long section's edge
        (60, 60, 1200, True),                       # all of a short section
        (60, 500, 60, True),                        # a short chunk made only of this section's text
        (59, 500, 60, False),                       # ...but not one with text from another section too
        (0, 500, 1200, False),
        (0, 0, 0, False),
    ],
)
def test_hit_rule(overlap, body, span_body, expected):
    assert overlap_retrieves(overlap, body, span_body) is expected


def test_a_heading_plus_one_short_line_counts_the_same_as_the_line_alone(doc, index):
    # Same text, same verdict, whether or not the chunker kept the heading attached.
    line = doc.text.index("Beta body.")
    heading = doc.text.index("## Beta")
    assert [r.location for r, _ in index.retrieved_by("guide", line, line + len("Beta body."))] == ["guide#beta"]
    assert [r.location for r, _ in index.retrieved_by("guide", heading, line + len("Beta body."))] == ["guide#beta"]


def test_a_lone_heading_retrieves_nothing(doc, index):
    start = doc.text.index("## Alpha")
    end = doc.text.index("\n", start)
    assert index.retrieved_by("guide", start, end) == []


def test_a_span_across_two_sections_retrieves_both_most_overlapped_first(doc, index):
    start = doc.text.index("Alpha body line one")
    found = [(ref.location, overlap) for ref, overlap in index.retrieved_by("guide", start, len(doc.text))]
    assert found == [("guide#alpha", index.get("guide#alpha").length), ("guide#beta", len("Beta body."))]


def test_a_fragment_of_one_section_alongside_another_does_not_retrieve_it(doc, index):
    # 20 of Alpha's 42 body chars, plus Beta's whole body: Beta yes, Alpha no.
    start = doc.text.index("Alpha body line two")
    assert [r.location for r, _ in index.retrieved_by("guide", start, len(doc.text))] == ["guide#beta"]


def test_unknown_documents_and_spans_before_any_body_retrieve_nothing(index):
    assert index.retrieved_by("nope", 0, 50) == []
    assert index.retrieved_by("guide", 0, 3) == []  # inside the "# Guide" heading line


def test_duplicate_locations_are_rejected():
    doc = make_doc(PAGE, "guide")
    with pytest.raises(ValueError, match="duplicate document"):
        SectionIndex([doc, doc])


@pytest.mark.corpus
def test_every_real_chunk_retrieves_the_section_it_lies_in(real_docs):
    # Paragraph and heading chunks never cross a section boundary, so each one
    # that holds any body text must retrieve exactly the section it starts in.
    index = SectionIndex(real_docs)
    by_doc = {d.doc_id: d for d in real_docs}
    for name in ("paragraph", "heading"):
        for chunk in chunk_corpus(real_docs, get_chunker(name)):
            found = index.retrieved_by(chunk.doc_id, chunk.start, chunk.end)
            body = index.get(chunk.location)
            if body is None or chunk.end <= body.start:  # a lone heading, or a heading-only section
                assert found == [], chunk.chunk_id
            else:
                assert [ref.location for ref, _ in found] == [chunk.location], chunk.chunk_id
    assert len(index) == sum(1 for d in by_doc.values() for _ in d.sections) - len(index.empty)


def test_sections_are_ordered_by_overlap_not_by_position():
    # Same span shape twice; only which section holds more of it changes.
    short_beta = SectionIndex([make_doc(PAGE, "guide")])
    long_beta = SectionIndex([make_doc(PAGE.replace("Beta body.", "Beta body. " * 20), "guide")])
    for idx, expected in ((short_beta, ["guide#alpha", "guide#beta"]), (long_beta, ["guide#beta", "guide#alpha"])):
        a, b = idx.get("guide#alpha"), idx.get("guide#beta")
        assert [r.location for r, _ in idx.retrieved_by("guide", a.start, b.end)] == expected
