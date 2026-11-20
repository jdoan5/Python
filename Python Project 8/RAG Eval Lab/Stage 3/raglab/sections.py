"""Sections as scoring units: where each section's body lies, and which sections a span retrieves.

Gold labels name sections (`doc_id#anchor`); retrievers return chunks. This
module is the bridge between the two, and it is the same for every chunking
strategy, which is what makes their scores comparable.

A section's *body* is its text after the heading line. A chunk that holds only
a heading has retrieved a title, not an answer, so it retrieves nothing.

A span *retrieves* a section when its overlap with the section's body is at
least the smallest of: MIN_OVERLAP_CHARS, the whole body, or all the body text
the span holds (headings aside). So a 60-character paragraph that sits inside
an answer counts, whether or not its heading came with it. A short section
fully inside a window counts. A window that merely clips the first line of the
next section does not.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from typing import Iterable, Iterator

from raglab.corpus import Document, Section

MIN_OVERLAP_CHARS = 100


def location_of(doc_id: str, anchor: str) -> str:
    """`doc_id#anchor` — the same form chunks report and gold labels use."""
    return f"{doc_id}#{anchor}" if anchor else doc_id


@dataclass(frozen=True)
class SectionRef:
    location: str
    doc_id: str
    heading_path: tuple[str, ...]
    start: int   # body start: after the heading line, whitespace trimmed
    end: int     # exclusive

    @property
    def length(self) -> int:
        return self.end - self.start


def body_span(text: str, section: Section) -> tuple[int, int] | None:
    """The section's text after its heading line, trimmed; None if there is none."""
    start, end = section.start, section.end
    if section.level > 0:
        newline = text.find("\n", start, end)
        if newline == -1:
            return None
        start = newline + 1
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return (start, end) if end > start else None


def overlap_retrieves(overlap: int, body_length: int, span_body_chars: int) -> bool:
    """The hit rule, on its own so tests can pin it down exactly.

    `span_body_chars` is how much section-body text the span holds in total, so
    a span made only of this section's text (plus headings) counts however short.
    """
    return overlap > 0 and overlap >= min(MIN_OVERLAP_CHARS, body_length, span_body_chars)


class SectionIndex:
    """Every section with a body, findable by location or by the spans that overlap it."""

    def __init__(self, docs: Iterable[Document]) -> None:
        self._by_location: dict[str, SectionRef] = {}
        self._by_doc: dict[str, tuple[list[int], list[SectionRef]]] = {}
        self._ink: dict[str, list[int]] = {}  # prefix counts of non-whitespace characters, per doc
        self.empty: set[str] = set()  # sections that are only a heading: nothing to retrieve
        for doc in docs:
            if doc.doc_id in self._by_doc:
                raise ValueError(f"duplicate document {doc.doc_id!r}")
            counts = [0]
            for ch in doc.text:
                counts.append(counts[-1] + (not ch.isspace()))
            self._ink[doc.doc_id] = counts
            refs: list[SectionRef] = []
            for section in doc.sections:
                location = location_of(doc.doc_id, section.anchor)
                if location in self._by_location or location in self.empty:
                    raise ValueError(f"duplicate section location {location!r}")
                span = body_span(doc.text, section)
                if span is None:
                    self.empty.add(location)
                    continue
                ref = SectionRef(location, doc.doc_id, section.heading_path, *span)
                self._by_location[location] = ref
                refs.append(ref)
            self._by_doc[doc.doc_id] = ([r.start for r in refs], refs)

    def __len__(self) -> int:
        return len(self._by_location)

    def __iter__(self) -> Iterator[SectionRef]:
        return iter(self._by_location.values())

    def __contains__(self, location: object) -> bool:
        return location in self._by_location

    def get(self, location: str) -> SectionRef | None:
        return self._by_location.get(location)

    def ink(self, doc_id: str, start: int, end: int) -> int:
        """Non-whitespace characters in `[start, end)`: text, not the gaps chunkers drop between it."""
        counts = self._ink[doc_id]
        start, end = max(start, 0), min(end, len(counts) - 1)
        return counts[end] - counts[start] if end > start else 0

    def retrieved_by(self, doc_id: str, start: int, end: int) -> list[tuple[SectionRef, int]]:
        """Sections the span `[start, end)` of `doc_id` retrieves, most-overlapped first.

        Bodies are disjoint and sorted, so only the body starting at or before
        `start` and those starting inside the span can overlap it.
        """
        starts, refs = self._by_doc.get(doc_id, ([], []))
        first = max(bisect.bisect_right(starts, start) - 1, 0)
        overlaps: list[tuple[SectionRef, int]] = []
        for ref in refs[first:]:
            if ref.start >= end:
                break
            overlap = min(end, ref.end) - max(start, ref.start)
            if overlap > 0:
                overlaps.append((ref, overlap))
        body_chars = sum(overlap for _, overlap in overlaps)
        found = [(ref, o) for ref, o in overlaps if overlap_retrieves(o, ref.length, body_chars)]
        found.sort(key=lambda pair: (-pair[1], pair[0].start))
        return found
