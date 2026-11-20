"""Three ways to cut documents into retrievable chunks — the first variable Stage 2 measures.

- paragraph: RAG Mini Chat's strategy, reproduced exactly (split on blank lines,
  no size limit) so it can serve as the control group.
- fixed:     a sliding character window with overlap — the common default.
- heading:   one chunk per section, packing whole paragraphs and keeping code
  blocks intact when a section is too long.

Every chunk records the exact span it came from: `doc.text[start:end] == chunk.text`.
That invariant is what lets Stage 2 score any strategy against the same
section-level gold labels.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import ClassVar, Protocol

from raglab.corpus import Document
from raglab.markdown import fence_closes, fence_open

DEFAULT_SIZE = 1200   # chars (~300 tokens); shared by fixed and heading so the comparison is fair
DEFAULT_OVERLAP = 200


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    strategy: str
    start: int
    end: int
    text: str
    heading_path: tuple[str, ...]  # path of the section the chunk starts in
    anchor: str

    @property
    def location(self) -> str:
        """`doc_id#anchor` — the same form Stage 2's gold labels use."""
        return f"{self.doc_id}#{self.anchor}" if self.anchor else self.doc_id

    def index_text(self, with_context: bool = True) -> str:
        """What a retriever indexes: optionally prefixed with the heading breadcrumb.

        The prefix is an independent switch rather than part of a strategy, so
        Stage 2 can measure its effect separately from where chunks are cut.
        """
        if with_context and self.heading_path:
            return " > ".join(self.heading_path) + "\n" + self.text
        return self.text


class Chunker(Protocol):
    name: ClassVar[str]

    def chunk(self, doc: Document) -> list[Chunk]: ...


def _trim(text: str, start: int, end: int) -> tuple[int, int] | None:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return (start, end) if end > start else None


def _make_chunks(doc: Document, strategy: str, spans: list[tuple[int, int]]) -> list[Chunk]:
    chunks = []
    for n, (start, end) in enumerate(spans):
        section = doc.section_at(start)
        chunks.append(
            Chunk(
                chunk_id=f"{doc.doc_id}::{strategy}-{n:03d}",
                doc_id=doc.doc_id,
                strategy=strategy,
                start=start,
                end=end,
                text=doc.text[start:end],
                heading_path=section.heading_path,
                anchor=section.anchor,
            )
        )
    return chunks


def _break_point(text: str, pos: int, cut: int, floor: int) -> int:
    for lo in (floor, pos + 1):
        for sep in ("\n", " "):
            at = text.rfind(sep, lo, cut)
            if at > pos:
                return at
    return cut


def _window_spans(text: str, start: int, end: int, size: int, overlap: int) -> list[tuple[int, int]]:
    """Fixed windows over text[start:end], ending on whitespace where possible.

    A cut prefers the last newline, then the last space, within the final 20%
    of the window, then anywhere in the window; only a window with no
    whitespace at all (a single giant token) is cut mid-token. Each next
    window begins `overlap` chars back, moved forward to a word start.
    """
    spans: list[tuple[int, int]] = []
    pos = start
    while pos < end:
        while pos < end and text[pos].isspace():
            pos += 1
        if pos >= end:
            break
        cut = min(pos + size, end)
        if cut < end:
            cut = _break_point(text, pos, cut, floor=pos + int(size * 0.8))
        span = _trim(text, pos, cut)
        if span:
            spans.append(span)
        if cut >= end:
            break
        nxt = max(cut - overlap, pos + 1)
        while nxt < cut and not text[nxt - 1].isspace():
            nxt += 1
        pos = nxt
    return spans


@dataclass(frozen=True)
class ParagraphChunker:
    """RAG Mini Chat's baseline: `text.split("\\n\\n")`, stripped, empties dropped."""

    name: ClassVar[str] = "paragraph"

    def chunk(self, doc: Document) -> list[Chunk]:
        text, spans, pos = doc.text, [], 0
        for sep in re.finditer(r"\n\n", text):
            span = _trim(text, pos, sep.start())
            if span:
                spans.append(span)
            pos = sep.end()
        span = _trim(text, pos, len(text))
        if span:
            spans.append(span)
        return _make_chunks(doc, self.name, spans)


@dataclass(frozen=True)
class FixedChunker:
    size: int = DEFAULT_SIZE
    overlap: int = DEFAULT_OVERLAP
    name: ClassVar[str] = "fixed"

    def __post_init__(self) -> None:
        if self.size < 100 or not 0 <= self.overlap < self.size // 2:
            raise ValueError("need size >= 100 and 0 <= overlap < size / 2")

    def chunk(self, doc: Document) -> list[Chunk]:
        spans = _window_spans(doc.text, 0, len(doc.text), self.size, self.overlap)
        return _make_chunks(doc, self.name, spans)


@dataclass(frozen=True)
class HeadingChunker:
    """One chunk per section; long sections are packed paragraph by paragraph.

    Code fences are treated as a single block even when they contain blank
    lines, so a function is never split from its own body unless the block
    alone exceeds `max_chars` (then it falls back to windows). Sections with
    no content beyond their heading line produce no chunk — their title still
    reaches the index through the children's breadcrumb.
    """

    max_chars: int = DEFAULT_SIZE
    overlap: int = DEFAULT_OVERLAP
    name: ClassVar[str] = "heading"

    def __post_init__(self) -> None:
        if self.max_chars < 100 or not 0 <= self.overlap < self.max_chars // 2:
            raise ValueError("need max_chars >= 100 and 0 <= overlap < max_chars / 2")

    def chunk(self, doc: Document) -> list[Chunk]:
        spans: list[tuple[int, int]] = []
        for section in doc.sections:
            span = _trim(doc.text, section.start, section.end)
            if span is None or not self._has_body(doc.text, *span, section.level):
                continue
            if span[1] - span[0] <= self.max_chars:
                spans.append(span)
            else:
                spans.extend(self._pack(doc.text, *span))
        return _make_chunks(doc, self.name, spans)

    @staticmethod
    def _has_body(text: str, start: int, end: int, level: int) -> bool:
        if level == 0:  # text before the first heading has no heading line
            return True
        newline = text.find("\n", start, end)
        return newline != -1 and bool(text[newline:end].strip())

    def _blocks(self, text: str, start: int, end: int) -> list[tuple[int, int]]:
        """Paragraphs and whole code fences, as spans."""
        blocks: list[tuple[int, int]] = []
        block_start: int | None = None
        fence: tuple[str, int] | None = None  # (char, length) of the open fence
        pos = start
        for line in text[start:end].split("\n"):
            line_end = pos + len(line)
            opened = fence_open(line) if fence is None else None
            if opened:
                if block_start is not None:  # a fence ends the paragraph before it
                    blocks.append((block_start, pos))
                block_start, fence = pos, (opened[1], opened[2])
            elif fence is not None:
                # Same rule as the cleaner: a "```python" line inside the block is content.
                if fence_closes(line, *fence):
                    blocks.append((block_start, line_end))
                    block_start, fence = None, None
            elif not line.strip():
                if block_start is not None:
                    blocks.append((block_start, pos))
                    block_start = None
            elif block_start is None:
                block_start = pos
            pos = line_end + 1
        if block_start is not None:
            blocks.append((block_start, end))
        return [s for b in blocks if (s := _trim(text, *b))]

    def _pack(self, text: str, start: int, end: int) -> list[tuple[int, int]]:
        spans: list[tuple[int, int]] = []
        current: tuple[int, int] | None = None
        blocks = self._blocks(text, start, end)
        # The heading line travels with the block after it, so a section can
        # never yield a chunk that is just its heading.
        if len(blocks) >= 2 and text[blocks[0][0]] == "#" and "\n" not in text[blocks[0][0]:blocks[0][1]]:
            blocks[:2] = [(blocks[0][0], blocks[1][1])]
        for b_start, b_end in blocks:
            if b_end - b_start > self.max_chars:
                if current:
                    spans.append(current)
                    current = None
                spans.extend(_window_spans(text, b_start, b_end, self.max_chars, self.overlap))
            elif current and b_end - current[0] <= self.max_chars:
                current = (current[0], b_end)
            else:
                if current:
                    spans.append(current)
                current = (b_start, b_end)
        if current:
            spans.append(current)
        return spans


CHUNKERS: dict[str, type] = {c.name: c for c in (ParagraphChunker, FixedChunker, HeadingChunker)}


def get_chunker(name: str) -> Chunker:
    try:
        return CHUNKERS[name]()
    except KeyError:
        raise ValueError(f"unknown chunker {name!r}; choose from {sorted(CHUNKERS)}") from None


def chunker_config(chunker: Chunker) -> dict:
    """Parameters for run manifests (Stage 2): name plus every tunable field."""
    return {"name": chunker.name, **asdict(chunker)}  # type: ignore[call-overload]


def chunk_corpus(docs: list[Document], chunker: Chunker) -> list[Chunk]:
    return [c for doc in docs for c in chunker.chunk(doc)]
