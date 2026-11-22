"""Retrieved chunks as citable documents, and citations back to the sections they came from.

Every arm shows the model its context the same way: one custom-content
document per retrieved chunk, in rank order, titled with the chunk's heading
path, cut into blocks that are paragraphs and whole code fences (the same cut
`HeadingChunker` packs by). The API cites whole blocks
(`content_block_location`: a document index and a block range, end
exclusive), so every citation names exact spans of the cleaned corpus, which
map to sections and their grades with no fuzzy matching. Titles are shown to
the model but can't be cited.

The API copies `cited_text` from the documents itself; `citation_intact`
checks that it matches the blocks the citation names. A mismatch is a bug in
this mapping, never a result.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping, Sequence

from raglab.chunking import HeadingChunker
from raglab.corpus import Document
from raglab.markdown import fence_open
from raglab.metrics import Span
from raglab.sections import SectionIndex

_SPACE = re.compile(r"\s+")


class CitationError(ValueError):
    """A citation that doesn't fit the documents it claims to cite: a mapping bug."""


@dataclass(frozen=True)
class Block:
    doc_id: str
    start: int
    end: int
    text: str
    section: str | None  # the section whose body holds the block; None for a heading line
    is_code: bool


@dataclass(frozen=True)
class WindowDoc:
    """One retrieved chunk, as the model sees it."""

    span: Span
    title: str
    blocks: tuple[Block, ...]

    def api_document(self) -> dict:
        return {
            "type": "document",
            "source": {"type": "content", "content": [{"type": "text", "text": b.text} for b in self.blocks]},
            "title": self.title,
            "citations": {"enabled": True},
        }


def _heading_path(doc: Document, pos: int) -> tuple[str, ...]:
    path: tuple[str, ...] = (doc.title,)
    for section in doc.sections:
        if section.start <= pos < section.end:
            path = section.heading_path
        elif section.start > pos:
            break
    return path


def _section_of(index: SectionIndex, doc_id: str, start: int, end: int) -> str | None:
    found = index.retrieved_by(doc_id, start, end)
    return found[0][0].location if found else None


_SPLITTER = HeadingChunker()


def window_documents(spans: Sequence[Span], docs: Mapping[str, Document], index: SectionIndex) -> list[WindowDoc]:
    """A packed window (Stage 3's `pack` output) as the documents a request carries."""
    out = []
    for doc_id, start, end in spans:
        doc = docs[doc_id]
        blocks = []
        for b_start, b_end in _SPLITTER._blocks(doc.text, start, end):
            text = doc.text[b_start:b_end]
            if not text.strip():
                continue
            first_line = text.split("\n", 1)[0]
            blocks.append(Block(doc_id, b_start, b_end, text, _section_of(index, doc_id, b_start, b_end),
                                fence_open(first_line) is not None))
        if not blocks:
            continue  # a chunk with no text to show (never happens for packed chunks; guarded anyway)
        out.append(WindowDoc((doc_id, start, end), " > ".join(_heading_path(doc, start)), tuple(blocks)))
    return out


# ---------------------------------------------------------------------------------------------------- citations

@dataclass(frozen=True)
class Citation:
    document_index: int
    start_block: int
    end_block: int  # exclusive
    cited_text: str

    @classmethod
    def from_api(cls, c: Mapping) -> "Citation":
        if c.get("type") != "content_block_location":
            raise CitationError(f"expected a content_block_location citation, got {c.get('type')!r}")
        return cls(int(c["document_index"]), int(c["start_block_index"]), int(c["end_block_index"]),
                   str(c.get("cited_text", "")))


def cited_blocks(c: Citation, docs: Sequence[WindowDoc]) -> tuple[Block, ...]:
    if not 0 <= c.document_index < len(docs):
        raise CitationError(f"document {c.document_index} doesn't exist ({len(docs)} documents)")
    blocks = docs[c.document_index].blocks
    if not 0 <= c.start_block < c.end_block <= len(blocks):
        raise CitationError(f"blocks [{c.start_block}, {c.end_block}) don't fit document {c.document_index}, "
                            f"which has {len(blocks)}")
    return blocks[c.start_block:c.end_block]


def squash(text: str) -> str:
    """Text with all whitespace removed: the form cited text is compared in (block joins vary)."""
    return _SPACE.sub("", text)


def citation_intact(c: Citation, docs: Sequence[WindowDoc]) -> bool:
    return squash(c.cited_text) == squash("".join(b.text for b in cited_blocks(c, docs)))
