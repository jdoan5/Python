"""A reply as the judges see it: its text, its units, and each unit's citations (PLAN.md, "By code").

The API returns an answer as text blocks, split wherever a citation starts or
ends, often mid-sentence. Joined, they are the answer's text. Units are cut
from that text by rules fixed here, not by the API:
- each fenced code block is one unit;
- everything else is cut into lines, and lines into sentences, never inside
  inline code;
- a unit with no letter or digit is dropped.
Each unit carries the citations of every reply block that overlaps it, so a
sentence the API split into three blocks is one unit with the union of their
citations. A citation that doesn't fit its documents, or whose cited text
doesn't match the blocks it names, is dropped and counted.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Mapping, Sequence

from raglab import params
from raglab.blocks import Citation, CitationError, WindowDoc, citation_intact
from raglab.markdown import fence_closes, fence_open

_ABBREVIATIONS = ("e.g", "i.e", "etc", "vs", "cf", "approx", "incl", "esp")
_BOUNDARY = re.compile(r"([.!?][)\]\"'*_]*)\s+(?=[A-Z0-9`\"'(\[*_])")
_INLINE_CODE = re.compile(r"(`+)(.+?)\1")
_LIST_NUMBER = re.compile(r"^\s*(?:[-*+]\s+)?[*_]*\d+$")  # "1." after an optional bullet or emphasis
# Paraphrased refusals to answer, for choosing which masked answers to judge (never for scoring).
_ABSTAIN_LIKE = re.compile(
    r"\b(?:does\s+not|doesn't|do\s+not|don't|cannot|can't|can\s+not)\s+(?:\w+\s+){0,3}"
    r"(?:answer|cover|contain|mention|address|include|say|describe|explain|specify|provide|discuss)\b"
    r"|\bno\s+(?:information|mention|details?|documentation)\b"
    r"|\bnot\s+(?:covered|mentioned|addressed|described|specified|included|discussed)\b", re.I)


@dataclass(frozen=True)
class Unit:
    id: str
    start: int
    end: int
    text: str
    is_code: bool
    citations: tuple[Citation, ...]


@dataclass(frozen=True)
class Answer:
    text: str
    status: str                       # ok | refused | truncated | empty
    refusal_category: str | None
    units: tuple[Unit, ...]
    citations_total: int
    citations_failed: int
    abstained: bool                   # the code flag: the prescribed sentence and nothing else
    failures: tuple[str, ...] = field(default=())

    @property
    def cited_units(self) -> int:
        return sum(1 for u in self.units if u.citations)


def _status(message: Mapping, text: str) -> tuple[str, str | None]:
    reason = message.get("stop_reason")
    if reason == "refusal":
        details = message.get("stop_details") or {}
        return "refused", details.get("category")
    if reason == "max_tokens":
        return "truncated", None
    if not text.strip():
        return "empty", None
    return "ok", None


def is_abstention(text: str) -> bool:
    """The code flag: the reply is the prescribed sentence, ignoring whitespace, case and final punctuation."""
    def norm(s: str) -> str:
        return " ".join(s.split()).strip().rstrip(".!").strip().lower()
    return bool(text.strip()) and norm(text) == norm(params.ABSTENTION_SENTENCE)


def looks_like_abstention(text: str) -> bool:
    """Tolerant of paraphrase: used only to choose which masked answers the judges see."""
    return is_abstention(text) or bool(_ABSTAIN_LIKE.search(text))


def _code_spans(text: str) -> list[tuple[int, int]]:
    """Fenced code blocks in the answer, as spans; an unclosed fence runs to the end."""
    spans, pos, start, fence = [], 0, None, None
    for line in text.split("\n"):
        end = pos + len(line)
        if fence is None:
            opened = fence_open(line)
            if opened:
                start, fence = pos, (opened[1], opened[2])
        elif fence_closes(line, *fence):
            spans.append((start, end))
            start, fence = None, None
        pos = end + 1
    if start is not None:
        spans.append((start, len(text)))
    return spans


def _sentences(text: str, start: int, end: int) -> list[tuple[int, int]]:
    """Lines, then sentences within each line, as spans of `text`. Inline code is never cut."""
    out = []
    # Inline code's contents are masked so nothing inside it is ever a boundary; its backticks are kept, so a
    # sentence that starts with inline code still starts a new unit.
    masked = _INLINE_CODE.sub(lambda m: m.group(1) + "x" * len(m.group(2)) + m.group(1), text)
    pos = start
    for line in text[start:end].split("\n"):
        l_start, l_end = pos, pos + len(line)
        cut = l_start
        for m in _BOUNDARY.finditer(masked, l_start, l_end):
            before = masked[cut:m.start()].rstrip()
            word = before.split()[-1].lower().rstrip(".").lstrip("([{\"'*_") if before.split() else ""
            if word in _ABBREVIATIONS or _LIST_NUMBER.match(masked[cut:m.start()]):
                continue  # "1." on its own, at the line's start or after the last cut, numbers what follows
            out.append((cut, m.end(1)))  # the sentence keeps its closing quotes and brackets
            cut = m.end()
        out.append((cut, l_end))
        pos = l_end + 1
    return out


def _trim(text: str, s: int, e: int) -> tuple[int, int] | None:
    while s < e and text[s].isspace():
        s += 1
    while e > s and text[e - 1].isspace():
        e -= 1
    if e <= s or not any(ch.isalnum() for ch in text[s:e]):
        return None
    return s, e


def parse_reply(message: Mapping, docs: Sequence[WindowDoc]) -> Answer:
    """The answer's text, status, units and checked citations."""
    pieces, blocks, failures = [], [], []
    total = failed = 0
    pos = 0
    for block in message.get("content") or ():
        if block.get("type") != "text":
            continue  # thinking blocks (empty text on this model by default) and anything else
        text = block.get("text") or ""
        good = []
        for raw in block.get("citations") or ():
            total += 1
            try:
                citation = Citation.from_api(raw)
                if not citation_intact(citation, docs):
                    raise CitationError("cited_text doesn't match the blocks it names")
                good.append(citation)
            except (CitationError, KeyError, TypeError, ValueError) as e:
                failed += 1
                failures.append(str(e))
        pieces.append(text)
        blocks.append((pos, pos + len(text), tuple(good)))
        pos += len(text)
    text = "".join(pieces)
    status, category = _status(message, text)

    spans: list[tuple[int, int, bool]] = []
    cursor = 0
    for c_start, c_end in _code_spans(text):
        spans += [(s, e, False) for s, e in _sentences(text, cursor, c_start)]
        spans.append((c_start, c_end, True))
        cursor = c_end
    spans += [(s, e, False) for s, e in _sentences(text, cursor, len(text))]

    units = []
    for s, e, is_code in spans:
        trimmed = _trim(text, s, e)
        if trimmed is None:
            continue
        s, e = trimmed
        seen, cites = set(), []
        for b_start, b_end, good in blocks:
            if b_start < e and b_end > s and text[max(s, b_start):min(e, b_end)].strip():
                for c in good:
                    key = (c.document_index, c.start_block, c.end_block)
                    if key not in seen:
                        seen.add(key)
                        cites.append(c)
        units.append(Unit(f"U{len(units) + 1}", s, e, text[s:e], is_code, tuple(cites)))
    return Answer(text, status, category, tuple(units), total, failed, is_abstention(text), tuple(failures))


def code_gate(answer: Answer) -> bool:
    """Every unit of at least 40 non-whitespace characters, other than a heading line, carries a citation."""
    for u in answer.units:
        if u.text.lstrip().startswith("#"):
            continue
        if len("".join(u.text.split())) >= params.CODE_GATE_MIN_CHARS and not u.citations:
            return False
    return True
