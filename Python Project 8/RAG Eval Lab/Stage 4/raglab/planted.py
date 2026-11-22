"""The judge's planted checks: near-miss swaps and true-support plants (PLAN.md, "Checks on the judges").

Built from stored answers, the pinned corpus and the section labels, without
any judge verdict. Each candidate replaces one unit's excerpts in an otherwise
unchanged support (F) request:
- **near-miss swap** (the right verdict is "not supported"): the unit's
  excerpts become one block that doesn't state the claim. The block comes from
  another part of the grade-2 section the unit cites into, or else from a
  grade-0 or grade-1 section on the target page.
  - lexical: the block shares at least one of the unit's backticked names
    but none of its numbers, quoted strings or parameter values;
  - topical: the block shares none of its backticked names.
- **cross-section plant** (the right verdict is "supported"): the unit's
  excerpts become one block from another grade-2 section of the question, the
  one sharing the most of the unit's backticked names, then of its words.
- **shape plant** (hhr only; "supported" by construction): every citation of
  two or more blocks in the answer is re-cut at the baseline's paragraph
  boundaries ("\\n\\n"), so the same characters arrive as several small
  excerpts. One unit is the gate's; the rest give a claim-level estimate.
A blind worksheet then confirms that a swap's block doesn't state the claim
and a cross-section plant's does, and records whether the answer's other
excerpts state it (the cross-unit case; `golden/prompts/v3-swap-check.md`).
Shape plants need no confirmation.

Candidates are units of an answer that cite into a grade-2 section: prose
claims (not code, not a heading line) for swaps and cross-section plants, and
for lexical swaps only those with at least one backticked name; any unit with
a multi-block citation for shape plants. They are drawn in sha256 order, at
most one per question per kind and arm.

Which candidates are judged is decided after the core wave: per kind and arm,
the first confirmed candidates whose original unit F judged supported with a
quote that holds (`gate`).
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from raglab.blocks import WindowDoc, cited_blocks
from raglab.chunking import HeadingChunker
from raglab.corpus import Document
from raglab.judge import is_heading
from raglab.sections import SectionIndex, SectionRef
from raglab.units import Answer, Unit

SWAPS = ("swap-lexical", "swap-topical")
PLANTS = ("plant-cross", "plant-shape")
KINDS = SWAPS + PLANTS

_NAME = re.compile(r"`([^`\n]+)`")
_NUMBER = re.compile(r"(?<![\w.])\d+(?:\.\d+)*(?![\w.])")
_DOUBLE_QUOTED = re.compile(r"\"([^\"\n]{1,80})\"")
_SINGLE_QUOTED = re.compile(r"(?<!\w)'([^'\n]{1,80})'(?!\w)")
_PARAMETER = re.compile(r"\b[A-Za-z_]\w*\s*=\s*([^\s,)`]+)")
_WORD = re.compile(r"[a-z][a-z_]{3,}")
_STOP = frozenset("that this with from your have will when what which them they their there then than into also "
                  "only some such each other more most must should would could does used uses using".split())
_SPLITTER = HeadingChunker()


@dataclass(frozen=True)
class Candidate:
    kind: str                 # one of KINDS
    arm: str                  # the answer's condition, e.g. hhr-4.8k-med
    question: str
    answered: str             # the answer's custom_id
    unit: str                 # the unit whose excerpts are replaced, e.g. U3
    claim: str                # that unit's text, for the worksheet
    texts: tuple[str, ...]    # what the judge is shown in place of the unit's excerpts
    spans: tuple[tuple[str, int, int], ...]  # where each text lies in the corpus: (doc_id, start, end)
    section: str | None       # the section a swap's or cross plant's block comes from
    order: str                # the sha256 that orders the draw
    others: tuple[tuple[str, tuple[str, ...]], ...] = ()  # a shape plant's other re-cut units: (unit, texts)
    context: tuple[str, ...] = ()  # the answer's other units' excerpts, for the worksheet's cross-unit question

    @property
    def name(self) -> str:
        """The planted item's name inside its judge request's condition."""
        return f"{self.kind}:{self.unit}"

    @property
    def needs_confirmation(self) -> bool:
        return self.kind != "plant-shape"

    def to_json(self) -> dict:
        return {"kind": self.kind, "arm": self.arm, "question": self.question, "answered": self.answered,
                "unit": self.unit, "spans": [list(s) for s in self.spans], "section": self.section,
                "order": self.order, "name": self.name, "recut_units": [u for u, _ in self.others]}


def draw_order(kind: str, answered: str, unit: str) -> str:
    return hashlib.sha256(f"{kind}|{answered}|{unit}".encode("utf-8")).hexdigest()


def names(text: str) -> list[str]:
    """The unit's backticked names, in order, without repeats."""
    return list(dict.fromkeys(n.strip() for n in _NAME.findall(text) if n.strip()))


def literals(text: str) -> list[str]:
    """Its numbers, quoted strings and parameter values: what a lexical near-miss must not contain."""
    found = _NUMBER.findall(text) + _DOUBLE_QUOTED.findall(text) + _SINGLE_QUOTED.findall(text)
    found += [v.strip("\"'") for v in _PARAMETER.findall(text)]
    return list(dict.fromkeys(x for x in found if x.strip()))


def lexical_near_miss(block: str, unit_names: Sequence[str], unit_literals: Sequence[str]) -> bool:
    return any(n in block for n in unit_names) and not any(_contains_literal(block, x) for x in unit_literals)


def topical_near_miss(block: str, unit_names: Sequence[str]) -> bool:
    return not any(n in block for n in unit_names)


def _contains_literal(block: str, literal: str) -> bool:
    if literal.replace(".", "").isdigit():  # a number counts only as a whole token: 4 isn't in 404
        return re.search(rf"(?<![\w.]){re.escape(literal)}(?![\w.])", block) is not None
    return literal in block


def eligible(kind: str, unit: Unit) -> bool:
    """Which units a kind is drawn from: cited prose claims for swaps and cross plants (lexical swaps also need
    a backticked name to share); any cited unit with a multi-block citation for shape plants."""
    if not unit.citations or is_heading(unit.text):
        return False
    if kind == "plant-shape":
        return any(c.end_block - c.start_block >= 2 for c in unit.citations)
    if unit.is_code:
        return False
    return bool(names(unit.text)) if kind == "swap-lexical" else True


def words(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOP}


def cited_spans(unit: Unit, docs: Sequence[WindowDoc]) -> list[tuple[str, int, int]]:
    return [(b.doc_id, b.start, b.end) for c in unit.citations for b in cited_blocks(c, docs)]


def cited_sections(unit: Unit, docs: Sequence[WindowDoc]) -> list[str]:
    seen = []
    for c in unit.citations:
        for b in cited_blocks(c, docs):
            if b.section and b.section not in seen:
                seen.append(b.section)
    return seen


def section_blocks(doc: Document, ref: SectionRef) -> list[tuple[int, int, str]]:
    """A section body's blocks, cut the way windows are (paragraphs and whole code fences)."""
    out = []
    for start, end in _SPLITTER._blocks(doc.text, ref.start, ref.end):
        text = doc.text[start:end]
        if text.strip():
            out.append((start, end, text))
    return out


def _overlaps(spans: Iterable[tuple[str, int, int]], doc_id: str, start: int, end: int) -> bool:
    return any(d == doc_id and s < end and start < e for d, s, e in spans)


def _by_hash(seed: str, blocks: Sequence[tuple[str, int, int, str]]) -> list[tuple[str, int, int, str]]:
    return sorted(blocks, key=lambda b: hashlib.sha256(f"{seed}|{b[0]}|{b[1]}".encode("utf-8")).hexdigest())


def swap_block(kind: str, unit: Unit, docs: Sequence[WindowDoc], grades: Mapping[str, int], target: str,
               corpus: Mapping[str, Document], index: SectionIndex, seed: str) -> tuple[str, int, int, str] | None:
    """The block a near-miss swap shows: first from the rest of the grade-2 section the unit cites, else from a
    grade-0 or grade-1 section on the target page; within each, the first qualifying block in sha256 order."""
    unit_names, unit_literals = names(unit.text), literals(unit.text)
    cited = cited_spans(unit, docs)
    ok = (lambda t: lexical_near_miss(t, unit_names, unit_literals)) if kind == "swap-lexical" else \
        (lambda t: topical_near_miss(t, unit_names))
    own = [s for s in cited_sections(unit, docs) if grades.get(s) == 2]
    page = target.split("#", 1)[0]
    elsewhere = sorted(loc for loc, g in grades.items() if g in (0, 1) and loc.split("#", 1)[0] == page)
    for pool in (own, elsewhere):
        blocks = []
        for loc in pool:
            ref = index.get(loc)
            if ref is None:
                continue
            for start, end, text in section_blocks(corpus[ref.doc_id], ref):
                if not _overlaps(cited, ref.doc_id, start, end) and ok(text):
                    blocks.append((ref.doc_id, start, end, loc))
        if blocks:
            return _by_hash(seed, blocks)[0]
    return None


def cross_plant_block(unit: Unit, docs: Sequence[WindowDoc], grades: Mapping[str, int],
                      corpus: Mapping[str, Document], index: SectionIndex,
                      seed: str) -> tuple[str, int, int, str] | None:
    """A block from another grade-2 section of the question: the one sharing the most of the unit's names."""
    unit_names, unit_words = names(unit.text), words(unit.text)
    own = set(cited_sections(unit, docs))
    best, best_score = [], (0, 0)
    for loc in sorted(l for l, g in grades.items() if g == 2 and l not in own):
        ref = index.get(loc)
        if ref is None:
            continue
        for start, end, text in section_blocks(corpus[ref.doc_id], ref):
            score = (sum(n in text for n in unit_names), len(unit_words & words(text)))
            if score > best_score:
                best, best_score = [(ref.doc_id, start, end, loc)], score
            elif score == best_score and score > (0, 0):
                best.append((ref.doc_id, start, end, loc))
    return _by_hash(seed, best)[0] if best else None


def baseline_pieces(text: str, start: int) -> list[tuple[int, int]]:
    """Spans of `text` (which starts at corpus offset `start`) cut where the baseline cuts: at blank lines."""
    out, pos = [], 0
    for sep in re.finditer(r"\n\n", text):
        piece = text[pos:sep.start()]
        if piece.strip():
            lead = len(piece) - len(piece.lstrip())
            out.append((start + pos + lead, start + pos + len(piece.rstrip())))
        pos = sep.end()
    piece = text[pos:]
    if piece.strip():
        lead = len(piece) - len(piece.lstrip())
        out.append((start + pos + lead, start + pos + len(piece.rstrip())))
    return out


def shape_texts(unit: Unit, docs: Sequence[WindowDoc], corpus: Mapping[str, Document]) -> list[tuple[str, int, int]]:
    """The unit's excerpts with every citation of two or more blocks re-cut as the baseline's paragraph chunks;
    a one-block citation keeps its excerpt as it was. Empty if no citation spans two or more blocks."""
    if not any(c.end_block - c.start_block >= 2 for c in unit.citations):
        return []
    out = []
    for c in unit.citations:
        blocks = cited_blocks(c, docs)
        if len(blocks) < 2:
            out += [(b.doc_id, b.start, b.end) for b in blocks]
            continue
        for b in blocks:
            text = corpus[b.doc_id].text[b.start:b.end]
            out += [(b.doc_id, s, e) for s, e in baseline_pieces(text, b.start)]
    return out


def candidates_for(kind: str, arm: str, question: str, answered: str, answer: Answer, docs: Sequence[WindowDoc],
                   grades: Mapping[str, int], target: str | None, corpus: Mapping[str, Document],
                   index: SectionIndex) -> list[Candidate]:
    """Every candidate of one kind in one answer, in draw order (the draw then keeps one per question)."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    if answer.status != "ok" or answer.abstained or target is None:
        return []
    from raglab.judge import excerpts

    shown = excerpts(answer, docs)
    out = []
    for unit in answer.units:
        if not eligible(kind, unit) or not any(grades.get(s) == 2 for s in cited_sections(unit, docs)):
            continue
        order = draw_order(kind, answered, unit.id)
        context = tuple(dict.fromkeys(t for k, t in shown.items() if not k.startswith(unit.id + ".")))
        if kind in SWAPS:
            found = swap_block(kind, unit, docs, grades, target, corpus, index, order)
            picked = [found] if found else []
        elif kind == "plant-cross":
            found = cross_plant_block(unit, docs, grades, corpus, index, order)
            picked = [found] if found else []
        else:
            picked = [(d, s, e, None) for d, s, e in shape_texts(unit, docs, corpus)]
        if not picked:
            continue
        texts = tuple(corpus[d].text[s:e] for d, s, e, _ in picked)
        others = ()
        if kind == "plant-shape":  # answer-level: every other multi-block citation is re-cut the same way
            others = tuple((u.id, tuple(corpus[d].text[s:e] for d, s, e in shape_texts(u, docs, corpus)))
                           for u in answer.units if u.id != unit.id and shape_texts(u, docs, corpus))
        out.append(Candidate(kind, arm, question, answered, unit.id, unit.text, texts,
                             tuple((d, s, e) for d, s, e, _ in picked), picked[0][3], order, others, context))
    return sorted(out, key=lambda c: c.order)


def draw(candidates: Iterable[Candidate], per_kind: int) -> list[Candidate]:
    """At most one per question, per kind and arm, in sha256 order, up to `per_kind` of each (oversampled)."""
    out, taken = [], {}
    for c in sorted(candidates, key=lambda c: (c.kind, c.arm, c.order)):
        key = (c.kind, c.arm)
        used = taken.setdefault(key, set())
        if c.question in used or len(used) >= per_kind:
            continue
        used.add(c.question)
        out.append(c)
    return out


def planted_excerpts(c: Candidate) -> dict[str, list[str]]:
    """What `support_body` shows in place of the units' own excerpts: the gate's unit, and for a shape plant
    every other re-cut unit of the answer."""
    return {c.unit: list(c.texts), **{u: list(t) for u, t in c.others}}


def gate(candidates: Sequence[Candidate], confirmed: Mapping[str, bool], originally_supported: Mapping[tuple[str, str], bool],
         needed: Mapping[tuple[str, str], int]) -> list[Candidate]:
    """The candidates judged: per (kind, arm), the first in draw order that the worksheet confirmed (shape plants
    need no confirmation) and whose original unit F judged supported with a quote that holds."""
    out, counts = [], {}
    for c in sorted(candidates, key=lambda c: (c.kind, c.arm, c.order)):
        key = (c.kind, c.arm)
        if counts.get(key, 0) >= needed.get(key, 0):
            continue
        if c.needs_confirmation and not confirmed.get(f"{c.answered}:{c.name}", False):
            continue
        if not originally_supported.get((c.answered, c.unit), False):
            continue
        counts[key] = counts.get(key, 0) + 1
        out.append(c)
    return out
