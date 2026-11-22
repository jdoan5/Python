"""The two judges: support (F) from a unit's own excerpts, correctness (C) against the key and reference.

PLAN.md, "The support judge" and "The correctness judge". Both are
claude-opus-5-5 with structured outputs. Neither sees the window, the arm or
any grade. F never sees the reference or the key, so it can't recognize a gold
passage by comparing it with one. C never sees what the answer cited.
Everything is rendered as plain text: API version 2026-10-21 refuses user text
blocks that carry citations.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from raglab import params
from raglab.blocks import Citation, WindowDoc, cited_blocks
from raglab.units import Answer

STAGE = Path(__file__).resolve().parents[1]
PROMPTS = STAGE / "prompts"
SUPPORT_LEVELS = ("supported", "partial", "unsupported", "contradicted", "uncited", "n/a")
_EXCERPT_ID = re.compile(r"^(U\d+)\.E(\d+)$")


JUDGES = ("F", "C")


def judge_condition(judge: str, answered: str, planted: str | None = None) -> str:
    """A judge request's condition: which judge, the answer it judges (that answer's custom_id), and, for a
    planted item, which plant. Two runs that give byte-identical answers are still two judge requests."""
    if judge not in JUDGES:
        raise ValueError(f"judge must be one of {JUDGES}")
    if not answered:
        raise ValueError("a judge request names the answer it judges")
    return f"{judge}:{answered}" + (f":{planted}" if planted else "")


def is_heading(text: str) -> bool:
    """A Markdown heading line in an answer: framing, never a claim (PLAN.md, "By code")."""
    first = text.lstrip()
    return first.startswith("#") and "\n" not in first.strip()


def quote_holds(quote: str, text: str) -> bool:
    """Stage 3's rule (analysis/v2_labels.py): a quote counts only if it is in the text, whitespace aside."""
    squash = lambda s: " ".join(s.split())  # noqa: E731
    return bool(quote.strip()) and squash(quote) in squash(text)


def _load(name: str) -> str:
    return (PROMPTS / name).read_text(encoding="utf-8")


def _schema(name: str) -> dict:
    return json.loads(_load(name))


# ---------------------------------------------------------------------------------------------------- F


Planted = Mapping[str, Sequence[str]]  # unit id -> the excerpt texts shown in place of that unit's own


def excerpts(answer: Answer, docs: Sequence[WindowDoc], planted: Planted | None = None) -> dict[str, str]:
    """Each unit's excerpts by id (U3.E1, ...): the text of the blocks each citation names.

    A planted item (a near-miss swap or a true-support plant, PLAN.md "Checks on
    the judges") shows other texts in place of one unit's excerpts; every other
    unit is shown exactly as in its original request.
    """
    planted = planted or {}
    unknown = set(planted) - {u.id for u in answer.units}
    if unknown:
        raise ValueError(f"planted excerpts for units the answer doesn't have: {sorted(unknown)}")
    out = {}
    for unit in answer.units:
        if unit.id in planted:
            if not unit.citations or not planted[unit.id]:
                raise ValueError(f"{unit.id}: only a cited unit's excerpts can be replaced, and by at least one text")
            texts = list(planted[unit.id])
        else:
            texts = ["\n\n".join(b.text for b in cited_blocks(c, docs)) for c in unit.citations]
        for n, text in enumerate(texts, start=1):
            out[f"{unit.id}.E{n}"] = text
    return out


def render_support(question: str, answer: Answer, docs: Sequence[WindowDoc], planted: Planted | None = None) -> str:
    ex = excerpts(answer, docs, planted)
    lines = [f"<question>\n{question}\n</question>", f"<answer>\n{answer.text}\n</answer>", "<units>"]
    for unit in answer.units:
        lines.append(f"{unit.id}: {unit.text}")
        ids = [k for k in ex if k.startswith(unit.id + ".")]
        if not ids:
            lines.append("  (no citations)")
        for k in ids:
            lines.append(f"  {k}: " + ex[k].replace("\n", "\n    "))
    lines.append("</units>")
    return "\n".join(lines)


def judge_body(rubric: str, schema: dict, user_text: str, max_tokens: int, effort: str) -> dict:
    return {
        "model": params.JUDGE,
        "max_tokens": int(max_tokens),
        "system": rubric,
        "messages": [{"role": "user", "content": [{"type": "text", "text": user_text}]}],
        "output_config": {"effort": effort, "format": {"type": "json_schema", "schema": schema}},
    }


def support_body(question: str, answer: Answer, docs: Sequence[WindowDoc], max_tokens: int,
                 effort: str = "medium", planted: Planted | None = None) -> dict:
    return judge_body(_load("judge-F-v1.md"), _schema("judge-F-v1.schema.json"),
                      render_support(question, answer, docs, planted), max_tokens, effort)


@dataclass(frozen=True)
class UnitVerdict:
    id: str
    claim: bool
    support: str         # one of SUPPORT_LEVELS, after code's corrections
    quotes_held: int
    quotes_failed: int
    reason: str

    @property
    def supported(self) -> bool:
        """Supported, with at least one quote that holds."""
        return self.claim and self.support == "supported" and self.quotes_held > 0


@dataclass(frozen=True)
class SupportVerdict:
    units: tuple[UnitVerdict, ...]
    warnings: tuple[str, ...]

    @property
    def claims(self) -> list[UnitVerdict]:
        return [u for u in self.units if u.claim]

    @property
    def all_supported(self) -> bool:
        return bool(self.claims) and all(u.supported for u in self.claims)

    @property
    def support_rate(self) -> float | None:
        return sum(u.supported for u in self.claims) / len(self.claims) if self.claims else None

    @property
    def any_unbacked(self) -> bool:
        """S3's penalty: a claim that is unsupported, uncited or contradicted."""
        return any(u.support in ("unsupported", "uncited", "contradicted") for u in self.claims)


class InvalidVerdict(ValueError):
    """Output the plan re-runs once: missing, duplicate or unknown ids, bad enums, misattributed or missing quotes."""


def read_output(message: Mapping) -> dict:
    text = "".join(b.get("text") or "" for b in message.get("content") or () if b.get("type") == "text")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        raise InvalidVerdict("the output isn't JSON") from None


def check_support(output: Mapping, answer: Answer, docs: Sequence[WindowDoc],
                  planted: Planted | None = None) -> SupportVerdict:
    """Validate F's output against the units it was shown, then score it (quotes must hold)."""
    ex = excerpts(answer, docs, planted)
    expected = [u.id for u in answer.units]
    got = [u.get("id") for u in output.get("units", [])]
    if sorted(got) != sorted(expected) or len(set(got)) != len(got):
        raise InvalidVerdict(f"unit ids {got} don't match {expected}")
    by_id = {u.id: u for u in answer.units}
    verdicts, warnings = [], []
    for raw in output["units"]:
        uid, claim, support = raw["id"], bool(raw["claim"]), raw["support"]
        if support not in SUPPORT_LEVELS:
            raise InvalidVerdict(f"{uid}: unknown support level {support!r}")
        unit = by_id[uid]
        if claim and is_heading(unit.text):
            warnings.append(f"{uid}: a heading line marked a claim; headings are framing")
            claim = False
        if not claim:
            support = "n/a"
        elif not unit.citations:
            if support != "uncited":
                warnings.append(f"{uid}: marked {support} but has no citations; counted as uncited")
            support = "uncited"
        elif support in ("uncited", "n/a"):
            raise InvalidVerdict(f"{uid}: a cited claim marked {support}")
        held = failed = 0
        for q in raw.get("quotes", []):
            m = _EXCERPT_ID.match(q.get("excerpt", ""))
            if not m or m.group(1) != uid or q["excerpt"] not in ex:
                raise InvalidVerdict(f"{uid}: a quote names excerpt {q.get('excerpt')!r}, not one of its own")
            if len(q.get("quote", "")) > params.QUOTE_MAX_CHARS:
                warnings.append(f"{uid}: a quote of {len(q['quote'])} characters")
            if quote_holds(q.get("quote", ""), ex[q["excerpt"]]):
                held += 1
            else:
                failed += 1
        if claim and support in ("supported", "partial") and held + failed == 0:
            raise InvalidVerdict(f"{uid}: a {support} claim with no quote")
        if len(raw.get("reason", "").split()) > params.REASON_MAX_WORDS:
            warnings.append(f"{uid}: a reason of {len(raw['reason'].split())} words")
        verdicts.append(UnitVerdict(uid, claim, support, held, failed, raw.get("reason", "")))
    order = {uid: n for n, uid in enumerate(expected)}
    return SupportVerdict(tuple(sorted(verdicts, key=lambda v: order[v.id])), tuple(warnings))


# ---------------------------------------------------------------------------------------------------- C


@dataclass(frozen=True)
class Nugget:
    id: str
    text: str
    essential: bool
    alternatives: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReferenceSection:
    location: str
    heading: str
    text: str


def render_correctness(question: str, key: Sequence[Nugget], reference: Sequence[ReferenceSection],
                       answer_text: str) -> str:
    lines = [f"<question>\n{question}\n</question>", "<key>"]
    for n in key:
        alt = f" Alternatives: {' | '.join(n.alternatives)}" if n.alternatives else ""
        lines.append(f"{n.id} ({'essential' if n.essential else 'supporting'}): {n.text}{alt}")
    lines += ["</key>", "<reference>"]
    for i, s in enumerate(reference, start=1):
        lines.append(f"R{i}: {s.heading}\n{s.text}")
    lines += ["</reference>", f"<answer>\n{answer_text}\n</answer>"]
    return "\n".join(lines)


def correctness_body(question: str, key: Sequence[Nugget], reference: Sequence[ReferenceSection], answer_text: str,
                     max_tokens: int, effort: str = "medium") -> dict:
    return judge_body(_load("judge-C-v1.md"), _schema("judge-C-v1.schema.json"),
                      render_correctness(question, key, reference, answer_text), max_tokens, effort)


@dataclass(frozen=True)
class CorrectnessVerdict:
    conveyed: dict[str, str]
    essential: tuple[str, ...]
    contradiction: bool          # only if its reference quote holds
    contradiction_claimed: bool  # as the judge said, before the quote check
    abstention: str
    resolves: str
    warnings: tuple[str, ...]

    @property
    def correct(self) -> bool:
        return all(self.conveyed[n] == "yes" for n in self.essential) and not self.contradiction


def check_correctness(output: Mapping, key: Sequence[Nugget], reference: Sequence[ReferenceSection]) -> CorrectnessVerdict:
    expected = [n.id for n in key]
    got = [n.get("id") for n in output.get("nuggets", [])]
    if sorted(got) != sorted(expected) or len(set(got)) != len(got):
        raise InvalidVerdict(f"nugget ids {got} don't match {expected}")
    conveyed = {}
    for raw in output["nuggets"]:
        if raw["conveyed"] not in ("yes", "partly", "no"):
            raise InvalidVerdict(f"{raw['id']}: unknown value {raw['conveyed']!r}")
        conveyed[raw["id"]] = raw["conveyed"]
    if output.get("abstention") not in ("none", "partial", "full"):
        raise InvalidVerdict(f"unknown abstention {output.get('abstention')!r}")
    if output.get("resolves") not in ("yes", "partly", "no"):
        raise InvalidVerdict(f"unknown resolves {output.get('resolves')!r}")
    c = output.get("contradiction") or {}
    claimed = bool(c.get("present"))
    holds = claimed and quote_holds(c.get("reference_quote", ""), "\n\n".join(s.text for s in reference))
    warnings = []
    if claimed and not holds:
        warnings.append("a contradiction whose reference quote doesn't hold: not counted")
    return CorrectnessVerdict(conveyed, tuple(n.id for n in key if n.essential), holds, claimed,
                              output["abstention"], output["resolves"], tuple(warnings))


# ---------------------------------------------------------------------------------------------------- SC


@dataclass(frozen=True)
class Outcome:
    sc: int
    correct: int
    support_rate: float | None
    penalized: int
    abstained: bool


def outcome(answer: Answer, support: SupportVerdict | None, correctness: CorrectnessVerdict | None) -> Outcome:
    """SC and the supporting metrics for one answer. An unjudged answer (a verdict of None) scores 0."""
    abstained = answer.abstained or (correctness is not None and correctness.abstention == "full")
    ok = answer.status == "ok" and not abstained
    is_correct = bool(ok and correctness is not None and correctness.correct)
    supported = bool(ok and support is not None and support.all_supported)
    sc = int(supported and is_correct)
    # S3 is unjudged if either judge is (PLAN.md, "When a judge fails"): it scores 0 here, and the worst-case
    # rules treat it as unjudged. A partial claim alone is neither +1 nor -1.
    both = support is not None and correctness is not None
    unbacked = bool(ok and both and (support.any_unbacked or correctness.contradiction))
    penalized = 1 if sc else (-1 if unbacked else 0)
    rate = support.support_rate if (ok and support is not None) else None
    return Outcome(sc, int(is_correct), rate, penalized, abstained)
