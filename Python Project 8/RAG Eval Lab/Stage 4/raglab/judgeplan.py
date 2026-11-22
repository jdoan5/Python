"""The judge request lists: every judge request a phase may send, built from stored answers before it is authorized.

PLAN.md, "The pilot" (its judging order), "The scope rule" and "Checks on the
judges". A list holds, for every answer a phase may judge, each judge's
primary request, its same-body retry and its truncation re-run (judging.py).
Which items are sent is decided afterwards, without changing the list: the
pilot sends everything except the swap candidates its gate passes over; the
golden set follows plan/scope.json.

The pilot's items, in the order they are sent under the pilot-judging
authorization:
- smoke: two identical F calls and one C call on a smoke-test answer, first,
  so a broken structured output costs three calls, not the pilot;
- pilot-medium: F and C at medium on hhr runs 1 and 2 and on the baseline;
- pilot-noise: hhr run 1 judged again at medium (the next unused sample);
- pilot-swaps-medium, pilot-swaps-low: the near-miss swap candidates, F only;
- pilot-low: F and C at low on hhr run 1 and on the baseline;
- pilot-closed: C on the closed-book answers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from raglab import params
from raglab.blocks import WindowDoc
from raglab.corpus import Document
from raglab.judge import Nugget, ReferenceSection
from raglab.judging import JudgeLine, check_unique, judge_requests
from raglab.plan4 import PlanLine
from raglab.planted import SWAPS, Candidate, candidates_for, draw, planted_excerpts
from raglab.sections import SectionIndex
from raglab.sets import AskedQuestion
from raglab.store import Record
from raglab.units import Answer, parse_reply

PILOT_ITEMS = ("smoke", "pilot-medium", "pilot-noise", "pilot-swaps-medium", "pilot-swaps-low", "pilot-low",
               "pilot-closed")
PILOT_ARMS = (("hhr-4.8k-med", 0), ("base-4.8k-med", 0))  # hhr run 1 and the baseline: the swaps' and low's arms
PILOT_SWAPS = {"swap-lexical": params.PILOT["lexical_per_arm"],
               "swap-topical": params.PILOT["swaps_per_arm"] - params.PILOT["lexical_per_arm"]}  # per arm
SAMPLES = {"primary": 0, "noise": 2, "smoke_again": 2}  # sample 1 is the primary's retry; 3 is the re-judge's
SMOKE_JUDGED = ("smoke:hhr-4.8k-med", "smoke:base-4.8k-med", "smoke:base-300-documents")  # the fallback order


@dataclass(frozen=True)
class Stored:
    """A stored generation answer, parsed against the documents its request carried."""

    line: PlanLine
    answer: Answer
    docs: tuple[WindowDoc, ...]


def stored_answers(records: Iterable[Record], built: Mapping[str, tuple[PlanLine, Mapping, Sequence[WindowDoc]]]
                   ) -> dict[str, Stored]:
    """Every succeeded answer in the store whose request the code rebuilds, keyed by custom_id."""
    out = {}
    for rec in records:
        if rec.outcome != "succeeded" or rec.custom_id not in built:
            continue
        line, _, docs = built[rec.custom_id]
        if rec.body_sha256 != line.body_sha256:
            raise ValueError(f"{rec.custom_id}: the stored answer's request isn't the one the code rebuilds")
        out[rec.custom_id] = Stored(line, parse_reply(rec.response, docs), tuple(docs))
    return out


def reference_sections(locations: Sequence[str], by_id: Mapping[str, Document],
                       index: SectionIndex) -> list[ReferenceSection]:
    """A key's reference as C sees it: each section whole (its heading line and body), in the order frozen with
    the key, which is document order (PLAN.md, "Keys and references")."""
    out = []
    for loc in locations:
        ref = index.get(loc)
        if ref is None:
            raise ValueError(f"the reference names {loc}, which the corpus doesn't have")
        doc = by_id[ref.doc_id]
        section = next(s for s in doc.sections if s.start <= ref.start < s.end)
        out.append(ReferenceSection(loc, " > ".join(ref.heading_path), doc.text[section.start:ref.end].strip()))
    return out


def key_nuggets(entry: Mapping) -> list[Nugget]:
    return [Nugget(n["id"], n["text"], bool(n["essential"]), tuple(n.get("alternatives") or ()))
            for n in entry["nuggets"]]


@dataclass(frozen=True)
class Graded:
    """What C needs for a question: its key and its reference (none for a question no section answers)."""

    key: tuple[Nugget, ...] | None
    reference: tuple[ReferenceSection, ...] | None


def graded_questions(questions: Sequence[AskedQuestion], keys: Mapping[str, Mapping], by_id, index) -> dict:
    out = {}
    for q in questions:
        entry = keys.get(q.id)
        if q.target is None or entry is None:
            out[q.id] = Graded(None, None)
        else:
            out[q.id] = Graded(tuple(key_nuggets(entry)), tuple(reference_sections(entry["reference"], by_id, index)))
    return out


def _requests(s: Stored, cid: str, q: AskedQuestion, g: Graded, item: str, effort: str, sample: int,
              max_tokens: Mapping[str, int], support: bool = True, correctness: bool = True,
              candidate: Candidate | None = None) -> list[tuple[JudgeLine, Mapping]]:
    return judge_requests(cid, q.id, q.text, s.answer, s.docs, g.key, g.reference, item, max_tokens, effort,
                          support=support, sample=sample, correctness=correctness,
                          planted=candidate.name if candidate else None,
                          planted_excerpts=planted_excerpts(candidate) if candidate else None,
                          mode="standard" if item == "smoke" else "batch")  # the smoke test is standard API


def _find(stored: Mapping[str, Stored], condition: str, question: str, sample: int) -> tuple[str, Stored] | None:
    for cid, s in stored.items():
        if s.line.condition == condition and s.line.question == question and s.line.sample == sample:
            return cid, s
    return None


def pilot_candidates(stored: Mapping[str, Stored], questions: Sequence[AskedQuestion],
                     grades: Mapping[str, Mapping[str, int]], by_id, index) -> list[Candidate]:
    """The pilot's swap candidates: lexical and topical, in hhr run 1 and the baseline, one per question each."""
    found = []
    for q in questions:
        for arm, sample in PILOT_ARMS:
            hit = _find(stored, arm, q.id, sample)
            if hit is None:
                continue
            cid, s = hit
            for kind in SWAPS:
                found += candidates_for(kind, arm, q.id, cid, s.answer, s.docs, grades.get(q.id, {}), q.target,
                                        by_id, index)
    return draw(found, per_kind=len(questions))  # oversampled: the gate takes the first confirmed and supported


def pilot_judge_list(stored: Mapping[str, Stored], questions: Sequence[AskedQuestion], graded: Mapping[str, Graded],
                     candidates: Sequence[Candidate], confirmed: Mapping[str, bool]) -> list[tuple[JudgeLine, Mapping]]:
    """Every request the pilot's judging may send, in PLAN.md's order."""
    mt = {"support": params.MAX_TOKENS_PILOT["judge"], "correctness": params.MAX_TOKENS_PILOT["judge"]}
    by_id = {q.id: q for q in questions}
    out: list[tuple[JudgeLine, Mapping]] = []

    # smoke: the first smoke answer, in this order, that has a claim to judge and a key to grade it by
    for name in SMOKE_JUDGED:
        hits = sorted(((cid, s) for cid, s in stored.items() if s.line.condition == name), key=lambda h: h[0])
        usable = [(cid, s) for cid, s in hits if s.answer.status == "ok" and not s.answer.abstained
                  and s.answer.units and graded[s.line.question].key]
        if usable:
            cid, s = usable[0]
            q, g = by_id[s.line.question], graded[s.line.question]
            out += _requests(s, cid, q, g, "smoke", "medium", SAMPLES["primary"], mt)
            out += _requests(s, cid, q, g, "smoke", "medium", SAMPLES["smoke_again"], mt, correctness=False)
            break
    else:
        raise ValueError("no smoke-test answer can be judged (every one abstained, failed or has no key): the "
                         "smoke test's judge check can't run, so the developer decides, as a logged deviation")

    def each(conditions: Sequence[tuple[str, int]], item: str, effort: str, sample: int, **kw) -> None:
        nonlocal out
        for q in questions:
            for condition, run in conditions:
                hit = _find(stored, condition, q.id, run)
                if hit:
                    out += _requests(hit[1], hit[0], q, graded[q.id], item, effort, sample, mt, **kw)

    each((("hhr-4.8k-med", 0), ("hhr-4.8k-med", 1), ("base-4.8k-med", 0)), "pilot-medium", "medium",
         SAMPLES["primary"])
    each((("hhr-4.8k-med", 0),), "pilot-noise", "medium", SAMPLES["noise"])
    for effort in ("medium", "low"):
        for c in candidates:
            if not confirmed.get(f"{c.answered}:{c.name}", False):
                continue
            s = stored[c.answered]
            out += _requests(s, c.answered, by_id[c.question], graded[c.question], f"pilot-swaps-{effort}", effort,
                             SAMPLES["primary"], mt, correctness=False, candidate=c)
    each(PILOT_ARMS, "pilot-low", "low", SAMPLES["primary"])
    each((("closed-med", 0),), "pilot-closed", "medium", SAMPLES["primary"], support=False)
    check_unique([jl for jl, _ in out])
    return out


def write_candidates(path: Path, candidates: Sequence[Candidate]) -> None:
    """The candidates as committed: where each text lies in the corpus, never the text itself."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([c.to_json() for c in candidates], indent=1) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------------------------------- golden draws


def golden_needed(hhr: str = "hhr-4.8k-med", base: str = "base-4.8k-med") -> dict[tuple[str, str], int]:
    """How many of each planted kind the golden gates judge, per arm (PLAN.md, "Checks on the judges")."""
    p = params.PLANTED
    out = {}
    for arm in (hhr, base):
        out[("swap-lexical", arm)] = p["lexical_per_arm"]
        out[("swap-topical", arm)] = p["swaps_per_arm"] - p["lexical_per_arm"]
        out[("plant-cross", arm)] = p["cross_section_plants_per_arm"]
    out[("plant-shape", hhr)] = p["shape_plants"]
    return out


def golden_candidates(found: Sequence[Candidate]) -> list[Candidate]:
    """The oversampled candidates listed at freeze point 4: at most one per question, 30 per kind and arm."""
    return draw(found, per_kind=params.PLANTED["candidates_per_kind"])


def masked_draw(answers: Mapping[str, tuple[str, str, bool]]) -> list[str]:
    """The 40 judged masked answers: custom_id -> (question, S's verdict on its window, abstains-like). Drawn
    from answers a paraphrase-tolerant detector doesn't flag, in sha256 order, first from windows S calls "no",
    then from "partly" ones."""
    import hashlib

    def order(cid: str) -> str:
        return hashlib.sha256(f"masked:{cid}".encode("utf-8")).hexdigest()

    out: list[str] = []
    for s_verdict in ("no", "partly"):
        pool = sorted((cid for cid, (_, v, abstains) in answers.items() if v == s_verdict and not abstains), key=order)
        out += pool[:params.MASKED_JUDGED - len(out)]
    return out
