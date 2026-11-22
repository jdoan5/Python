"""Each condition's context: which chunks a question's request carries (PLAN.md, "Conditions and runs").

The pipelines are Stage 3's configurations, unchanged:
- hhr: heading chunks with the breadcrumb, hybrid+rerank (Stage 3's CHALLENGER);
- base: RAG Mini Chat as it shipped (Stage 3's BASELINE);
- bm25: heading+bm25; hybrid: heading+hybrid.
A window is Stage 3's `pack` of a pipeline's full ranking to a character budget.

Built from them:
- masked: hhr's full ranking with every chunk that retrieves one of the
  question's grade-2 sections removed, re-packed to 4.8k. Only for questions
  whose hhr 4.8k window retrieves one (there is nothing to mask otherwise);
- oracle: the chunks that retrieve a grade-2 section, those retrieving the
  target first in document order, then the rest in hhr's rank order, packed;
- closed: no documents at all.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping, Sequence

from raglab import params
from raglab.evaluate import ranked_spans
from raglab.experiments import BASELINE, CHALLENGER, INCUMBENT, RetrievalConfig, build_retrievers
from raglab.metrics import Span, pack
from raglab.sections import SectionIndex

EFFORTS = {"low": "low", "med": "medium", "high": "high", "xhigh": "xhigh", "max": "max", "off": "medium"}
HYBRID = RetrievalConfig("heading", "hybrid")
PIPELINES = {"hhr": CHALLENGER, "base": BASELINE, "bm25": INCUMBENT, "hybrid": HYBRID}
_NAME = re.compile(r"^(?P<pipeline>hhr|base|bm25|hybrid|masked|oracle)-(?P<budget>\d+(?:\.\d+)?)k-"
                   r"(?P<effort>low|med|high|xhigh|max|off)(?P<notitle>-notitle)?$")


@dataclass(frozen=True)
class Condition:
    name: str
    pipeline: str          # hhr | base | bm25 | hybrid | masked | oracle | closed
    budget: int | None     # characters of chunk text; None for closed-book
    effort: str            # the API's output_config.effort
    thinking: str          # adaptive | disabled
    titles: bool
    runs: int

    @property
    def closed(self) -> bool:
        return self.pipeline == "closed"


def parse_condition(name: str) -> Condition:
    runs = params.RUNS.get(name, 1)
    if name == "closed-med":
        return Condition(name, "closed", None, "medium", "adaptive", False, runs)
    m = _NAME.match(name)
    if not m:
        raise ValueError(f"not a Stage 4 condition: {name!r}")
    budget = round(float(m["budget"]) * 1000)
    off = m["effort"] == "off"
    return Condition(name, m["pipeline"], budget, EFFORTS[m["effort"]], "disabled" if off else "adaptive",
                     not m["notitle"], runs)


CONDITIONS: dict[str, Condition] = {name: parse_condition(name) for name in params.CONDITIONS}


class Windows:
    """Rankings computed once per pipeline and question; windows packed from them on demand."""

    def __init__(self, docs, index: SectionIndex, retrievers=None) -> None:
        self.index = index
        self.retrievers = retrievers if retrievers is not None else build_retrievers(
            list(docs), tuple(PIPELINES.values()))
        self._rankings: dict[tuple[str, str], list[Span]] = {}

    def ranking(self, pipeline: str, question: str) -> list[Span]:
        key = (pipeline, question)
        if key not in self._rankings:
            self._rankings[key] = ranked_spans(self.retrievers[PIPELINES[pipeline].name], question)
        return self._rankings[key]

    def _answers(self, spans: Sequence[Span], answering: frozenset[str]) -> list[bool]:
        return [any(ref.location in answering for ref, _ in self.index.retrieved_by(*span)) for span in spans]

    def window(self, condition: Condition, question: str, answering: frozenset[str] = frozenset(),
               target: str | None = None) -> list[Span] | None:
        """The packed window, [] for closed-book, or None when the condition has no window for this question."""
        if condition.closed:
            return []
        if condition.pipeline in PIPELINES:
            return pack(self.ranking(condition.pipeline, question), condition.budget)
        ranked = self.ranking("hhr", question)
        hits = self._answers(ranked, answering)
        if condition.pipeline == "masked":
            if not any(self._answers(pack(ranked, condition.budget), answering)):
                return None
            return pack([s for s, hit in zip(ranked, hits) if not hit], condition.budget)
        if condition.pipeline == "oracle":
            chosen = [s for s, hit in zip(ranked, hits) if hit]
            if target is None:
                raise ValueError("the oracle window needs the target section")
            on_target = sorted((s for s in chosen if any(r.location == target for r, _ in self.index.retrieved_by(*s))),
                               key=lambda s: (s[0], s[1]))
            rest = [s for s in chosen if s not in on_target]
            return pack(on_target + rest, condition.budget)
        raise ValueError(f"unknown pipeline {condition.pipeline!r}")


def window_facts(spans: Sequence[Span], answering: frozenset[str], index: SectionIndex) -> dict:
    """What a window holds, by section labels (S, the sufficiency audit, is the other view)."""
    delivered = {ref.location for span in spans for ref, _ in index.retrieved_by(*span)}
    return {"chunks": len(spans), "chars": sum(e - s for _, s, e in spans),
            "answering_sections": sorted(delivered & answering)}


def answering_sections(judgments: Mapping[str, int]) -> frozenset[str]:
    return frozenset(loc for loc, grade in judgments.items() if grade == 2)
