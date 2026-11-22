"""The judge request lists: every candidate judge request, with the retries it may need, before any is sent.

PLAN.md, "The scope rule" and "When a judge fails". A judge request list holds,
for each answer to judge and each judge (F support, C correctness):
- the primary request, keyed by the judge and the exact answer it judges;
- one same-body retry (the next sample), used only if the primary is refused
  or its output is invalid;
- one truncation re-run (twice max_tokens, at most 16,000), used only if the
  primary stops at max_tokens.
Listing the retries up front means the authorized list already holds every
request a phase can send. Which items are judged is the scope decision, a
separate record; the list itself never changes after it is authorized.

Answers that need no judge are left out: a code-flagged full abstention scores 0
whatever a judge says, and C has no reference for a question no section answers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from raglab import params
from raglab.blocks import WindowDoc
from raglab.judge import Nugget, Planted, ReferenceSection, correctness_body, judge_condition, support_body
from raglab.plan4 import PlanLine, line_for
from raglab.units import Answer

ROLES = ("primary", "retry", "truncation")


@dataclass(frozen=True)
class JudgeLine:
    """A judge request list's line: the plan line, and what it is for."""

    line: PlanLine
    judge: str          # "F" or "C"
    answered: str       # the custom_id of the answer it judges
    role: str           # primary | retry | truncation
    item: str           # the scope item it belongs to, e.g. "core", "link1-run2", "closed-book"
    planted: str | None = None

    def to_json(self) -> dict:
        return {**self.line.to_json(), "judge": self.judge, "answered": self.answered, "role": self.role,
                "item": self.item, "planted": self.planted}


def _with_retries(judge: str, answered: str, question: str, body: Mapping, item: str, sample: int,
                  planted: str | None, mode: str = "batch") -> list[tuple[JudgeLine, Mapping]]:
    condition = judge_condition(judge, answered, planted)

    def line(s: int, b: Mapping):
        return line_for(condition, question, s, b, mode=mode)

    out = [(JudgeLine(line(sample, body), judge, answered, "primary", item, planted), body),
           (JudgeLine(line(sample + 1, body), judge, answered, "retry", item, planted), body)]
    longer = {**body, "max_tokens": min(int(body["max_tokens"]) * 2, params.MAX_TOKENS_BOUNDS["judge_truncation_retry"])}
    if longer["max_tokens"] > int(body["max_tokens"]):
        out.append((JudgeLine(line(sample, longer), judge, answered, "truncation", item, planted), longer))
    return out


def needs_judging(answer: Answer) -> bool:
    return answer.status == "ok" and not answer.abstained


def judge_requests(answered: str, question_id: str, question: str, answer: Answer, docs: Sequence[WindowDoc],
                   key: Sequence[Nugget] | None, reference: Sequence[ReferenceSection] | None, item: str,
                   max_tokens: Mapping[str, int], effort: str = "medium", support: bool = True, sample: int = 0,
                   planted: str | None = None, planted_excerpts: Planted | None = None,
                   correctness: bool = True, mode: str = "batch") -> list[tuple[JudgeLine, Mapping]]:
    """Every request (and its listed retries) that judging one answer may need.

    A planted item (`planted` names it, `planted_excerpts` holds what replaces
    one unit's excerpts) is judged by F alone: C never sees excerpts, so a plant
    can't change its verdict.
    """
    if not needs_judging(answer):
        return []
    if (planted is None) != (planted_excerpts is None):
        raise ValueError("a planted item needs both its name and its excerpts")
    out: list[tuple[JudgeLine, Mapping]] = []
    if support and answer.units:
        body = support_body(question, answer, docs, max_tokens["support"], effort, planted_excerpts)
        out += _with_retries("F", answered, question_id, body, item, sample, planted, mode)
    if planted is not None:
        return out
    if correctness and key and reference:
        body = correctness_body(question, key, reference, answer.text, max_tokens["correctness"], effort)
        out += _with_retries("C", answered, question_id, body, item, sample, planted, mode)
    return out


def write_judge_list(path, lines: Sequence[JudgeLine]) -> None:
    """A judge request list as committed: plan lines plus what each is for. No bodies (they hold doc text)."""
    import os

    from raglab.store import canonical

    check_unique(lines)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(canonical(jl.to_json()).decode("utf-8") + "\n" for jl in lines), encoding="utf-8")
    os.replace(tmp, path)


def read_judge_list(path) -> list[JudgeLine]:
    import json

    from raglab.plan4 import PlanLine

    out = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        d = json.loads(raw)
        extra = {k: d.pop(k) for k in ("judge", "answered", "role", "item", "planted")}
        out.append(JudgeLine(PlanLine(**d), **extra))
    return out


def check_unique(lines: Sequence[JudgeLine]) -> None:
    """No two lines may share a custom_id, whatever their role (PLAN.md, "Keys")."""
    seen: dict[str, JudgeLine] = {}
    for jl in lines:
        cid = jl.line.custom_id
        if cid in seen:
            raise ValueError(f"two judge requests share {cid}: {seen[cid].role} and {jl.role} for {jl.answered}")
        seen[cid] = jl
