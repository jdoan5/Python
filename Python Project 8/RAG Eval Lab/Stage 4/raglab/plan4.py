"""The request lists: every planned request, its key and its worst case, before any is sent.

A plan line names a request without carrying its body: the body holds
documentation text, which this repository never commits, and it can be rebuilt
exactly from the line's recipe (condition, question, sample) with the pinned
corpus and the code. Each line records the body's SHA-256, so the body sent is
checked to be the body planned. Input tokens come from the free count_tokens
endpoint; the bound is the ledger's worst case for that request.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from decimal import Decimal
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

from raglab import params
from raglab.ledger import Pricing, input_bound, money
from raglab.runner import Request
from raglab.store import body_sha256, canonical, custom_id, request_key


@dataclass(frozen=True)
class PlanLine:
    custom_id: str
    request_key: str
    condition: str
    question: str
    sample: int
    model: str
    body_sha256: str
    max_tokens: int
    counted_input: int | None = None
    input_tokens: int | None = None   # counted plus the margin: what the ledger reserves for
    bound: str | None = None          # the worst case at this line's prices, in dollars
    mode: str = "batch"               # batch | standard | stream
    role: str = "answer"              # answer | smoke | timing

    def to_json(self) -> dict:
        return asdict(self)


MODES = ("batch", "standard", "stream")


def line_for(condition: str, question: str, sample: int, body: Mapping, mode: str = "batch",
             role: str = "answer") -> PlanLine:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    key = request_key(condition, body)
    return PlanLine(custom_id(key, sample), key, condition, question, sample, body["model"], body_sha256(body),
                    int(body["max_tokens"]), mode=mode, role=role)


def counted(line: PlanLine, tokens: int, pricing: Pricing, cache_marked: bool = False) -> PlanLine:
    bounded = input_bound(tokens, params.INPUT_MARGIN, params.INPUT_SLACK)
    bound = pricing.bound(line.model, bounded, line.max_tokens, cache_marked,
                          "batch" if line.mode == "batch" else "standard")
    return PlanLine(**{**line.to_json(), "counted_input": tokens, "input_tokens": bounded, "bound": money(bound)})


def write_plan(path: Path, lines: Sequence[PlanLine]) -> None:
    ids = [l.custom_id for l in lines]
    if len(set(ids)) != len(ids):
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        raise ValueError(f"two planned requests share a custom_id, e.g. {dupes[0]}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(canonical(l.to_json()).decode("utf-8") + "\n" for l in lines), encoding="utf-8")
    os.replace(tmp, path)


def read_plan(path: Path) -> list[PlanLine]:
    return [PlanLine(**json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def total_bound(lines: Iterable[PlanLine]) -> Decimal:
    return sum((Decimal(l.bound) for l in lines if l.bound is not None), Decimal(0))


def as_request(line: PlanLine, body: Mapping) -> Request:
    """The runner's request for a plan line, after checking the rebuilt body is the one planned."""
    if body_sha256(body) != line.body_sha256:
        raise ValueError(f"{line.custom_id}: the rebuilt body differs from the planned one (code or corpus changed)")
    if line.input_tokens is None:
        raise ValueError(f"{line.custom_id}: its input tokens were never counted")
    req = Request(line.condition, body, line.sample, line.input_tokens)
    if req.custom_id != line.custom_id:
        raise ValueError(f"{line.custom_id}: the request key changed")
    return req


def count_all(lines: Sequence[PlanLine], bodies: Mapping[str, Mapping], count: Callable[[Mapping], int],
              pricing: Pricing) -> list[PlanLine]:
    """Count every request's input tokens (free), once per distinct body; each line is bounded at its own mode."""
    from raglab.runner import _has_cache_control

    cache: dict[str, int] = {}
    out = []
    for line in lines:
        if line.body_sha256 not in cache:
            cache[line.body_sha256] = int(count(bodies[line.custom_id]))
        out.append(counted(line, cache[line.body_sha256], pricing, _has_cache_control(bodies[line.custom_id])))
    return out
