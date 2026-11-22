"""From stored judge outputs to verdicts, following PLAN.md's "When a judge fails".

A judgment is one judge's verdict on one answer in one scope item. Its listed
requests are a primary, a same-body retry (the next sample) and a truncation
re-run (twice max_tokens). The rule:
- a primary whose output validates is the verdict;
- a refusal or an invalid output sends the retry; a `max_tokens` stop sends
  the truncation re-run;
- an API error (errored, expired or canceled in a batch) sends the same
  request once more;
- the second request's output is the verdict if it validates; any second
  failure leaves the answer unjudged by that judge.
A verdict that passed validation is never asked for again.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

from raglab.judge import InvalidVerdict, read_output
from raglab.judging import JudgeLine
from raglab.store import Store

RESUBMITS = 1  # an API error is sent once more under the same custom_id


@dataclass(frozen=True)
class Judgment:
    """One judge's verdict on one answer in one item, or why there is none yet."""

    judge: str
    answered: str
    item: str
    planted: str | None
    state: str                    # verdict | send | wait | unjudged
    verdict: object | None = None
    used: str | None = None       # the custom_id whose output is the verdict
    next: JudgeLine | None = None  # for state "send": the request to send next
    log: tuple[str, ...] = field(default=())


def groups(lines: Sequence[JudgeLine]) -> dict[tuple, dict[str, JudgeLine]]:
    """The judge list's lines by judgment: (judge, answered, item, planted, primary sample) -> role -> line."""
    out: dict[tuple, dict[str, JudgeLine]] = {}
    primaries = {}
    for jl in lines:
        if jl.role == "primary":
            key = (jl.judge, jl.answered, jl.item, jl.planted, jl.line.sample)
            out.setdefault(key, {})["primary"] = jl
            primaries[(jl.judge, jl.answered, jl.item, jl.planted, jl.line.sample)] = key
    for jl in lines:
        if jl.role == "retry":
            key = primaries.get((jl.judge, jl.answered, jl.item, jl.planted, jl.line.sample - 1))
        elif jl.role == "truncation":
            key = primaries.get((jl.judge, jl.answered, jl.item, jl.planted, jl.line.sample))
        else:
            continue
        if key is None:
            raise ValueError(f"{jl.line.custom_id}: a {jl.role} line without its primary")
        out[key][jl.role] = jl
    return out


def _attempt(store: Store, failures: Mapping[str, int], jl: JudgeLine, validate: Callable) -> tuple[str, object, str]:
    """What one listed request came to: ("valid", verdict, note), ("wait", ...), ("resend", ...),
    or a failure kind ("refused", "invalid", "truncated", "error")."""
    cid = jl.line.custom_id
    rec = store.get(cid)
    if rec is None:
        n = failures.get(cid, 0)
        if n == 0:
            return "wait", None, f"{cid}: not answered yet"
        if n <= RESUBMITS:
            return "resend", None, f"{cid}: an API error; sent once more"
        return "error", None, f"{cid}: an API error twice"
    message = rec.response or {}
    stop = message.get("stop_reason")
    if stop == "refusal":
        return "refused", None, f"{cid}: refused"
    if stop == "max_tokens":
        return "truncated", None, f"{cid}: stopped at max_tokens"
    try:
        return "valid", validate(read_output(message)), f"{cid}: valid"
    except InvalidVerdict as e:
        return "invalid", None, f"{cid}: invalid output ({e})"


def resolve(roles: Mapping[str, JudgeLine], store: Store, failures: Mapping[str, int],
            validate: Callable[[Mapping], object]) -> Judgment:
    """Where one judgment stands. `failures` counts each custom_id's settled API errors (from the ledger)."""
    primary = roles["primary"]
    base = dict(judge=primary.judge, answered=primary.answered, item=primary.item, planted=primary.planted)
    kind, verdict, note = _attempt(store, failures, primary, validate)
    log = [note]
    if kind == "valid":
        return Judgment(**base, state="verdict", verdict=verdict, used=primary.line.custom_id, log=tuple(log))
    if kind == "wait":
        return Judgment(**base, state="send", next=primary, log=tuple(log))
    if kind == "resend":
        return Judgment(**base, state="send", next=primary, log=tuple(log))
    second = roles.get("truncation") if kind == "truncated" else roles.get("retry")
    if second is None:
        return Judgment(**base, state="unjudged", log=tuple(log + ["no listed request for a second try"]))
    kind2, verdict2, note2 = _attempt(store, failures, second, validate)
    log.append(note2)
    if kind2 == "valid":
        return Judgment(**base, state="verdict", verdict=verdict2, used=second.line.custom_id, log=tuple(log))
    if kind2 in ("wait", "resend"):
        return Judgment(**base, state="send", next=second, log=tuple(log))
    return Judgment(**base, state="unjudged", log=tuple(log))


TRANSIENT = ("rate_limited", "server")  # retried by the runner within one standard call, never billed


def failures_from(ledger_items: Mapping) -> dict[str, int]:
    """Each custom_id's settled API errors: what its one resubmission counts against.

    Counted: an errored or expired batch item (PLAN.md, "When a judge fails"),
    and a standard call refused outright. Not counted: a canceled item (the
    developer stopped it), and the rate limits and server errors the runner
    retries inside one standard call, which are never billed.
    """
    out: dict[str, int] = {}
    for (_, cid), item in ledger_items.items():
        if item.status != "settled" or item.outcome not in ("errored", "expired"):
            continue
        if item.mode == "standard" and item.detail in TRANSIENT:
            continue
        out[cid] = out.get(cid, 0) + 1
    return out
