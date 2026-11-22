"""The cost guard: nothing billed is sent unless its worst case fits under the caps.

Stage 4 spends real money, so every billed request goes through this ledger.

1. **Reserve** before sending. A request's bound is its worst case: its input
   tokens as the API's free count_tokens endpoint counts them, plus a margin,
   all priced at the cache-write rate when the request carries a cache marker;
   plus `max_tokens` at the output rate. Thinking is billed as output and counts
   toward `max_tokens`, so `max_tokens` bounds it. Batches use the batch rate.
   A reservation is admitted only if settled spend, plus every open
   reservation, plus the new bound stays within the stage cap and within its
   phase's cap.
2. **Settle** each request from the usage its response reports. A batch item
   that errored, was canceled or expired is not billed, so it settles at $0.
3. **Unknown**: a standard request whose connection dropped after sending may
   have been billed, so it stays spent at its bound.
4. **Release** a reservation that was never sent: the process stopped before
   sending, or `batch reconcile` shows a batch create never reached the API.

The ledger is an append-only JSONL file, one fsynced line per event, so a crash
loses at most the line being written and never forgets an open reservation: it
counts against the caps until it is settled or released. One process at a time
holds the ledger (an exclusive lock on a sibling file). Every line is written
twice, to the ledger and then to a mirror outside the repository, so losing one
copy can't reset the spend to $0. On opening:
- the two copies must agree, except that the mirror may lack the ledger's last
  line (a crash between the two writes), which is copied over;
- a torn last line, a missing copy or any other difference is refused until
  `budget repair` is run and its change reviewed.
"""

from __future__ import annotations

import fcntl
import json
import math
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

MILLION = Decimal(1_000_000)
MODES = ("standard", "batch")
OUTCOMES = ("succeeded", "errored", "canceled", "expired")
# Usage fields the price table covers. Any other positive count is flagged on settlement, never ignored silently.
_PRICED = ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")
# Fields that carry no count of their own, or whose values are checked one by one below.
_UNPRICED_OK = ("cache_creation", "service_tier", "inference_geo", "server_tool_use", "iterations", "speed",
                "output_tokens_details")
KNOWN_USAGE_FIELDS = frozenset(_PRICED + _UNPRICED_OK)  # every other usage field with a count is flagged


class LedgerError(RuntimeError):
    """The ledger can't be read or written as asked; nothing was sent."""


class CapExceeded(LedgerError):
    """Admitting this reservation could take spend past a cap."""


def money(x: Decimal) -> str:
    """Exact, human-readable decimal (no exponent), the form every amount is stored in."""
    return format(x.normalize(), "f") if x else "0"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------------------------------- prices

@dataclass(frozen=True)
class Rates:
    """One model's list prices, in dollars per token."""

    input: Decimal
    output: Decimal
    cache_write_5m: Decimal
    cache_write_1h: Decimal
    cache_read: Decimal
    max_prompt_tokens: int | None = None  # above this the model has a second, dearer rate card: refused


@dataclass(frozen=True)
class Pricing:
    as_of: str
    batch_multiplier: Decimal
    models: Mapping[str, Rates]
    us_inference_multiplier: Decimal = Decimal("1.1")

    @classmethod
    def load(cls, path: Path) -> "Pricing":
        raw = json.loads(path.read_text(encoding="utf-8"))
        models = {}
        for name, r in raw["per_million_tokens"].items():
            models[name] = Rates(
                input=Decimal(r["input"]) / MILLION,
                output=Decimal(r["output"]) / MILLION,
                cache_write_5m=Decimal(r["cache_write_5m"]) / MILLION,
                cache_write_1h=Decimal(r["cache_write_1h"]) / MILLION,
                cache_read=Decimal(r["cache_read"]) / MILLION,
                max_prompt_tokens=r.get("max_prompt_tokens"),
            )
        multiplier = Decimal(raw["batch_multiplier"])
        if not Decimal(0) < multiplier <= Decimal(1):
            raise LedgerError(f"batch_multiplier must be in (0, 1], not {multiplier}")
        us = Decimal(raw.get("us_inference_multiplier", "1.1"))
        if us < 1:
            raise LedgerError(f"us_inference_multiplier must be at least 1, not {us}")
        return cls(as_of=raw["as_of"], batch_multiplier=multiplier, models=models, us_inference_multiplier=us)

    def rates(self, model: str) -> Rates:
        try:
            return self.models[model]
        except KeyError:
            raise LedgerError(f"no price for model {model!r}; pricing.json lists {sorted(self.models)}") from None

    def _mode(self, mode: str) -> Decimal:
        if mode not in MODES:
            raise LedgerError(f"mode must be one of {MODES}, not {mode!r}")
        return self.batch_multiplier if mode == "batch" else Decimal(1)

    def bound(self, model: str, input_tokens: int, max_tokens: int, cache_marked: bool, mode: str) -> Decimal:
        """The most a request can cost: every input token at the dearest input rate it could be billed at."""
        r = self.rates(model)
        if input_tokens < 0 or max_tokens <= 0:
            raise LedgerError(f"bad token bound: input {input_tokens}, max_tokens {max_tokens}")
        if r.max_prompt_tokens is not None and input_tokens > r.max_prompt_tokens:
            raise LedgerError(f"{model}: a {input_tokens}-token prompt is priced on the second rate card; refused")
        per_input = max(r.input, r.cache_write_5m) if cache_marked else r.input
        # The workspace's default inference geography isn't known here, so every bound assumes US-only (1.1x).
        return (input_tokens * per_input + max_tokens * r.output) * self._mode(mode) * self.us_inference_multiplier

    def usage_cost(self, model: str, usage: Mapping, mode: str) -> tuple[Decimal, list[str]]:
        """What a response's usage costs, and any usage field this table can't price (flagged, not ignored)."""
        r = self.rates(model)
        flags = []
        for key, value in usage.items():
            if key in _PRICED or key in _UNPRICED_OK:
                continue
            if isinstance(value, (int, float)) and value > 0:
                flags.append(f"unpriced usage field {key}={value}")
        written = int(usage.get("cache_creation_input_tokens") or 0)
        detail = usage.get("cache_creation") or {}
        written_1h = int(detail.get("ephemeral_1h_input_tokens") or 0) if isinstance(detail, Mapping) else 0
        if written_1h > written:
            raise LedgerError(f"usage reports {written_1h} 1-hour cache writes but {written} cache writes in all")
        cost = (int(usage.get("input_tokens") or 0) * r.input
                + (written - written_1h) * r.cache_write_5m
                + written_1h * r.cache_write_1h
                + int(usage.get("cache_read_input_tokens") or 0) * r.cache_read
                + int(usage.get("output_tokens") or 0) * r.output)
        if written_1h:
            flags.append(f"{written_1h} tokens written to the 1-hour cache, which this stage never asks for")
        geo = usage.get("inference_geo")
        multiplier = Decimal(1)
        if geo == "us":
            multiplier = self.us_inference_multiplier
            flags.append("served with US-only inference (billed at 1.1x): set the workspace's default to global")
        elif geo not in (None, "global", "not_available"):
            flags.append(f"unknown inference_geo {geo!r}")
        if usage.get("service_tier") not in (None, "standard", "batch"):
            flags.append(f"service_tier {usage.get('service_tier')!r}")
        if usage.get("speed") not in (None, "standard"):
            flags.append(f"speed {usage.get('speed')!r}")
        iterations = usage.get("iterations")
        if isinstance(iterations, list) and len(iterations) > 1:
            flags.append(f"{len(iterations)} iterations in one response (a fallback ran?)")
        return cost * self._mode(mode) * multiplier, flags


def input_bound(counted: int, margin: Decimal, slack: int) -> int:
    """count_tokens' figure plus a safety margin: the input side of a reservation."""
    if counted < 0:
        raise LedgerError(f"negative token count {counted}")
    return math.ceil(counted * (1 + margin)) + slack


# ---------------------------------------------------------------------------------------------------- caps

@dataclass(frozen=True)
class Caps:
    total: Decimal
    phases: Mapping[str, Decimal]

    def phase(self, name: str) -> Decimal:
        try:
            return self.phases[name]
        except KeyError:
            raise LedgerError(f"no cap for phase {name!r}; the plan's phases are {sorted(self.phases)}") from None

    def __post_init__(self) -> None:
        if self.total <= 0 or any(cap <= 0 for cap in self.phases.values()):
            raise LedgerError("every cap must be positive")


# ---------------------------------------------------------------------------------------------------- state

@dataclass
class Item:
    group: str
    custom_id: str
    model: str
    phase: str
    mode: str
    bound: Decimal
    status: str = "reserved"  # reserved -> sent -> settled | unknown; reserved -> released
    cost: Decimal | None = None
    outcome: str | None = None
    flags: list[str] = field(default_factory=list)
    detail: str | None = None  # for an errored standard call: the error's kind (rate_limited, server, client)

    @property
    def committed(self) -> Decimal:
        """What this item counts against the caps: its cost once settled, its bound until then."""
        if self.status == "settled":
            return self.cost or Decimal(0)
        if self.status == "released":
            return Decimal(0)
        return self.bound


@dataclass
class State:
    items: dict[tuple[str, str], Item] = field(default_factory=dict)
    groups: dict[str, dict] = field(default_factory=dict)
    cleared: set[tuple[str, str]] = field(default_factory=set)  # unknown attempts a "resend" event let go of
    acknowledged: set[tuple[str, str]] = field(default_factory=set)  # flagged settlements a person has accepted
    accepted: set[tuple[str, str]] = field(default_factory=set)  # (phase, flag kind) accepted for a whole phase

    def stopping_flags(self, item: Item) -> list[str]:
        """An item's flags that still stop a run: all but those of a kind accepted for its phase."""
        return [f for f in item.flags if (item.phase, flag_kind(f)) not in self.accepted]

    def unacknowledged(self) -> list[Item]:
        """Settled items whose flags nobody has accepted: they stop every new reservation (PLAN.md, "The guard")."""
        out = []
        for (g, c), i in self.items.items():
            if i.status != "settled" or not i.flags or (g, c) in self.acknowledged:
                continue
            if all((i.phase, flag_kind(f)) in self.accepted for f in i.flags):
                continue
            out.append(i)
        return out

    def blocking(self, custom_id: str) -> list[Item]:
        """Attempts of this request that are still open, or unknown and not cleared for a resend."""
        return [i for (g, c), i in self.items.items() if c == custom_id
                and (i.status in ("reserved", "sent") or (i.status == "unknown" and (g, c) not in self.cleared))]

    def answered(self, custom_id: str) -> list[Item]:
        """Attempts of this request that succeeded: a request is paid for once (a new sample is a new custom_id)."""
        return [i for (_, c), i in self.items.items() if c == custom_id and i.status == "settled"
                and i.outcome == "succeeded"]

    def committed(self, phase: str | None = None) -> Decimal:
        return sum((i.committed for i in self.items.values() if phase is None or i.phase == phase), Decimal(0))

    def settled(self, phase: str | None = None) -> Decimal:
        return sum((i.cost or Decimal(0) for i in self.items.values()
                    if i.status == "settled" and (phase is None or i.phase == phase)), Decimal(0))

    def open_items(self) -> list[Item]:
        return [i for i in self.items.values() if i.status in ("reserved", "sent")]

    def unknown(self) -> list[Item]:
        return [i for i in self.items.values() if i.status == "unknown"]


FLAG_KINDS = ("us_inference", "other")


def flag_kind(flag: str) -> str:
    """US-only inference may be accepted once for a whole phase; every other flag is accepted item by item."""
    return "us_inference" if flag.startswith("served with US-only inference") else "other"


def _check(state: State, event: Mapping, line: int) -> None:
    """Raise if this event can't follow the state. Mutates nothing, so a refused event leaves no trace."""
    kind = event.get("event")
    if kind == "reserve":
        group = event["group"]
        if group in state.groups:
            raise LedgerError(f"line {line}: group {group} reserved twice")
        if event["mode"] not in MODES:
            raise LedgerError(f"line {line}: unknown mode {event['mode']!r}")
        ids = [it["custom_id"] for it in event["items"]]
        if not ids or len(set(ids)) != len(ids):
            raise LedgerError(f"line {line}: a reservation needs distinct custom_ids")
        if not event.get("authorization"):
            raise LedgerError(f"line {line}: a reservation must name the authorization it was made under")
        for cid in ids:
            open_attempts = state.blocking(cid)
            if open_attempts:
                a = open_attempts[0]
                raise LedgerError(f"line {line}: {cid} already has an attempt that is {a.status} in group {a.group}; "
                                  "collect or reconcile it first (an unknown one needs an explicit resend)")
            answered = state.answered(cid)
            if answered:
                raise LedgerError(f"line {line}: {cid} was already answered (group {answered[0].group}), so its "
                                  "response belongs in the store: restore the store (a batch's results can be "
                                  "fetched again) rather than pay for it twice")
        return
    if kind == "acknowledge":
        if "phase" in event:
            if event.get("kind") != "us_inference":
                raise LedgerError(f"line {line}: only US-only inference can be accepted for a whole phase")
            return
        item = state.items.get((event.get("group"), event.get("custom_id")))
        if item is None or item.status != "settled" or not item.flags:
            raise LedgerError(f"line {line}: only a flagged settlement can be acknowledged")
        return
    if kind == "resend":
        key = (event.get("group"), event.get("custom_id"))
        item = state.items.get(key)
        if item is None or item.status != "unknown":
            raise LedgerError(f"line {line}: only an unknown attempt can be cleared for a resend")
        if key in state.cleared:
            raise LedgerError(f"line {line}: {key[1]} was already cleared for a resend")
        return
    if kind not in ("sent", "release", "settle", "unknown"):
        raise LedgerError(f"line {line}: unknown event {kind!r}")
    group = event.get("group")
    if group not in state.groups:
        raise LedgerError(f"line {line}: {kind} for unknown group {group!r}")
    members = [i for (g, _), i in state.items.items() if g == group]
    if kind == "sent":
        if state.groups[group]["sent"] is not None:
            raise LedgerError(f"line {line}: group {group} was already sent")
        if any(i.status != "reserved" for i in members):
            raise LedgerError(f"line {line}: group {group} has items that are no longer just reserved")
    elif kind == "release":
        if state.groups[group]["sent"] is not None or any(i.status != "reserved" for i in members):
            raise LedgerError(f"line {line}: group {group} was sent, so it can only be settled or marked unknown")
    else:
        item = state.items.get((group, event.get("custom_id")))
        if item is None:
            raise LedgerError(f"line {line}: {kind} for {event.get('custom_id')}, which group {group} never reserved")
        if item.status != "sent":
            raise LedgerError(f"line {line}: {item.custom_id} is {item.status}; only a sent request can be {kind}")
        if kind == "settle" and event.get("outcome") not in OUTCOMES:
            raise LedgerError(f"line {line}: unknown outcome {event.get('outcome')!r}")


def _apply(state: State, event: Mapping, line: int) -> None:
    _check(state, event, line)
    kind = event["event"]
    if kind == "acknowledge":
        if "phase" in event:
            state.accepted.add((event["phase"], event["kind"]))
        else:
            state.acknowledged.add((event["group"], event["custom_id"]))
        return
    group = event["group"]
    if kind == "resend":
        state.cleared.add((group, event["custom_id"]))  # still counted at its bound: it may have been billed
        return
    if kind == "reserve":
        state.groups[group] = {"phase": event["phase"], "mode": event["mode"], "sent": None}
        for it in event["items"]:
            state.items[(group, it["custom_id"])] = Item(group, it["custom_id"], it["model"], event["phase"],
                                                        event["mode"], Decimal(it["bound"]))
        return
    members = [i for (g, _), i in state.items.items() if g == group]
    if kind == "sent":
        state.groups[group]["sent"] = {k: event[k] for k in ("batch_id", "request_id") if k in event}
        for i in members:
            i.status = "sent"
    elif kind == "release":
        for i in members:
            i.status = "released"
    else:
        item = state.items[(group, event["custom_id"])]
        if kind == "unknown":
            item.status = "unknown"
        else:
            item.status, item.cost, item.outcome = "settled", Decimal(event["cost"]), event["outcome"]
            item.flags = list(event.get("flags", ()))
            item.detail = event.get("detail")


def read_events(path: Path) -> tuple[list[dict], bool]:
    """Every complete event, and whether a torn tail follows them.

    An event exists only once its closing newline is on disk, so bytes after
    the last newline (a crash mid-write) are a torn tail and never an event.
    A line before the last newline that isn't valid JSON is corruption: refused.
    """
    if not path.exists():
        return [], False
    raw = path.read_bytes()
    complete, _, tail = raw.rpartition(b"\n")
    events = []
    for n, chunk in enumerate(complete.split(b"\n") if complete else [], start=1):
        if not chunk.strip():
            continue
        try:
            events.append(json.loads(chunk))
        except json.JSONDecodeError:
            raise LedgerError(f"{path}: line {n} is not valid JSON") from None
    return events, bool(tail)


def _lines(path: Path) -> list[bytes] | None:
    """A copy's complete lines (bytes), or None if the copy doesn't exist."""
    if not path.exists():
        return None
    complete, _, _ = path.read_bytes().rpartition(b"\n")
    return [line for line in complete.split(b"\n") if line.strip()] if complete else []


def _truncate_torn_tail(path: Path) -> None:
    raw = path.read_bytes()
    keep = raw[: raw.rfind(b"\n") + 1]
    with open(path, "r+b") as f:
        f.truncate(len(keep))
        f.flush()
        os.fsync(f.fileno())


def default_mirror(stage: str = "stage4") -> Path:
    """Outside the repository and outside iCloud Drive: losing data/ can't take the mirror with it."""
    override = os.environ.get("RAGLAB_LEDGER_MIRROR")
    if override:
        return Path(override).expanduser()
    return Path.home() / "Library" / "Application Support" / "raglab" / stage / "ledger.jsonl"


def fold(events: Iterable[Mapping]) -> State:
    state = State()
    for n, event in enumerate(events, start=1):
        _apply(state, event, n)
    return state


# ---------------------------------------------------------------------------------------------------- the ledger

@dataclass(frozen=True)
class Planned:
    """One request about to be reserved."""

    custom_id: str
    model: str
    input_tokens: int  # already including the margin (input_bound)
    max_tokens: int
    cache_marked: bool


class Ledger:
    """The append-only spend ledger. Use as a context manager: it holds an exclusive lock while open."""

    def __init__(self, path: Path, pricing: Pricing, caps: Caps, mirror: Path | None = None,
                 repair: bool = False, expect_history: bool = False) -> None:
        self.path = path
        self.mirror = mirror
        self.expect_history = expect_history  # stored responses exist, so an empty ledger would mean lost spend
        self.pricing = pricing
        self.caps = caps
        self.repair = repair
        self.repairs: list[str] = []
        self._lock_file = None
        self._events: list[dict] = []
        self.state = State()

    def __enter__(self) -> "Ledger":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock_file = open(self.path.with_suffix(".lock"), "a+")
        try:
            fcntl.flock(self._lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self._lock_file.close()
            self._lock_file = None
            raise LedgerError(f"another process holds {self.path}; one run at a time") from None
        try:
            self._events = self._open_copies()
            self.state = fold(self._events)
        except BaseException:
            self.__exit__()
            raise
        return self

    def _open_copies(self) -> list[dict]:
        """Read the ledger and its mirror, heal the one safe difference, refuse every other."""
        copies = [self.path] + ([self.mirror] if self.mirror else [])
        for copy in copies:
            if copy.exists():
                events, torn = read_events(copy)
                if torn:
                    if not self.repair:
                        raise LedgerError(f"{copy} ends in a torn line (a crash mid-write). Run `budget repair` to "
                                          "drop it; the event it held never took effect.")
                    _truncate_torn_tail(copy)
                    self.repairs.append(f"dropped a torn last line from {copy}")
        primary = _lines(self.path)
        if self.mirror is None:
            if primary is None and self.expect_history:
                raise LedgerError(f"no ledger at {self.path}, but responses are stored: the spend record is missing")
            return read_events(self.path)[0]
        mirror = _lines(self.mirror)
        if primary is None and mirror is None:
            if self.expect_history:
                raise LedgerError(f"no ledger at {self.path} or its mirror, but responses are stored: the spend record "
                                  "is missing. Restore a copy; an empty ledger would reset spend to $0.")
            return []
        if primary is None or mirror is None:
            missing, present = (self.path, self.mirror) if primary is None else (self.mirror, self.path)
            if not self.repair:
                raise LedgerError(f"{missing} is missing but {present} exists. Run `budget repair` to restore it "
                                  "from the copy that exists, after checking why it went missing.")
            missing.parent.mkdir(parents=True, exist_ok=True)
            missing.write_bytes(present.read_bytes())
            self.repairs.append(f"restored {missing} from {present}")
            return read_events(self.path)[0]
        if primary == mirror:
            return read_events(self.path)[0]
        if mirror == primary[: len(mirror)] and len(primary) - len(mirror) == 1:
            with open(self.mirror, "ab") as f:  # the crash fell between the two writes of one event
                f.write(primary[-1] + b"\n")
                f.flush()
                os.fsync(f.fileno())
            self.repairs.append(f"copied the last event to {self.mirror}")
            return read_events(self.path)[0]
        raise LedgerError(f"{self.path} and its mirror {self.mirror} disagree beyond one trailing event; "
                          "compare them by hand before spending anything")

    def __exit__(self, *exc) -> None:
        if self._lock_file is not None:
            fcntl.flock(self._lock_file, fcntl.LOCK_UN)
            self._lock_file.close()
            self._lock_file = None


    def _append(self, event: dict) -> None:
        if self._lock_file is None:
            raise LedgerError("open the ledger with `with Ledger(...) as ledger:` before writing")
        _check(self.state, event, len(self._events) + 1)  # refused events never reach the file
        line = json.dumps(event, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
        for copy in [self.path] + ([self.mirror] if self.mirror else []):  # the ledger first, then its mirror
            copy.parent.mkdir(parents=True, exist_ok=True)
            with open(copy, "a", encoding="utf-8") as f:
                f.write(line)
                f.flush()
                os.fsync(f.fileno())
        self._events.append(event)
        _apply(self.state, event, len(self._events))

    # -- spending

    def headroom(self, phase: str) -> Decimal:
        """How much more can be reserved in this phase right now."""
        return min(self.caps.total - self.state.committed(), self.caps.phase(phase) - self.state.committed(phase))

    def reserve(self, phase: str, mode: str, planned: Sequence[Planned], authorization: str) -> str:
        """Reserve a request (standard) or a whole batch; returns the group id. Raises CapExceeded.

        `authorization` is the SHA-256 of the authorization record the caller has just checked; a
        reservation without one is refused.
        """
        if not authorization:
            raise LedgerError("a reservation must name the authorization it was made under")
        if not planned:
            raise LedgerError("nothing to reserve")
        flagged = self.state.unacknowledged()
        if flagged:
            i = flagged[0]
            raise LedgerError(f"{len(flagged)} settlement(s) carry flags nobody has accepted, e.g. {i.custom_id} in "
                              f"{i.group}: {'; '.join(i.flags)}. Read them with `budget status`, then run `budget "
                              "acknowledge` yourself before anything else is sent")
        if mode == "standard" and len(planned) != 1:
            raise LedgerError("a standard reservation holds exactly one request")
        self.caps.phase(phase)
        ids = [p.custom_id for p in planned]
        if len(set(ids)) != len(ids):
            raise LedgerError("a reservation can't hold the same custom_id twice")
        bounds = [self.pricing.bound(p.model, p.input_tokens, p.max_tokens, p.cache_marked, mode) for p in planned]
        total = sum(bounds, Decimal(0))
        if total > self.headroom(phase):
            raise CapExceeded(
                f"reserving ${money(total)} for {len(planned)} request(s) in phase {phase!r} could exceed a cap: "
                f"committed ${money(self.state.committed())} of ${money(self.caps.total)} in all, "
                f"${money(self.state.committed(phase))} of ${money(self.caps.phase(phase))} in this phase")
        group = f"{phase}-{sum(1 for e in self._events if e['event'] == 'reserve') + 1:05d}"
        self._append({"event": "reserve", "group": group, "phase": phase, "mode": mode, "at": _now(),
                      "authorization": authorization,
                      "items": [{"custom_id": p.custom_id, "model": p.model, "bound": money(b),
                                 "input_tokens": p.input_tokens, "max_tokens": p.max_tokens}
                                for p, b in zip(planned, bounds)]})
        return group

    def mark_sent(self, group: str, *, batch_id: str | None = None, request_id: str | None = None) -> None:
        event = {"event": "sent", "group": group, "at": _now()}
        if batch_id:
            event["batch_id"] = batch_id
        if request_id:
            event["request_id"] = request_id
        self._append(event)

    def settle(self, group: str, custom_id: str, outcome: str, model: str | None, usage: Mapping | None,
               detail: str | None = None) -> Decimal:
        """Record what one request cost. Unbilled outcomes (errored, canceled, expired) settle at $0. `detail`
        names an errored standard call's error kind, so a transient one the runner retried is told apart."""
        if outcome not in OUTCOMES:
            raise LedgerError(f"outcome must be one of {OUTCOMES}, not {outcome!r}")
        item = self.state.items.get((group, custom_id))
        if item is None:
            raise LedgerError(f"{custom_id} was never reserved in group {group}")
        if item.status != "sent":
            raise LedgerError(f"{custom_id} is {item.status}; only a sent request can be settled")
        flags: list[str] = []
        if outcome == "succeeded":
            if usage is None:
                raise LedgerError(f"{custom_id} succeeded without usage")
            served = model or item.model
            if served in self.pricing.models:
                cost, flags = self.pricing.usage_cost(served, usage, item.mode)
            else:  # no price to settle at: count it at its bound, the most it could have cost, and stop
                cost, flags = item.bound, [f"served by {served}, which pricing.json can't price: counted at its bound"]
            if served != item.model:
                flags.append(f"reserved for {item.model}, served by {served}")
            if cost > item.bound:
                flags.append(f"cost ${money(cost)} exceeded its bound ${money(item.bound)}")
        else:
            cost = Decimal(0)
        event = {"event": "settle", "group": group, "custom_id": custom_id, "outcome": outcome, "model": model,
                 "usage": dict(usage or {}), "cost": money(cost), "flags": flags, "at": _now()}
        if detail:
            event["detail"] = detail
        self._append(event)
        return cost

    def resend(self, group: str, custom_id: str, reason: str) -> None:
        """Let an unknown attempt be sent again. It stays counted at its bound, since it may have been billed."""
        self._append({"event": "resend", "group": group, "custom_id": custom_id, "reason": reason, "at": _now()})

    def mark_unknown(self, group: str, custom_id: str, reason: str) -> None:
        self._append({"event": "unknown", "group": group, "custom_id": custom_id, "reason": reason, "at": _now()})

    def acknowledge(self, group: str, custom_id: str, reason: str) -> None:
        """A person has read a flagged settlement and accepts it: it no longer stops new reservations."""
        self._append({"event": "acknowledge", "group": group, "custom_id": custom_id, "reason": reason, "at": _now()})

    def accept_for_phase(self, phase: str, kind: str, reason: str) -> None:
        """Accept one kind of flag (US-only inference) for every settlement of a phase, past and future."""
        self._append({"event": "acknowledge", "phase": phase, "kind": kind, "reason": reason, "at": _now()})

    def release(self, group: str, reason: str) -> None:
        """Free a reservation that was never sent. Refused once anything in the group was sent."""
        self._append({"event": "release", "group": group, "reason": reason, "at": _now()})

    # -- reading

    def summary(self) -> dict:
        s = self.state
        phases = sorted({i.phase for i in s.items.values()} | set(self.caps.phases))
        return {
            "pricing_as_of": self.pricing.as_of,
            "cap": money(self.caps.total),
            "committed": money(s.committed()),
            "settled": money(s.settled()),
            "open_reservations": len(s.open_items()),
            "unknown": len(s.unknown()),
            "unacknowledged_flags": len(s.unacknowledged()),
            "phases": {p: {"cap": money(self.caps.phases[p]) if p in self.caps.phases else None,
                           "committed": money(s.committed(p)), "settled": money(s.settled(p))} for p in phases},
            "flags": [f"{i.custom_id}: {f}" for i in s.items.values() for f in i.flags],
        }

    def events(self) -> Iterator[dict]:
        yield from self._events
