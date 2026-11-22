"""Sending billed requests: check, reserve, send, store, settle, always in that order.

Before each reservation the gate runs again: the authorization must still hold
(PLAN.md, "Phases"), and the request must be on the authorized request list (or
be one of the retries listed with it). Then:
- a crash before sending leaves an unsent reservation, which `reconcile` releases;
- a crash after sending but before the response is stored leaves the request
  counted at its bound (the money may be spent, so the cap still sees it);
- a crash after storing but before settling leaves a stored response whose
  reservation is still open: `settle_from_store` settles it from the stored usage;
- a response already in the store is never asked for again, and the ledger
  refuses a second attempt while an earlier one is open or unknown, or once
  one has succeeded (even if the store lost its response).

Standard requests retry rate limits and server errors (never billed) a fixed
number of times. A dropped connection stops the run: the request may have been
billed, so sending it again could pay twice, and only an explicit resend in the
ledger lets it go out again. Creating a batch is never retried for the same
reason, and no batch is created while an earlier create is unresolved. Any
pricing flag on a settlement (a cost above its bound, another model, US-only
inference, a priority tier) stops the run once the batch is booked.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable, Iterable, Mapping, Sequence

from raglab import params
from raglab.ledger import Ledger, LedgerError, Planned
from raglab.llm import REQUEST_ID, ApiError, Transport
from raglab.store import Record, Store, StoreError, body_sha256, custom_id as make_custom_id, request_key

MAX_SERVER_RETRIES = 5
MAX_RATE_LIMIT_WAITS = 10
BACKOFF_SECONDS = (2, 4, 8, 16, 32)


class RunStopped(RuntimeError):
    """Something needs a person before more money is spent."""


def _has_cache_control(obj) -> bool:
    if isinstance(obj, Mapping):
        return "cache_control" in obj or any(_has_cache_control(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return any(_has_cache_control(v) for v in obj)
    return False


@dataclass(frozen=True)
class Request:
    """One billed request: its condition, its body, which sample it is, and its counted input (with the margin)."""

    condition: str
    body: Mapping
    sample: int
    input_tokens: int

    @property
    def key(self) -> str:
        return request_key(self.condition, self.body)

    @property
    def custom_id(self) -> str:
        return make_custom_id(self.key, self.sample)

    @property
    def model(self) -> str:
        return self.body["model"]

    @property
    def max_tokens(self) -> int:
        return int(self.body["max_tokens"])

    @property
    def cache_marked(self) -> bool:
        """Read from the body, never declared: a cache marker anywhere makes every input token a possible write."""
        return _has_cache_control(self.body)

    def planned(self) -> Planned:
        return Planned(self.custom_id, self.model, self.input_tokens, self.max_tokens, self.cache_marked)


@dataclass(frozen=True)
class Gate:
    """What may be sent. `authorize` re-checks the authorization and returns its SHA-256 (or raises);
    `allowed_ids` holds the custom_ids on the authorized request list, its listed retries included, so an
    unlisted sample of a listed request is refused too."""

    authorize: Callable[[], str]
    allowed_ids: frozenset[str] | None

    def admit(self, reqs: Sequence[Request]) -> str:
        if self.allowed_ids is not None:
            stray = [r.custom_id for r in reqs if r.custom_id not in self.allowed_ids]
            if stray:
                raise RunStopped(f"{len(stray)} request(s) aren't on the authorized request list, e.g. {stray[0]}")
        return self.authorize()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _record(req: Request, phase: str, message: Mapping, batch_id: str | None, group: str,
            timing: Mapping | None = None) -> Record:
    response = {k: v for k, v in message.items() if k != REQUEST_ID}
    return Record(custom_id=req.custom_id, request_key=req.key, condition=req.condition,
                  body_sha256=body_sha256(req.body), phase=phase, outcome="succeeded", response=response, error=None,
                  batch_id=batch_id, request_id=message.get(REQUEST_ID), received_at=_now(), group=group,
                  timing=dict(timing) if timing is not None else None)


def _store(store: Store, record: Record) -> bool:
    """Store a response. A second, different response to the same request (a duplicate attempt that was
    paid for) goes to the store's side file instead, and is still settled."""
    try:
        store.add(record)
        return True
    except StoreError:
        store.add_duplicate(record)
        return False


def _flags(ledger: Ledger, group: str, custom_id: str) -> list[str]:
    """The settlement's flags that stop the run: one of a kind accepted for the phase (US-only inference after
    `budget acknowledge --us-inference`) is booked and reported, but doesn't stop anything."""
    return ledger.state.stopping_flags(ledger.state.items[(group, custom_id)])


def send_standard(ledger: Ledger, store: Store, transport: Transport, phase: str, req: Request, gate: Gate,
                  sleep: Callable[[float], None] = time.sleep) -> Record:
    """One request through the standard API. Returns its stored record."""
    return _send_one(ledger, store, phase, req, gate, lambda body: (transport.create(body), None), sleep)


def send_stream(ledger: Ledger, store: Store, transport: Transport, phase: str, req: Request, gate: Gate,
                sleep: Callable[[float], None] = time.sleep) -> Record:
    """One request through the standard API, streamed and timed (the timing probe and run). Priced as standard.

    The record keeps seconds to the first text and seconds in all. Once a
    stream has opened, any failure counts as a dropped connection: it may have
    been billed, so it stops the run like one.
    """
    def call(body: Mapping):
        out = transport.stream(body)
        return out["message"], {"first_text_seconds": out["first_text_seconds"],
                                "total_seconds": out["total_seconds"]}

    return _send_one(ledger, store, phase, req, gate, call, sleep)


def _send_one(ledger: Ledger, store: Store, phase: str, req: Request, gate: Gate,
              call: Callable[[Mapping], tuple[Mapping, Mapping | None]], sleep: Callable[[float], None]) -> Record:
    stored = store.get(req.custom_id)
    if stored is not None:
        return stored
    server_errors = rate_limits = 0
    while True:
        group = ledger.reserve(phase, "standard", [req.planned()], gate.admit([req]))
        try:
            ledger.mark_sent(group)
        except BaseException:  # never sent, so never billed: free the reservation rather than strand it
            ledger.release(group, "interrupted before it was sent")
            raise
        try:
            message, timing = call(req.body)
        except ApiError as e:
            if e.billed_possibly:
                ledger.mark_unknown(group, req.custom_id, str(e))
                raise RunStopped(f"{req.custom_id}: no response ({e}). It may have been billed, so it was not "
                                 "sent again; it stays counted at its bound until a person resends it.") from None
            ledger.settle(group, req.custom_id, "errored", None, None, detail=e.kind)
            if e.kind == "rate_limited" and rate_limits < MAX_RATE_LIMIT_WAITS:
                rate_limits += 1
                sleep(e.retry_after if e.retry_after is not None else 60.0)
                continue
            if e.kind == "server" and server_errors < MAX_SERVER_RETRIES:
                sleep(BACKOFF_SECONDS[server_errors])
                server_errors += 1
                continue
            raise RunStopped(f"{req.custom_id}: {e}") from None
        except BaseException as e:  # Ctrl-C, a killed process, an SDK bug: it may have been billed
            ledger.mark_unknown(group, req.custom_id, f"interrupted while sending ({type(e).__name__})")
            raise
        record = _record(req, phase, message, None, group, timing)
        _store(store, record)  # stored first: a crash now costs a settlement, not a second request
        ledger.settle(group, req.custom_id, "succeeded", message.get("model"), message.get("usage"))
        flags = _flags(ledger, group, req.custom_id)
        if flags:
            raise RunStopped(f"{req.custom_id}: " + "; ".join(flags))
        return store.get(req.custom_id) or record


def submit_batch(ledger: Ledger, store: Store, transport: Transport, phase: str, reqs: Sequence[Request],
                 gate: Gate) -> str | None:
    """Reserve and create one batch of the requests not stored and not already in flight. Returns its group."""
    pending = unsent_batches(ledger)
    if pending:
        raise RunStopped(f"batch create(s) {', '.join(pending)} got no answer; run `batch reconcile` first")
    todo = [r for r in reqs if r.custom_id not in store and not ledger.state.blocking(r.custom_id)]
    ids = [r.custom_id for r in todo]
    if len(set(ids)) != len(ids):
        raise LedgerError("a batch can't hold the same custom_id twice")
    if not todo:
        return None
    models = {r.model for r in todo}
    if len(models) != 1:
        raise LedgerError(f"a batch holds one model's requests, not {sorted(models)}")
    limit = params.BATCH_MAX[models.pop()]
    if len(todo) > limit:
        raise LedgerError(f"{len(todo)} requests is more than this model's batch limit of {limit}")
    group = ledger.reserve(phase, "batch", [r.planned() for r in todo], gate.admit(todo))
    try:
        batch = transport.batch_create([{"custom_id": r.custom_id, "params": dict(r.body)} for r in todo])
    except ApiError as e:
        if e.billed_possibly:
            raise RunStopped(f"batch create for {group} got no response ({e}); it may exist. Run `batch reconcile` "
                             "before submitting again.") from None
        ledger.release(group, f"batch create refused: {e}")
        raise RunStopped(f"batch create for {group} was refused: {e}") from None
    ledger.mark_sent(group, batch_id=batch["id"])
    return group


def collect_batch(ledger: Ledger, store: Store, transport: Transport, group: str,
                  requests: Mapping[str, Request]) -> dict:
    """Store and settle an ended batch's results. Safe to run again: settled items are skipped."""
    sent = ledger.state.groups[group]["sent"]
    if not sent or "batch_id" not in sent:
        raise LedgerError(f"group {group} has no batch")
    batch_id = sent["batch_id"]
    status = transport.batch_retrieve(batch_id)["processing_status"]
    counts = {"status": status, "succeeded": 0, "errored": 0, "canceled": 0, "expired": 0, "already": 0,
              "duplicates": 0}
    if status != "ended":
        return counts
    phase = ledger.state.groups[group]["phase"]
    seen, flagged = set(), []
    for item in transport.batch_results(batch_id):
        cid, result = item["custom_id"], item["result"]
        entry = ledger.state.items.get((group, cid))
        if entry is None or cid not in requests:
            raise RunStopped(f"batch {batch_id} returned {cid}, which group {group} never reserved")
        seen.add(cid)
        if entry.status != "sent":
            counts["already"] += 1
            continue
        kind = result["type"]
        if kind == "succeeded":
            message = result["message"]
            if not _store(store, _record(requests[cid], phase, message, batch_id, group)):
                counts["duplicates"] += 1
            ledger.settle(group, cid, "succeeded", message.get("model"), message.get("usage"))
            flagged += [f"{cid}: {f}" for f in _flags(ledger, group, cid)]
        else:
            ledger.settle(group, cid, kind, None, None)
        counts[kind] += 1
    missing = [i.custom_id for (g, _), i in ledger.state.items.items() if g == group and i.status == "sent"
               and i.custom_id not in seen]
    if missing:
        raise RunStopped(f"batch {batch_id} ended without results for {len(missing)} request(s), e.g. {missing[0]}")
    if flagged:
        raise RunStopped(f"batch {batch_id} is booked, but its settlements were flagged: " + "; ".join(flagged[:5]))
    return counts


def settle_from_store(ledger: Ledger, store: Store) -> int:
    """Settle reservations whose responses were stored before a crash stopped the settlement.

    A stored record names the ledger group of the attempt that produced it, so a
    response is only ever matched to its own attempt.
    """
    n = 0
    for (group, cid), item in list(ledger.state.items.items()):
        record = store.get(cid)
        if item.status == "sent" and record is not None and record.outcome == "succeeded" and record.group == group:
            ledger.settle(group, cid, "succeeded", record.response.get("model"), record.response.get("usage"))
            n += 1
    return n


def unsent_batches(ledger: Ledger) -> list[str]:
    """Batch reservations with no batch id: a create that crashed or got no response."""
    return [g for g, info in ledger.state.groups.items()
            if info["mode"] == "batch" and info["sent"] is None
            and any(i.status == "reserved" for (gg, _), i in ledger.state.items.items() if gg == g)]


def attached_batch_ids(ledger: Ledger) -> set[str]:
    return {info["sent"]["batch_id"] for info in ledger.state.groups.values()
            if info["sent"] and "batch_id" in info["sent"]}


def reconcile(ledger: Ledger, transport: Transport, release: bool = False) -> dict:
    """Match unsent batch reservations to batches the API holds, by their custom_ids.

    Every batch created since the oldest pending reservation is examined (the
    listing is paged back that far). A batch that ended and returns exactly a
    pending group's custom_ids is attached to it. With release=True, and only
    when no batch in that period is still running or unmatched, a reservation
    that matches no batch is released: it was never created, so never billed.
    """
    pending = unsent_batches(ledger)
    report = {"pending": pending, "attached": {}, "released": [], "unmatched_batches": []}
    if not pending:
        return report
    known = attached_batch_ids(ledger)
    since = min(_reserved_at(ledger, g) for g in pending) - timedelta(minutes=5)  # clock skew
    for batch in transport.batch_list():  # newest first, paged
        created = _parse_time(batch.get("created_at"))
        if created < since:
            break  # older than every pending reservation: someone else's work, never opened
        if batch["id"] in known:
            continue
        if batch.get("processing_status") != "ended":
            report["unmatched_batches"].append(batch["id"])
            continue
        ids = {item["custom_id"] for item in transport.batch_results(batch["id"])}
        for g in pending:
            members = {cid for (gg, cid) in ledger.state.items if gg == g}
            if ids == members and g not in report["attached"]:
                ledger.mark_sent(g, batch_id=batch["id"])
                report["attached"][g] = batch["id"]
                break
        else:
            report["unmatched_batches"].append(batch["id"])
    if release and not report["unmatched_batches"]:
        for g in pending:
            if g not in report["attached"]:
                ledger.release(g, "reconcile: the API holds no batch with these requests")
                report["released"].append(g)
    return report


def _parse_time(value) -> datetime:
    if not value:
        return datetime.min.replace(tzinfo=timezone.utc)
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def _reserved_at(ledger: Ledger, group: str) -> datetime:
    for event in ledger.events():
        if event["event"] == "reserve" and event["group"] == group:
            return _parse_time(event["at"])
    raise LedgerError(f"no reservation for {group}")


def requests_by_id(reqs: Iterable[Request]) -> dict[str, Request]:
    out: dict[str, Request] = {}
    for r in reqs:
        if r.custom_id in out:
            raise StoreError(f"two planned requests share {r.custom_id}")
        out[r.custom_id] = r
    return out
