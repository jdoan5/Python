"""Every billed request and its response, kept under the request's hash.

A request's key is the SHA-256 of its canonical JSON: the condition (arm) it
belongs to, the endpoint, the anthropic-version header and the body, so the
same request always has the same key and any change to a prompt, a document or
a setting makes a new one. The body's own hash is kept beside it.
Sampling can't be fixed on these models (temperature and its relatives are
rejected), so asking the same request again is a deliberate new *sample*: it
gets its own `custom_id`, ``r<key[:56]>s<sample:02d>`` (60 characters; the
Batch API allows 64).

Before anything is sent, the store is asked: a request whose custom_id is
already stored is never sent again. Records are appended, one fsynced JSON line
each, and a custom_id stored twice with different responses is refused, which
is what keeps results reproducible from the stored responses alone.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping

from raglab.ledger import LedgerError, read_events

API_VERSION = "2026-10-21"  # what the pinned SDK (anthropic 2.x) sends
MESSAGES = "/v1/messages"
_CUSTOM_ID = re.compile(r"^r([0-9a-f]{56})s(\d{2})$")


class StoreError(RuntimeError):
    pass


def canonical(obj: object) -> bytes:
    """One byte string per value: sorted keys, no spaces, UTF-8, no NaN."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def body_sha256(body: Mapping) -> str:
    """The request body's own hash: equal for identical bodies in different conditions."""
    return hashlib.sha256(canonical(body)).hexdigest()


def request_key(condition: str, body: Mapping, endpoint: str = MESSAGES, api_version: str = API_VERSION) -> str:
    """The key of one request *in one condition*.

    Two conditions can send byte-identical bodies (a question whose 2.4k window
    equals its oracle window; two identical abstentions sent to the judge), and
    each must stay its own request, so the condition is part of the key.
    """
    if not condition or not isinstance(condition, str):
        raise StoreError("every request belongs to a named condition")
    return hashlib.sha256(canonical({"condition": condition, "endpoint": endpoint, "anthropic_version": api_version,
                                     "body": body})).hexdigest()


def custom_id(key: str, sample: int = 0) -> str:
    if not re.fullmatch(r"[0-9a-f]{64}", key):
        raise StoreError(f"not a request key: {key!r}")
    if not 0 <= sample <= 99:
        raise StoreError(f"sample must be 0-99, not {sample}")
    return f"r{key[:56]}s{sample:02d}"


def parse_custom_id(cid: str) -> tuple[str, int]:
    """The key prefix and sample a custom_id names."""
    m = _CUSTOM_ID.match(cid)
    if not m:
        raise StoreError(f"not a Stage 4 custom_id: {cid!r}")
    return m.group(1), int(m.group(2))


@dataclass(frozen=True)
class Record:
    custom_id: str
    request_key: str
    condition: str
    body_sha256: str
    phase: str
    outcome: str          # succeeded | errored | canceled | expired | refused_locally
    response: Mapping | None
    error: Mapping | None
    batch_id: str | None
    request_id: str | None
    received_at: str
    group: str | None = None  # the ledger group of the attempt that produced it
    timing: Mapping | None = None  # a streamed request's seconds to first text and in all; never scored

    def to_json(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


class Store:
    """An append-only JSONL file of records for one phase (pilot, generation, judging, ...)."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._records: dict[str, Record] = {}
        events, torn = read_events(path) if path.exists() else ([], False)
        if torn:
            raise StoreError(f"{path} ends in a torn line; inspect it before going on (no record was lost: "
                             "the request is still open in the ledger)")
        for raw in events:
            rec = Record(**raw)
            if self._is_new(rec):
                self._records[rec.custom_id] = rec

    def _is_new(self, rec: Record) -> bool:
        """True for a record not stored yet; False for an exact repeat; raises on a conflict. Changes nothing."""
        prefix, _ = parse_custom_id(rec.custom_id)
        if not rec.request_key.startswith(prefix):
            raise StoreError(f"{rec.custom_id} doesn't match its request key")
        old = self._records.get(rec.custom_id)
        if old is None:
            return True
        if canonical(old.to_json() | {"received_at": None}) != canonical(rec.to_json() | {"received_at": None}):
            raise StoreError(f"{rec.custom_id} is already stored with a different response")
        return False

    def __contains__(self, cid: object) -> bool:
        return cid in self._records

    def __len__(self) -> int:
        return len(self._records)

    def get(self, cid: str) -> Record | None:
        return self._records.get(cid)

    def __iter__(self) -> Iterator[Record]:
        return iter(sorted(self._records.values(), key=lambda r: r.custom_id))

    def add(self, rec: Record) -> bool:
        """Append a record; False if this exact record was already there. Refuses a conflicting one."""
        if not self._is_new(rec):
            return False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(canonical(rec.to_json()).decode("utf-8") + "\n")
            f.flush()
            os.fsync(f.fileno())
        self._records[rec.custom_id] = rec  # only once it is on disk
        return True

    @property
    def duplicates_path(self) -> Path:
        return self.path.with_name(self.path.stem + ".duplicates.jsonl")

    def add_duplicate(self, rec: Record) -> None:
        """A second, different response to a request already stored: a duplicate attempt that was paid for.

        It is kept beside the store, never in it, so results always come from
        the first response stored.
        """
        if rec.custom_id not in self._records:
            raise StoreError(f"{rec.custom_id} isn't stored, so it can't have a duplicate")
        with open(self.duplicates_path, "a", encoding="utf-8") as f:
            f.write(canonical(rec.to_json()).decode("utf-8") + "\n")
            f.flush()
            os.fsync(f.fileno())


__all__ = ["API_VERSION", "MESSAGES", "Record", "Store", "StoreError", "LedgerError", "body_sha256", "canonical",
           "custom_id", "parse_custom_id", "request_key"]
