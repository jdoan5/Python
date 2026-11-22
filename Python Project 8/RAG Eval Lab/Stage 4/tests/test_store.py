"""The response store: keys, custom_ids, append-only records. No network, no key."""

from __future__ import annotations

from pathlib import Path

import pytest

from raglab.store import Record, Store, StoreError, body_sha256, canonical, custom_id, parse_custom_id, request_key

BODY = {"model": "claude-haiku-5-5", "max_tokens": 4000, "messages": [{"role": "user", "content": "Hi"}]}


def record(cid: str, key: str, text: str = "Hello", outcome: str = "succeeded") -> Record:
    return Record(custom_id=cid, request_key=key, condition="hhr-4.8k-med", body_sha256=body_sha256(BODY),
                  phase="pilot", outcome=outcome,
                  response={"content": [{"type": "text", "text": text}]}, error=None, batch_id="msgbatch_1",
                  request_id=None, received_at="2026-11-21T00:00:00+00:00")


def test_the_key_depends_on_every_byte_of_the_request_but_not_on_key_order() -> None:
    k = request_key("hhr-4.8k-med", BODY)
    assert k == request_key("hhr-4.8k-med", dict(reversed(list(BODY.items()))))
    assert k != request_key("hhr-4.8k-med", BODY | {"max_tokens": 4001})
    assert k != request_key("hhr-4.8k-med", BODY, api_version="2023-06-01")
    assert canonical({"b": 1, "a": "é"}) == '{"a":"é","b":1}'.encode("utf-8")


def test_custom_ids_fit_the_batch_limit_and_round_trip() -> None:
    k = request_key("hhr-4.8k-med", BODY)
    cid = custom_id(k, 3)
    assert len(cid) == 60 <= 64 and cid.startswith("r") and cid.endswith("s03")
    assert parse_custom_id(cid) == (k[:56], 3)
    with pytest.raises(StoreError):
        custom_id(k, 100)
    with pytest.raises(StoreError):
        parse_custom_id("request-1")


def test_records_are_appended_and_reloaded(tmp_path: Path) -> None:
    k = request_key("hhr-4.8k-med", BODY)
    store = Store(tmp_path / "pilot.jsonl")
    assert store.add(record(custom_id(k, 0), k))
    assert not store.add(record(custom_id(k, 0), k))  # an exact repeat is a no-op
    assert store.add(record(custom_id(k, 1), k, text="Hi there"))  # another sample of the same request
    again = Store(tmp_path / "pilot.jsonl")
    assert len(again) == 2 and custom_id(k, 1) in again
    assert again.get(custom_id(k, 1)).response["content"][0]["text"] == "Hi there"


def test_a_conflicting_response_is_refused(tmp_path: Path) -> None:
    k = request_key("hhr-4.8k-med", BODY)
    store = Store(tmp_path / "pilot.jsonl")
    store.add(record(custom_id(k, 0), k))
    with pytest.raises(StoreError, match="different response"):
        store.add(record(custom_id(k, 0), k, text="Something else"))
    assert len((tmp_path / "pilot.jsonl").read_text(encoding="utf-8").splitlines()) == 1


def test_a_record_must_match_its_key(tmp_path: Path) -> None:
    k = request_key("hhr-4.8k-med", BODY)
    other = request_key("hhr-4.8k-med", BODY | {"max_tokens": 1})
    with pytest.raises(StoreError, match="doesn't match"):
        Store(tmp_path / "x.jsonl").add(record(custom_id(k, 0), other))


def test_a_torn_store_is_refused(tmp_path: Path) -> None:
    k = request_key("hhr-4.8k-med", BODY)
    path = tmp_path / "pilot.jsonl"
    Store(path).add(record(custom_id(k, 0), k))
    with path.open("a", encoding="utf-8") as f:
        f.write('{"custom_id": "r')
    with pytest.raises(StoreError, match="torn"):
        Store(path)


def test_the_same_body_in_two_conditions_is_two_requests() -> None:
    a, b = request_key("hhr-2.4k-med", BODY), request_key("oracle-4.8k-med", BODY)
    assert a != b and custom_id(a, 0) != custom_id(b, 0)
    with pytest.raises(StoreError):
        request_key("", BODY)
