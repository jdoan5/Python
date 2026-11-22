"""When a judge fails: retries, truncation re-runs, resubmissions and unjudged answers. No API."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from raglab.judge import InvalidVerdict
from raglab.judging import JudgeLine
from raglab.llm import fake_message
from raglab.plan4 import line_for
from raglab.store import Record, Store
from raglab.verdicts import failures_from, groups, resolve

BODY = {"model": "claude-opus-5-5", "max_tokens": 8000, "messages": [{"role": "user", "content": "x"}]}
LONG = {**BODY, "max_tokens": 16000}


def lines(answered="r" + "a" * 56 + "s00") -> list[JudgeLine]:
    cond = f"F:{answered}"
    return [JudgeLine(line_for(cond, "p01", 0, BODY), "F", answered, "primary", "pilot-medium"),
            JudgeLine(line_for(cond, "p01", 1, BODY), "F", answered, "retry", "pilot-medium"),
            JudgeLine(line_for(cond, "p01", 0, LONG), "F", answered, "truncation", "pilot-medium")]


def put(store: Store, jl: JudgeLine, text: str = '{"ok": true}', stop: str = "end_turn") -> None:
    msg = fake_message(text, model="claude-opus-5-5", stop_reason=stop)
    store.add(Record(jl.line.custom_id, jl.line.request_key, jl.line.condition, jl.line.body_sha256, "pilot",
                     "succeeded", msg, None, None, None, "t"))


def validate(output):
    if not output.get("ok"):
        raise InvalidVerdict("not ok")
    return output


@pytest.fixture
def store(tmp_path: Path) -> Store:
    return Store(tmp_path / "s.jsonl")


def roles(ls):
    (g,) = groups(ls).values()
    return g


def test_a_valid_primary_is_the_verdict(store) -> None:
    ls = lines()
    put(store, ls[0])
    j = resolve(roles(ls), store, {}, validate)
    assert j.state == "verdict" and j.used == ls[0].line.custom_id


def test_nothing_stored_means_send_the_primary(store) -> None:
    ls = lines()
    j = resolve(roles(ls), store, {}, validate)
    assert j.state == "send" and j.next is ls[0]


def test_invalid_or_refused_output_sends_the_retry_then_settles(store) -> None:
    for first in ({"text": '{"ok": false}'}, {"text": "", "stop": "refusal"}):
        s = Store(store.path.with_name(f"{len(first)}-{first.get('stop', 'x')}.jsonl"))
        ls = lines()
        put(s, ls[0], **first)
        j = resolve(roles(ls), s, {}, validate)
        assert j.state == "send" and j.next is ls[1]
        put(s, ls[1])
        j = resolve(roles(ls), s, {}, validate)
        assert j.state == "verdict" and j.used == ls[1].line.custom_id and len(j.log) == 2


def test_a_truncated_primary_sends_the_longer_rerun(store) -> None:
    ls = lines()
    put(store, ls[0], stop="max_tokens")
    j = resolve(roles(ls), store, {}, validate)
    assert j.state == "send" and j.next is ls[2]


def test_a_second_failure_leaves_the_answer_unjudged(store) -> None:
    ls = lines()
    put(store, ls[0], text='{"ok": false}')
    put(store, ls[1], stop="max_tokens")
    assert resolve(roles(ls), store, {}, validate).state == "unjudged"


def test_an_api_error_is_resubmitted_once(store) -> None:
    ls = lines()
    cid = ls[0].line.custom_id
    j = resolve(roles(ls), store, {cid: 1}, validate)
    assert j.state == "send" and j.next is ls[0]
    j = resolve(roles(ls), store, {cid: 2}, validate)
    assert j.state == "send" and j.next is ls[1]  # a second API error counts as a failure: the retry goes out
    j = resolve(roles(ls), store, {cid: 2, ls[1].line.custom_id: 2}, validate)
    assert j.state == "unjudged"


def test_failures_come_from_the_ledgers_errored_and_expired_items() -> None:
    from raglab.ledger import Item
    from decimal import Decimal

    items = {("g1", "a"): Item("g1", "a", "m", "pilot", "batch", Decimal(1), "settled", Decimal(0), "errored"),
             ("g2", "a"): Item("g2", "a", "m", "pilot", "batch", Decimal(1), "settled", Decimal(0), "expired"),
             ("g3", "b"): Item("g3", "b", "m", "pilot", "batch", Decimal(1), "settled", Decimal(0), "canceled"),
             ("g4", "c"): Item("g4", "c", "m", "pilot", "batch", Decimal(1), "settled", Decimal(1), "succeeded")}
    assert failures_from(items) == {"a": 2}


def test_a_retry_line_without_its_primary_is_refused() -> None:
    with pytest.raises(ValueError, match="without its primary"):
        groups(lines()[1:])


def test_a_retried_rate_limit_on_a_standard_call_is_never_a_judge_failure() -> None:
    from decimal import Decimal

    from raglab.ledger import Item

    items = {("g1", "a"): Item("g1", "a", "m", "pilot", "standard", Decimal(1), "settled", Decimal(0), "errored",
                               detail="rate_limited"),
             ("g2", "a"): Item("g2", "a", "m", "pilot", "standard", Decimal(1), "settled", Decimal(0), "errored",
                               detail="server"),
             ("g3", "b"): Item("g3", "b", "m", "pilot", "standard", Decimal(1), "settled", Decimal(0), "errored",
                               detail="client")}
    assert failures_from(items) == {"b": 1}  # a refusal of the request itself still counts
