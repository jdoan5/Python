"""Billed requests end to end on the fake API: the gate, order of operations, retries, crashes, batches."""

from __future__ import annotations

import socket
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from raglab.ledger import CapExceeded, Caps, Ledger, LedgerError, Pricing
from raglab.llm import ApiError, FakeTransport, LlmError, check_environment, fake_message
from raglab.runner import (
    Gate, Request, RunStopped, collect_batch, reconcile, requests_by_id, send_standard, send_stream, settle_from_store,
    submit_batch,
)
from raglab.store import Store

STAGE = Path(__file__).resolve().parents[1]
US = Decimal("1.1")


@pytest.fixture
def ledger(tmp_path: Path):
    caps = Caps(total=Decimal("1.00"), phases={"pilot": Decimal("0.50"), "main": Decimal("1.00")})
    with Ledger(tmp_path / "ledger.jsonl", Pricing.load(STAGE / "pricing.json"), caps) as led:
        yield led


@pytest.fixture
def store(tmp_path: Path) -> Store:
    return Store(tmp_path / "store.jsonl")


OPEN = Gate(lambda: "a" * 64, None)  # authorization always holds, every request allowed


def req(n: int = 0, sample: int = 0, model: str = "claude-haiku-5-5", max_tokens: int = 1000) -> Request:
    body = {"model": model, "max_tokens": max_tokens, "messages": [{"role": "user", "content": f"question {n}"}]}
    return Request(condition="hhr-4.8k-med", body=body, sample=sample, input_tokens=1100)


def answer(body) -> dict:
    return fake_message("An answer.", model=body["model"], input_tokens=1000, output_tokens=200)


def test_no_test_can_open_a_socket() -> None:
    with pytest.raises(OSError, match="never open network"):
        socket.create_connection(("api.anthropic.com", 443))


def test_the_environment_is_checked_and_values_never_echoed() -> None:
    check_environment({})
    check_environment({"ANTHROPIC_BASE_URL": "https://api.anthropic.com/"})
    secret = "not-a-real-key-0000"
    for env, match in (({"ANTHROPIC_API_KEY": secret}, "Keychain"),
                       ({"ANTHROPIC_AUTH_TOKEN": secret}, "ANTHROPIC_AUTH_TOKEN"),
                       ({"ANTHROPIC_PROFILE": secret}, "ANTHROPIC_PROFILE"),
                       ({"ANTHROPIC_CUSTOM_HEADERS": secret}, "ANTHROPIC_CUSTOM_HEADERS"),
                       ({"ANTHROPIC_LOG": "debug"}, "ANTHROPIC_LOG"),
                       ({"ANTHROPIC_BASE_URL": "https://example.com"}, "BASE_URL")):
        with pytest.raises(LlmError, match=match) as e:
            check_environment(env)
        assert secret not in str(e.value)


def test_the_key_comes_from_the_keychain_and_errors_never_show_it() -> None:
    import subprocess

    from raglab.llm import keychain_key

    def found(cmd, **kw):
        assert cmd == ["security", "find-generic-password", "-s", "raglab-stage4", "-w"]
        assert kw["timeout"] >= 120  # time to answer the Keychain's dialog
        return subprocess.CompletedProcess(cmd, 0, stdout="not-a-real-key-0000\n", stderr="")

    def missing(cmd, **kw):
        return subprocess.CompletedProcess(cmd, 44, stdout="", stderr="could not be found")

    assert keychain_key(run=found) == "not-a-real-key-0000"
    with pytest.raises(LlmError, match='-T ""') as e:
        keychain_key(run=missing)
    assert "could not be found" not in str(e.value)


def test_only_the_pinned_sdk_may_make_live_calls() -> None:
    import types

    from raglab.llm import check_sdk

    good = types.SimpleNamespace(__version__="2.6.0", _client=types.SimpleNamespace(ANTHROPIC_API_VERSION="2026-10-21"))
    check_sdk(good)
    with pytest.raises(LlmError, match="pinned"):
        check_sdk(types.SimpleNamespace(__version__="2.7.0", _client=good._client))
    with pytest.raises(LlmError, match="API version"):
        check_sdk(types.SimpleNamespace(__version__="2.6.0",
                                        _client=types.SimpleNamespace(ANTHROPIC_API_VERSION="2023-06-01")))
    import anthropic
    check_sdk(anthropic)  # the installed SDK is the pinned one


def test_a_standard_request_is_checked_reserved_sent_stored_then_settled(ledger, store) -> None:
    api = FakeTransport(answer)
    checks = []
    gate = Gate(lambda: checks.append(1) or "b" * 64, frozenset({req().custom_id}))
    record = send_standard(ledger, store, api, "pilot", req(), gate)
    assert record.response["content"][0]["text"] == "An answer." and record.group == "pilot-00001"
    assert [e["event"] for e in ledger.events()] == ["reserve", "sent", "settle"]
    assert next(ledger.events())["authorization"] == "b" * 64 and checks == [1]
    assert ledger.state.settled() == Decimal("0.0002")  # (1000 * 0.10 + 200 * 0.50) / 1e6
    assert send_standard(ledger, store, api, "pilot", req(), gate) == record  # stored, so never sent again
    assert sum(1 for c in api.calls if c[0] == "create") == 1


def test_a_request_off_the_authorized_list_is_never_sent(ledger, store) -> None:
    api = FakeTransport(answer)
    listed = Gate(lambda: "a" * 64, frozenset({req(0).custom_id}))
    with pytest.raises(RunStopped, match="authorized request list"):
        send_standard(ledger, store, api, "pilot", req(1), listed)
    with pytest.raises(RunStopped, match="authorized request list"):
        send_standard(ledger, store, api, "pilot", req(0, sample=7), listed)  # an unlisted sample of a listed key
    assert api.calls == [] and not list(ledger.events())


def test_an_interrupted_call_is_marked_unknown_never_left_sent(ledger, store) -> None:
    def interrupted(body):
        raise KeyboardInterrupt

    api = FakeTransport(interrupted)
    with pytest.raises(KeyboardInterrupt):
        send_standard(ledger, store, api, "pilot", req(3), OPEN)
    (item,) = ledger.state.items.values()
    assert item.status == "unknown" and ledger.state.blocking(req(3).custom_id)
    ledger.resend(item.group, item.custom_id, "checked the Console")
    api.responder = answer
    assert send_standard(ledger, store, api, "pilot", req(3), OPEN).outcome == "succeeded"


def test_a_flag_stops_every_later_reservation_until_it_is_acknowledged(ledger, store) -> None:
    def us(body):
        msg = answer(body)
        msg["usage"]["inference_geo"] = "us"
        return msg

    api = FakeTransport(us)
    with pytest.raises(RunStopped, match="US-only"):
        send_standard(ledger, store, api, "pilot", req(4), OPEN)
    api.responder = answer
    with pytest.raises(LedgerError, match="budget acknowledge"):
        send_standard(ledger, store, api, "pilot", req(5), OPEN)  # a second process can't spend past it
    ledger.accept_for_phase("pilot", "us_inference", "the workspace is US-only; 1.1x accepted")
    assert send_standard(ledger, store, api, "pilot", req(5), OPEN).outcome == "succeeded"
    api.responder = lambda body: fake_message("x", model="claude-opus-5-5", input_tokens=10, output_tokens=10)
    with pytest.raises(RunStopped, match="served by"):
        send_standard(ledger, store, api, "pilot", req(6), OPEN)
    with pytest.raises(LedgerError, match="budget acknowledge"):
        send_standard(ledger, store, api, "pilot", req(7), OPEN)  # phase acceptance covers US-only, nothing else
    (flagged,) = ledger.state.unacknowledged()
    ledger.acknowledge(flagged.group, flagged.custom_id, "read it")
    api.responder = answer
    assert send_standard(ledger, store, api, "pilot", req(7), OPEN).outcome == "succeeded"


def test_a_lapsed_authorization_stops_the_next_reservation(ledger, store) -> None:
    from raglab.authorize import AuthorizationError

    state = {"ok": True}

    def check():
        if not state["ok"]:
            raise AuthorizationError("PLAN.md changed since this phase was authorized")
        return "a" * 64

    api = FakeTransport(answer)
    send_standard(ledger, store, api, "pilot", req(0), Gate(check, None))
    state["ok"] = False
    with pytest.raises(AuthorizationError):
        send_standard(ledger, store, api, "pilot", req(1), Gate(check, None))
    assert sum(1 for c in api.calls if c[0] == "create") == 1


def test_a_new_sample_of_the_same_request_is_a_new_custom_id(ledger, store) -> None:
    api = FakeTransport(answer)
    a, b = (send_standard(ledger, store, api, "pilot", req(sample=s), OPEN) for s in (0, 1))
    assert a.request_key == b.request_key and a.custom_id != b.custom_id


def test_rate_limits_and_server_errors_are_retried_at_no_cost(ledger, store) -> None:
    failures = [ApiError("rate_limited", 429, retry_after=7.0), ApiError("server", 529)]
    waits = []

    def flaky(body):
        if failures:
            raise failures.pop(0)
        return answer(body)

    send_standard(ledger, store, FakeTransport(flaky), "pilot", req(), OPEN, sleep=waits.append)
    assert waits == [7.0, 2]
    assert [e["outcome"] for e in ledger.events() if e["event"] == "settle"] == ["errored", "errored", "succeeded"]
    assert ledger.state.settled() == Decimal("0.0002")


def test_repeated_overload_errors_stop_the_run_at_no_cost(ledger, store) -> None:
    def down(body):
        raise ApiError("server", 529)

    with pytest.raises(RunStopped):
        send_standard(ledger, store, FakeTransport(down), "pilot", req(), OPEN, sleep=lambda s: None)
    assert ledger.state.committed() == 0
    assert sum(1 for e in ledger.events() if e["event"] == "reserve") == 6


def test_a_500_may_follow_work_that_was_billed_so_it_stops_and_stays_counted(ledger, store) -> None:
    for status in (500, 502, 504, 524):
        r = req(20 + status)
        with pytest.raises(RunStopped, match="may have been billed"):
            send_standard(ledger, store, FakeTransport(lambda body, s=status: (_ for _ in ()).throw(
                ApiError("server", s))), "pilot", r, OPEN, sleep=lambda s: None)
        assert ledger.state.blocking(r.custom_id)  # unknown, counted at its bound, until a person resends it
    assert ledger.summary()["unknown"] == 4


def test_a_dropped_connection_stops_the_run_and_a_second_process_cant_resend_it(ledger, store) -> None:
    def drop(body):
        raise ApiError("connection", message="APIConnectionError")

    with pytest.raises(RunStopped, match="may have been billed"):
        send_standard(ledger, store, FakeTransport(drop), "pilot", req(), OPEN)
    bound = (1100 * Decimal("0.10") + 1000 * Decimal("0.50")) / 10**6 * US
    assert ledger.state.unknown() and ledger.state.committed() == bound and len(store) == 0
    with pytest.raises(LedgerError, match="unknown"):
        send_standard(ledger, store, FakeTransport(answer), "pilot", req(), OPEN)


def test_a_client_error_stops_the_run_at_no_cost(ledger, store) -> None:
    def bad(body):
        raise ApiError("client", 400, message="invalid_request_error")

    with pytest.raises(RunStopped, match="400"):
        send_standard(ledger, store, FakeTransport(bad), "pilot", req(), OPEN)
    assert ledger.state.committed() == 0


def test_the_cap_stops_a_request_before_it_is_sent(ledger, store) -> None:
    api = FakeTransport(answer)
    big = req(model="claude-opus-5-5", max_tokens=30_000)  # 30,000 x $20/M = $0.60 > the $0.50 pilot cap
    with pytest.raises(CapExceeded):
        send_standard(ledger, store, api, "pilot", big, OPEN)
    assert api.calls == []


def test_a_flagged_settlement_stops_the_run_after_it_is_booked(ledger, store) -> None:
    def us_only(body):
        m = answer(body)
        m["usage"]["inference_geo"] = "us"
        return m

    with pytest.raises(RunStopped, match="US-only"):
        send_standard(ledger, store, FakeTransport(us_only), "pilot", req(), OPEN)
    assert ledger.state.settled() == Decimal("0.0002") * US and len(store) == 1  # booked at the dearer rate


def test_a_batch_settles_each_outcome_and_resubmits_only_what_failed(ledger, store) -> None:
    reqs = [req(n) for n in range(5)]
    plans = requests_by_id(reqs)
    outcome = {reqs[1].custom_id: "errored", reqs[2].custom_id: "expired"}
    api = FakeTransport(answer, batch_outcome=lambda cid: outcome.get(cid, "succeeded"), polls_until_ended=2)
    group = submit_batch(ledger, store, api, "main", reqs, OPEN)
    assert collect_batch(ledger, store, api, group, plans)["status"] == "in_progress"
    counts = collect_batch(ledger, store, api, group, plans)
    assert (counts["succeeded"], counts["errored"], counts["expired"]) == (3, 1, 1)
    assert len(store) == 3 and ledger.state.committed() == 3 * Decimal("0.0001")
    assert collect_batch(ledger, store, api, group, plans)["already"] == 5  # collecting again changes nothing
    outcome.clear()
    again = submit_batch(ledger, store, api, "main", reqs, OPEN)
    assert api.calls[-1] == ("batch_create", [reqs[1].custom_id, reqs[2].custom_id])
    while collect_batch(ledger, store, api, again, plans)["status"] != "ended":
        pass
    assert len(store) == 5 and submit_batch(ledger, store, api, "main", reqs, OPEN) is None


def test_a_batch_keeps_to_its_models_limit(ledger, store) -> None:
    reqs = [req(n, model="claude-opus-5-5", max_tokens=10) for n in range(51)]
    with pytest.raises(LedgerError, match="batch limit of 50"):
        submit_batch(ledger, store, FakeTransport(answer), "main", reqs, OPEN)


def test_a_canceled_batch_settles_its_unprocessed_items_at_zero(ledger, store) -> None:
    reqs = [req(n) for n in range(3)]
    api = FakeTransport(answer, polls_until_ended=99)
    group = submit_batch(ledger, store, api, "main", reqs, OPEN)
    api.batch_cancel(ledger.state.groups[group]["sent"]["batch_id"])
    counts = collect_batch(ledger, store, api, group, requests_by_id(reqs))
    assert counts["canceled"] == 3 and ledger.state.committed() == 0


class Lost(FakeTransport):
    def batch_create(self, requests):
        super().batch_create(requests)  # the API made the batch, but the response never arrived
        raise ApiError("connection")


def test_a_lost_batch_create_blocks_every_new_batch_until_reconcile(ledger, store) -> None:
    reqs = [req(n) for n in range(2)]
    api = Lost(answer)
    with pytest.raises(RunStopped, match="reconcile"):
        submit_batch(ledger, store, api, "main", reqs, OPEN)
    held = ledger.state.committed()
    assert held > 0  # still counted: it may exist
    with pytest.raises(RunStopped, match="reconcile"):
        submit_batch(ledger, store, FakeTransport(answer), "main", [req(9)], OPEN)  # nothing new goes out
    # While the batch runs its requests can't be listed, so nothing is matched and nothing is released.
    report = reconcile(ledger, api, release=True)
    assert report["unmatched_batches"] == ["msgbatch_fake0001"] and not report["released"]
    assert ledger.state.committed() == held
    api.batch_retrieve("msgbatch_fake0001")  # time passes; the batch ends
    report = reconcile(ledger, api, release=True)
    assert list(report["attached"].values()) == ["msgbatch_fake0001"] and not report["released"]
    collect_batch(ledger, store, api, next(iter(report["attached"])), requests_by_id(reqs))
    assert len(store) == 2


def test_reconcile_releases_a_reservation_the_api_never_saw(ledger, store) -> None:
    class Refused(FakeTransport):
        def batch_create(self, requests):
            raise ApiError("connection")  # failed before reaching the API

    api = Refused(answer)
    with pytest.raises(RunStopped):
        submit_batch(ledger, store, api, "main", [req(n) for n in range(2)], OPEN)
    report = reconcile(ledger, api, release=True)
    assert report["released"] and ledger.state.committed() == 0


def test_reconcile_pages_back_past_newer_batches_and_reads_real_datetimes(ledger, store) -> None:
    reqs = [req(n) for n in range(2)]

    class Busy(Lost):
        def batch_list(self):
            # The real SDK yields datetimes, and fifty newer batches from other work sit in front of ours. A
            # running one is never released against, so here they have all ended and hold other requests.
            for i in range(50):
                self._batches[f"msgbatch_other{i:02d}"] = {"requests": [{"custom_id": f"other{i}",
                                                                         "params": {"model": "claude-haiku-5-5"}}],
                                                           "status": "ended",
                                                           "created_at": "2100-01-01T00:00:00Z"}
            for batch in super().batch_list():
                yield {**batch, "created_at": datetime.fromisoformat(batch["created_at"].replace("Z", "+00:00"))}

    api = Busy(answer)
    with pytest.raises(RunStopped):
        submit_batch(ledger, store, api, "main", reqs, OPEN)
    api.batch_retrieve("msgbatch_fake0001")
    report = reconcile(ledger, api, release=True)
    assert list(report["attached"].values()) == ["msgbatch_fake0001"]
    assert len(report["unmatched_batches"]) == 50  # the others were examined, never attached or released


def test_a_second_paid_attempt_at_an_answered_request_is_refused_before_it_is_sent(ledger, store) -> None:
    reqs = [req(0)]
    api = FakeTransport(answer)
    group = submit_batch(ledger, store, api, "main", reqs, OPEN)
    collect_batch(ledger, store, api, group, requests_by_id(reqs))
    # Even with the store lost (iCloud, say), the ledger knows the request was answered and paid for.
    empty = Store(store.path.with_name("lost.jsonl"))
    calls = len(api.calls)
    assert submit_batch(ledger, empty, api, "main", reqs, OPEN) is None  # skipped: answered and paid for
    with pytest.raises(LedgerError, match="already answered"):
        send_standard(ledger, empty, api, "main", reqs[0], OPEN)  # the ledger refuses it in any case
    assert len(api.calls) == calls  # nothing went out


def test_a_different_response_to_a_stored_request_goes_to_the_side_file(store) -> None:
    from raglab.runner import _record, _store

    r = req(0)
    assert _store(store, _record(r, "main", fake_message("An answer."), "msgbatch_1", "main-00001"))
    assert not _store(store, _record(r, "main", fake_message("A different answer."), "msgbatch_2", "main-00002"))
    assert store.get(r.custom_id).response["content"][0]["text"] == "An answer."
    assert store.duplicates_path.read_text(encoding="utf-8").count("A different answer.") == 1


def test_a_crash_between_storing_and_settling_is_repaired_from_its_own_attempt(ledger, store) -> None:
    reqs = [req(0)]
    api = FakeTransport(answer)
    group = submit_batch(ledger, store, api, "main", reqs, OPEN)
    batch_id = ledger.state.groups[group]["sent"]["batch_id"]
    api.batch_retrieve(batch_id)
    item = next(iter(api.batch_results(batch_id)))
    from raglab.runner import _record
    store.add(_record(reqs[0], "main", item["result"]["message"], batch_id, group))  # stored, then the process died
    assert ledger.state.items[(group, reqs[0].custom_id)].status == "sent"
    assert settle_from_store(ledger, store) == 1
    assert ledger.state.committed() == Decimal("0.0001")


def test_a_cache_marker_in_the_body_is_never_missed() -> None:
    plain = req(0)
    marked = Request("judge", {**plain.body, "system": [{"type": "text", "text": "rubric",
                                                         "cache_control": {"type": "ephemeral"}}]}, 0, 100)
    assert not plain.cache_marked and marked.cache_marked and marked.planned().cache_marked


def test_a_streamed_request_keeps_its_timing_and_is_priced_as_standard(ledger, store) -> None:
    api = FakeTransport(answer)
    rec = send_stream(ledger, store, api, "pilot", req(7), OPEN)
    assert [c[0] for c in api.calls] == ["stream"]
    assert rec.timing == {"first_text_seconds": 0.1, "total_seconds": 0.5}
    group = rec.group
    assert ledger.state.groups[group]["mode"] == "standard"
    assert store.get(rec.custom_id).timing == rec.timing
    assert send_stream(ledger, store, api, "pilot", req(7), OPEN) == rec  # stored: never streamed again
    assert len(api.calls) == 1


def test_a_stream_that_breaks_after_opening_stops_the_run_and_stays_counted(ledger, store) -> None:
    def broken(body):
        raise ApiError("connection", message="stream broke after it opened (ReadError)")

    api = FakeTransport(broken)
    api.stream = lambda body: broken(body)  # the transport's own classification of a post-open failure
    with pytest.raises(RunStopped, match="may have been billed"):
        send_stream(ledger, store, api, "pilot", req(8), OPEN)
    assert ledger.state.blocking(req(8).custom_id)
    assert ledger.summary()["unknown"] == 1
    with pytest.raises(LedgerError):
        send_stream(ledger, store, api, "pilot", req(8), OPEN)  # a second process can't send it again


def test_a_stream_refused_before_it_opens_is_retried_at_no_cost(ledger, store) -> None:
    waits = []
    tries = {"n": 0}

    def flaky(body):
        tries["n"] += 1
        if tries["n"] == 1:
            raise ApiError("rate_limited", 429, 2.0)
        return {"message": answer(body), "first_text_seconds": 0.2, "total_seconds": 0.9}

    api = FakeTransport(answer)
    api.stream = flaky
    rec = send_stream(ledger, store, api, "pilot", req(9), OPEN, sleep=waits.append)
    assert waits == [2.0] and rec.timing["total_seconds"] == 0.9


def test_a_response_from_a_model_with_no_price_settles_at_its_bound(ledger, store) -> None:
    api = FakeTransport(lambda body: fake_message("x", model="claude-sonnet-5-5"))
    with pytest.raises(RunStopped, match="can't price"):
        send_standard(ledger, store, api, "pilot", req(8), OPEN)
    (item,) = ledger.state.items.values()
    assert item.status == "settled" and item.cost == item.bound  # never left open, never undercounted


@pytest.mark.parametrize("failing_fsync, healed", [(3, True), (4, False)])
def test_a_failed_mark_sent_writes_nothing_more_and_reopening_heals_it(tmp_path, store, monkeypatch, failing_fsync,
                                                                      healed) -> None:
    """An interrupt after the 'sent' line reached the ledger file: this process writes nothing more, and the next
    open reads what reached disk (healing a mirror one event behind). Before this rule, a 'release' written after
    'sent' left a ledger no command could open. Fsync 3 is the ledger file's for 'sent' (the mirror never gets
    the line); fsync 4 is the mirror's (both copies have it)."""
    import os as os_module

    caps = Caps(total=Decimal("1.00"), phases={"pilot": Decimal("0.50")})
    pricing = Pricing.load(STAGE / "pricing.json")
    path, mirror = tmp_path / "ledger.jsonl", tmp_path / "mirror" / "ledger.jsonl"
    real_fsync = os_module.fsync
    calls = {"n": 0}
    with Ledger(path, pricing, caps, mirror=mirror) as ledger:
        def flaky_fsync(fd):
            calls["n"] += 1
            if calls["n"] == failing_fsync:
                raise KeyboardInterrupt
            return real_fsync(fd)

        monkeypatch.setattr(os_module, "fsync", flaky_fsync)
        with pytest.raises(KeyboardInterrupt):
            send_standard(ledger, store, FakeTransport(answer), "pilot", req(13), OPEN)
        monkeypatch.setattr(os_module, "fsync", real_fsync)
        with pytest.raises(LedgerError, match="interrupted"):
            ledger.mark_unknown("pilot-00001", req(13).custom_id, "anything")  # refused: disk may be ahead
    with Ledger(path, pricing, caps, mirror=mirror) as reopened:
        (item,) = reopened.state.items.values()
        assert item.status == "sent" and bool(reopened.repairs) is healed
        reopened.mark_unknown(item.group, item.custom_id, "orphaned")  # what `batch orphan` does
        assert reopened.state.blocking(item.custom_id)


def test_a_failed_mark_sent_before_any_write_leaves_a_reservation_orphan_can_release(ledger, store,
                                                                                    monkeypatch) -> None:
    def boom(group, **kw):
        raise KeyboardInterrupt

    monkeypatch.setattr(ledger, "mark_sent", boom)
    with pytest.raises(KeyboardInterrupt):
        send_standard(ledger, store, FakeTransport(answer), "pilot", req(11), OPEN)
    (item,) = ledger.state.items.values()
    assert item.status == "reserved" and [e["event"] for e in ledger.events()] == ["reserve"]


def test_a_500_on_batch_create_keeps_the_reservation_for_reconcile(ledger, store) -> None:
    reqs = [req(30), req(31)]
    api = FakeTransport(answer)
    api.batch_create = lambda requests: (_ for _ in ()).throw(ApiError("server", 502))
    with pytest.raises(RunStopped, match="batch reconcile"):
        submit_batch(ledger, store, api, "main", reqs, OPEN)
    assert all(i.status == "reserved" for i in ledger.state.items.values())  # never released: it may exist
    api2 = FakeTransport(answer)
    with pytest.raises(RunStopped, match="reconcile"):
        submit_batch(ledger, store, api2, "main", reqs, OPEN)  # no new batch while that create is unresolved
    api.batch_create = lambda requests: (_ for _ in ()).throw(ApiError("server", 529))
    other = [req(32)]
    ledger.release(next(iter(ledger.state.groups)), "reconcile found no batch")
    with pytest.raises(RunStopped, match="was refused"):
        submit_batch(ledger, store, api, "main", other, OPEN)  # overloaded: refused before any work
    assert not ledger.state.blocking(other[0].custom_id)


def test_a_retried_server_error_is_settled_with_its_kind(ledger, store) -> None:
    tries = {"n": 0}

    def flaky(body):
        tries["n"] += 1
        if tries["n"] == 1:
            raise ApiError("server", 529)
        return answer(body)

    send_standard(ledger, store, FakeTransport(flaky), "pilot", req(12), OPEN, sleep=lambda s: None)
    details = sorted((i.outcome, i.detail) for i in ledger.state.items.values())
    assert details == [("errored", "server"), ("succeeded", None)]


def test_a_wave_cuts_its_last_batch_to_what_the_caps_leave(tmp_path, store) -> None:
    from raglab.runner import submit_wave

    pricing = Pricing.load(STAGE / "pricing.json")
    reqs = [req(100 + n, max_tokens=4000) for n in range(10)]
    one = pricing.bound("claude-haiku-5-5", 1100, 4000, False, "batch")
    caps = Caps(total=Decimal("1.00"), phases={"main": one * Decimal("3.5")})  # room for 3 of the 10, not a batch
    api = FakeTransport(answer)
    lines = []
    with Ledger(tmp_path / "wave.jsonl", pricing, caps) as led:
        assert submit_wave(led, store, api, "main", reqs, OPEN, report=lines.append) == 1
        (created,) = [c for c in api.calls if c[0] == "batch_create"]
        assert len(created[1]) == 3 and "room for 3 of 10" in lines[-1]
        assert submit_wave(led, store, api, "main", reqs[3:], OPEN, report=lines.append) == 0  # nothing fits now
        assert lines[-1].startswith("stopping for now")
