"""Stage 4's commands end to end on the fake API: plan, smoke test, batches, collection, the timing probe.

The live pieces (the freeze-point check, the Keychain, the signed authorization)
have their own tests; here `_live` is replaced by an open gate and the fake
transport, so these tests check what each command sends, in what order, and
what it refuses.
"""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest
from rich.console import Console
from typer.testing import CliRunner

from raglab import cli, cli4, llm
from raglab.blocks import window_documents
from raglab.generate import generation_body
from raglab.ledger import Caps, Ledger, Pricing
from raglab.llm import FakeTransport, fake_message
from raglab.plan4 import line_for, read_plan
from raglab.runner import Gate
from raglab.sections import SectionIndex
from raglab.store import Store
from raglab.windows import CONDITIONS
from tests.conftest import make_doc

STAGE = Path(__file__).resolve().parents[1]
runner = CliRunner()

PAGE = """\
# Errors { #errors }

Use `HTTPException` to return errors.

It carries a status code.
"""

LINES = [("hhr-4.8k-med", "batch", "answer"), ("base-4.8k-med", "batch", "answer"),
         ("smoke:hhr-4.8k-med", "standard", "smoke"), ("smoke:base-4.8k-med", "standard", "smoke"),
         ("smoke:closed-med", "standard", "smoke"), ("timing:hhr-4.8k-low", "stream", "timing")]


def cited(body) -> dict:
    """The fake model: cites the first two blocks of the first document, exactly as the API copies them."""
    content = body["messages"][0]["content"]
    documents = [c for c in content if c.get("type") == "document"]
    if not documents:
        return fake_message("Use HTTPException.")
    blocks = documents[0]["source"]["content"]
    citation = {"type": "content_block_location", "document_index": 0, "start_block_index": 0,
                "end_block_index": 2, "document_title": documents[0].get("title"),
                "cited_text": "".join(b["text"] for b in blocks[:2])}
    msg = fake_message(content=[{"type": "text", "text": "Raise HTTPException with a status code.",
                                 "citations": [citation]}])
    msg["usage"] |= {"inference_geo": "global", "service_tier": "standard"}
    return msg


@pytest.fixture
def stage4(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(cli4, "console", Console(width=250))
    doc = make_doc(PAGE, "tutorial/errors")
    index = SectionIndex([doc])
    docs = window_documents([(doc.doc_id, 0, len(doc.text.rstrip()))], {doc.doc_id: doc}, index)

    def requests(which):
        out = []
        for n, (name, mode, role) in enumerate(LINES):
            cond = CONDITIONS[name.split(":")[-1]]
            shown = [] if cond.closed else docs
            body = generation_body(cond, f"question {n}?", shown, 1000, "answer prompt", "closed prompt")
            out.append((line_for(name, f"p{n:02d}", 0, body, mode=mode, role=role), body, shown))
        return out

    api = FakeTransport(cited)
    store = Store(cli4.api_dir() / "store" / "pilot.jsonl")  # where every command looks: a temp folder in tests
    caps = Caps(total=Decimal("25.00"), phases={"pilot": Decimal("6.50"), "generation": Decimal("3.00"),
                                                "judging": Decimal("25.00")})
    pricing = Pricing.load(STAGE / "pricing.json")
    lists = {**cli4.REQUEST_LISTS, "pilot": tmp_path / "pilot" / "requests.jsonl"}
    monkeypatch.setattr(cli4, "REQUEST_LISTS", lists)
    monkeypatch.setattr(cli4, "SMOKE_REPORT", tmp_path / "pilot" / "SMOKE.json")
    monkeypatch.setattr(cli4, "generation_requests", requests)
    monkeypatch.setattr(cli4, "open_ledger",
                        lambda repair=False: Ledger(cli4.api_dir() / "ledger.jsonl", pricing, caps))
    live = []

    def fake_live(auth_phase, requests_path, keys):
        live.append((auth_phase, keys))
        return Gate(lambda: "a" * 64, keys), api, Store(store.path)  # read from disk, as the real one is

    monkeypatch.setattr(cli4, "_live", fake_live)
    monkeypatch.setattr(llm, "AnthropicTransport", lambda key: api)
    monkeypatch.setattr(llm, "keychain_key", lambda: "not-a-real-key")
    return {"api": api, "store": store, "tmp": tmp_path, "live": live, "ledger": cli4.open_ledger}


def run(*args: str):
    return runner.invoke(cli.app, list(args))


def kinds(api) -> list[str]:
    return [c[0] for c in api.calls if c[0] != "count_tokens"]


def test_the_pilot_plan_lists_batch_smoke_and_timing_lines(stage4) -> None:
    r = run("gen", "plan", "--set", "pilot")
    assert r.exit_code == 0, r.output
    lines = read_plan(stage4["tmp"] / "pilot" / "requests.jsonl")
    assert [(l.mode, l.role) for l in lines] == [(m, r) for _, m, r in LINES]
    assert all(l.input_tokens is None for l in lines)  # counted (free) when sent, never before freeze point 1


def test_pilot_batches_wait_for_a_passed_smoke_test(stage4) -> None:
    assert run("gen", "plan", "--set", "pilot").exit_code == 0
    r = run("gen", "submit", "--set", "pilot")
    assert r.exit_code == 1 and "smoke test" in r.output
    assert stage4["api"].calls == []


def test_the_smoke_test_sends_its_answers_one_at_a_time_and_records_its_checks(stage4) -> None:
    assert run("gen", "plan", "--set", "pilot").exit_code == 0
    r = run("gen", "smoke")
    assert r.exit_code == 0, r.output
    assert kinds(stage4["api"]) == ["create"] * 3  # the three smoke lines, standard API, nothing batched
    report = json.loads((stage4["tmp"] / "pilot" / "SMOKE.json").read_text())
    assert report["verdict"]["passed"] and report["verdict"]["join_rule"] == "concatenated"
    assert report["verdict"]["inference_geo"] == ["global", "None"] or report["verdict"]["inference_geo"]
    text = json.dumps(report)
    assert "HTTPException` to return" not in text  # no documentation text in the report
    assert Decimal(report["settled"]) > 0
    # Run again: everything is stored, so nothing is sent twice.
    assert run("gen", "smoke").exit_code == 0
    assert kinds(stage4["api"]) == ["create"] * 3


def test_after_the_smoke_test_only_batch_lines_are_batched_and_collected(stage4) -> None:
    assert run("gen", "plan", "--set", "pilot").exit_code == 0
    assert run("gen", "smoke").exit_code == 0
    r = run("gen", "submit", "--set", "pilot")
    assert r.exit_code == 0, r.output
    batches = [c for c in stage4["api"].calls if c[0] == "batch_create"]
    assert len(batches) == 1 and len(batches[0][1]) == 2  # the two answer lines, not smoke or timing
    r = run("gen", "collect", "--set", "pilot")
    assert r.exit_code == 0, r.output
    assert len(Store(stage4["store"].path)) == 3 + 2  # reread from disk: what collect wrote
    assert run("gen", "submit", "--set", "pilot").output.startswith("nothing to submit")


def test_the_timing_probe_streams_and_keeps_its_timings(stage4) -> None:
    assert run("gen", "plan", "--set", "pilot").exit_code == 0
    assert run("gen", "smoke").exit_code == 0
    stage4["api"].calls.clear()
    r = run("gen", "timing")
    assert r.exit_code == 0, r.output
    assert kinds(stage4["api"]) == ["stream"]
    timed = [rec for rec in Store(stage4["store"].path) if rec.timing]
    assert len(timed) == 1 and timed[0].timing["total_seconds"] == 0.5
    with stage4["ledger"]() as ledger:
        assert {i.mode for i in ledger.state.items.values()} == {"standard"}


def test_a_failed_smoke_citation_fails_the_command(stage4, monkeypatch) -> None:
    def broken(body):
        msg = cited(body)
        for block in msg["content"]:
            for c in block.get("citations") or ():
                c["cited_text"] = "not what the blocks say"
        return msg

    stage4["api"].responder = broken
    assert run("gen", "plan", "--set", "pilot").exit_code == 0
    r = run("gen", "smoke")
    assert r.exit_code == 1 and "failed their check" in r.output
    assert not json.loads((stage4["tmp"] / "pilot" / "SMOKE.json").read_text())["verdict"]["passed"]
    assert run("gen", "submit", "--set", "pilot").exit_code == 1  # and the batches stay blocked


def test_no_test_can_reach_the_real_spend_records() -> None:
    from raglab.corpus import project_root
    from raglab.ledger import default_mirror

    assert not cli4.api_dir().is_relative_to(project_root())
    assert not default_mirror().is_relative_to(Path.home() / "Library")


def test_batch_status_shows_open_batches_and_unknown_attempts(stage4) -> None:
    from raglab.ledger import Planned

    with stage4["ledger"]() as ledger:
        open_batch = ledger.reserve("pilot", "batch", [Planned("r" + "1" * 56 + "s00", "claude-haiku-5-5", 10, 10,
                                                               False)], "a" * 64)
        ledger.mark_sent(open_batch, batch_id="msgbatch_open")
        lost = ledger.reserve("pilot", "standard", [Planned("r" + "2" * 56 + "s00", "claude-haiku-5-5", 10, 10,
                                                            False)], "a" * 64)
        ledger.mark_sent(lost)
        ledger.mark_unknown(lost, "r" + "2" * 56 + "s00", "connection dropped")
    r = run("batch", "status")
    assert r.exit_code == 0, r.output
    assert "msgbatch_open" in r.output and lost in r.output


def test_a_resend_is_typed_by_the_developer_and_still_counts(stage4, monkeypatch) -> None:
    from raglab.ledger import Planned

    cid = "r" + "3" * 56 + "s00"
    with stage4["ledger"]() as ledger:
        group = ledger.reserve("pilot", "standard", [Planned(cid, "claude-haiku-5-5", 10, 10, False)], "a" * 64)
        ledger.mark_sent(group)
        ledger.mark_unknown(group, cid, "connection dropped")
        committed = ledger.state.committed()
    monkeypatch.setenv("CLAUDECODE", "1")
    r = run("batch", "resend", group, cid, "--reason", "the Console shows no call that minute")
    assert r.exit_code == 1 and "own terminal" in r.output
    monkeypatch.setattr(cli4, "typed_by_developer", lambda expected, prompt: None)  # the developer typed it
    r = run("batch", "resend", group, cid, "--reason", "the Console shows no call that minute")
    assert r.exit_code == 0, r.output
    with stage4["ledger"]() as ledger:
        assert (group, cid) in ledger.state.cleared and ledger.state.committed() == committed
        assert not ledger.state.blocking(cid)


def test_cancel_needs_a_batch_and_reconcile_attaches_a_lost_create(stage4) -> None:
    assert run("batch", "cancel", "pilot-09999").exit_code == 1
    from raglab.ledger import Planned

    cid = "r" + "4" * 56 + "s00"
    with stage4["ledger"]() as ledger:
        group = ledger.reserve("pilot", "batch", [Planned(cid, "claude-haiku-5-5", 10, 10, False)], "a" * 64)
    body = {"model": "claude-haiku-5-5", "max_tokens": 10,
            "messages": [{"role": "user", "content": [{"type": "text", "text": "q"}]}]}
    created = stage4["api"].batch_create([{"custom_id": cid, "params": body}])
    stage4["api"].batch_retrieve(created["id"])  # it ended
    r = run("batch", "reconcile")
    assert r.exit_code == 0, r.output
    assert json.loads(r.output)["attached"] == {group: created["id"]}
    assert run("batch", "cancel", group).exit_code == 0


def test_acknowledging_flags_is_typed_by_the_developer(stage4, monkeypatch) -> None:
    from raglab.ledger import Planned

    cid = "r" + "5" * 56 + "s00"
    with stage4["ledger"]() as ledger:
        group = ledger.reserve("pilot", "standard", [Planned(cid, "claude-haiku-5-5", 10, 10, False)], "a" * 64)
        ledger.mark_sent(group)
        ledger.settle(group, cid, "succeeded", "claude-opus-5-5", {"input_tokens": 10, "output_tokens": 10})
        assert ledger.state.unacknowledged()
    monkeypatch.setenv("CLAUDECODE", "1")
    assert run("budget", "acknowledge", "--reason", "read").exit_code == 1
    monkeypatch.setattr(cli4, "typed_by_developer", lambda expected, prompt: None)
    r = run("budget", "acknowledge", "--reason", "served by Opus once; fine")
    assert r.exit_code == 0 and "1 flag(s) acknowledged" in r.output
    with stage4["ledger"]() as ledger:
        assert not ledger.state.unacknowledged()


def test_an_orphaned_standard_call_becomes_unknown_and_the_judging_cap_waits_for_it(stage4) -> None:
    from raglab.ledger import Planned

    cid = "r" + "6" * 56 + "s00"
    with stage4["ledger"]() as ledger:
        group = ledger.reserve("pilot", "standard", [Planned(cid, "claude-haiku-5-5", 10, 10, False)], "a" * 64)
        ledger.mark_sent(group)  # then the process was killed
    r = run("batch", "orphan", group)
    assert r.exit_code == 0, r.output
    with stage4["ledger"]() as ledger:
        assert ledger.state.items[(group, cid)].status == "unknown"
        with pytest.raises(cli4.LedgerError, match="open or unknown"):
            cli4.phase_cap("judging", ledger)
        ledger.resend(group, cid, "the Console shows it never ran")
        assert cli4.phase_cap("judging", ledger) < Decimal("25.00")  # cleared: still counted at its bound


def test_a_failed_generation_request_is_resubmitted_once_then_left_as_an_error(stage4) -> None:
    assert run("gen", "plan", "--set", "pilot").exit_code == 0
    assert run("gen", "smoke").exit_code == 0
    stage4["api"].batch_outcome = lambda cid: "errored"
    for _ in range(2):
        assert run("gen", "submit", "--set", "pilot").exit_code == 0
        assert run("gen", "collect", "--set", "pilot").exit_code == 0
    r = run("gen", "submit", "--set", "pilot")
    assert "left as status error" in r.output and "nothing to submit" in r.output
    assert len([c for c in stage4["api"].calls if c[0] == "batch_create"]) == 2


def test_a_lost_store_is_restored_from_the_batch_without_paying_again(stage4) -> None:
    assert run("gen", "plan", "--set", "pilot").exit_code == 0
    assert run("gen", "smoke").exit_code == 0
    assert run("gen", "submit", "--set", "pilot").exit_code == 0
    assert run("gen", "collect", "--set", "pilot").exit_code == 0
    path = stage4["store"].path
    kept = [l for l in path.read_text().splitlines() if '"batch_id":null' in l]  # the smoke answers stay
    path.write_text("\n".join(kept) + "\n")
    assert len(Store(path)) == 3
    with stage4["ledger"]() as ledger:
        group = next(g for g, info in ledger.state.groups.items() if info["mode"] == "batch")
    r = run("gen", "collect", "--set", "pilot", "--refetch", group)
    assert r.exit_code == 0 and "2 lost record(s) restored" in r.output
    assert len(Store(path)) == 5
    assert len([c for c in stage4["api"].calls if c[0] == "batch_create"]) == 1


def test_the_golden_list_is_counted_and_bounded(stage4, monkeypatch, tmp_path) -> None:
    from raglab import freeze

    lists = {**cli4.REQUEST_LISTS, "generation": tmp_path / "plan" / "requests.jsonl"}
    monkeypatch.setattr(cli4, "REQUEST_LISTS", lists)
    closed = []
    monkeypatch.setattr(freeze, "forbid", lambda fp, what, *a, **k: closed.append((fp, what)))
    assert run("gen", "plan", "--set", "golden").exit_code == 1  # never written uncounted
    r = run("gen", "plan", "--set", "golden", "--count")
    assert r.exit_code == 0, r.output
    lines = read_plan(lists["generation"])
    assert lines and all(l.counted_input == 1000 and l.input_tokens and l.bound for l in lines)
    assert [c[0] for c in stage4["api"].calls if c[0] != "count_tokens"] == []  # counting is all it did
    assert closed == [(3, "planning golden generation")]  # refused once the golden answers are sealed

    def sealed(fp, what, *a, **k):
        raise freeze.FreezeError(f"{what} is closed")

    monkeypatch.setattr(freeze, "forbid", sealed)
    for args in (("gen", "plan", "--set", "golden", "--count"), ("gen", "collect", "--set", "golden")):
        r = run(*args)
        assert r.exit_code == 1 and "is closed" in r.output, r.output


def test_orphan_releases_a_standard_reservation_that_was_never_sent(stage4) -> None:
    from raglab.ledger import Planned

    cid = "r" + "7" * 56 + "s00"
    with stage4["ledger"]() as ledger:
        group = ledger.reserve("pilot", "standard", [Planned(cid, "claude-haiku-5-5", 10, 10, False)], "a" * 64)
    r = run("batch", "orphan", group)
    assert r.exit_code == 0 and "released at no cost" in r.output
    with stage4["ledger"]() as ledger:
        assert ledger.state.items[(group, cid)].status == "released" and not ledger.state.blocking(cid)


def test_a_us_only_workspace_passes_the_smoke_test_once_the_developer_accepts_it(stage4, monkeypatch) -> None:
    def us(body):
        msg = cited(body)
        msg["usage"]["inference_geo"] = "us"
        return msg

    stage4["api"].responder = us
    assert run("gen", "plan", "--set", "pilot").exit_code == 0
    r = run("gen", "smoke")
    assert r.exit_code == 1 and "US-only" in r.output
    assert len([c for c in stage4["api"].calls if c[0] == "create"]) == 1  # stopped after the first, booked
    monkeypatch.setattr(cli4, "typed_by_developer", lambda expected, prompt: None)  # the developer typed it
    assert run("budget", "acknowledge", "--us-inference", "pilot", "--reason", "workspace is US-only").exit_code == 0
    r = run("gen", "smoke")
    assert r.exit_code == 0, r.output
    assert len([c for c in stage4["api"].calls if c[0] == "create"]) == 3  # the stored answer was never re-sent
    report = json.loads((stage4["tmp"] / "pilot" / "SMOKE.json").read_text())
    assert report["verdict"]["passed"] and report["verdict"]["notes"] and report["verdict"]["inference_geo"] == ["us"]


def test_us_acceptance_names_the_ledger_phase_and_refuses_a_typo(stage4, monkeypatch) -> None:
    monkeypatch.setattr(cli4, "typed_by_developer", lambda expected, prompt: None)
    r = run("budget", "acknowledge", "--us-inference", "pilot-judging", "--reason", "US workspace")
    assert r.exit_code == 0 and "accepted for pilot" in r.output  # pilot-judging's calls are booked under pilot
    with stage4["ledger"]() as ledger:
        assert ("pilot", "us_inference") in ledger.state.accepted
    assert run("budget", "acknowledge", "--us-inference", "pilt", "--reason", "typo").exit_code == 1


def test_the_timing_probe_waits_for_the_smoke_test(stage4) -> None:
    (stage4["tmp"] / "pilot").mkdir(exist_ok=True)
    assert run("gen", "plan", "--set", "pilot").exit_code == 0
    stage4["tmp"].joinpath("pilot", "SMOKE.json").unlink(missing_ok=True)
    r = run("gen", "timing")
    assert r.exit_code == 1 and "smoke test" in r.output and stage4["api"].calls == []


def test_a_batch_whose_results_are_gone_is_abandoned_by_the_developer(stage4, monkeypatch) -> None:
    assert run("gen", "plan", "--set", "pilot").exit_code == 0
    assert run("gen", "smoke").exit_code == 0
    assert run("gen", "submit", "--set", "pilot").exit_code == 0
    with stage4["ledger"]() as ledger:
        group = next(g for g, info in ledger.state.groups.items() if info["mode"] == "batch")
    stage4["api"].batch_retrieve = lambda batch_id: (_ for _ in ()).throw(llm.ApiError("client", 404))
    r = run("gen", "collect", "--set", "pilot")
    assert r.exit_code == 1 and "batch abandon" in r.output  # reported, never silently skipped
    monkeypatch.setenv("CLAUDECODE", "1")
    assert run("batch", "abandon", group, "--reason", "deleted in the Console").exit_code == 1  # typed only
    monkeypatch.setattr(cli4, "typed_by_developer", lambda expected, prompt: None)
    r = run("batch", "abandon", group, "--reason", "deleted in the Console")
    assert r.exit_code == 0, r.output
    with stage4["ledger"]() as ledger:
        items = [i for (g, _), i in ledger.state.items.items() if g == group]
        assert all(i.status == "unknown" and not ledger.state.blocking(i.custom_id) for i in items)
        assert cli4.phase_cap("judging", ledger) < Decimal("25.00")  # counted at their bounds, not blocking


def test_batches_interleave_in_sha256_order_of_custom_id(stage4, monkeypatch) -> None:
    import hashlib

    real = cli4.generation_requests

    def more(which):  # six answer lines, so the order is a real test
        out = real(which)
        extra = []
        for n in range(4):
            line, body, docs = out[0]
            body = {**body, "messages": [{"role": "user", "content": body["messages"][0]["content"][:-1]
                                          + [{"type": "text", "text": f"extra question {n}?"}]}]}
            extra.append((line_for(line.condition, f"x{n}", 0, body), body, docs))
        return out + extra

    monkeypatch.setattr(cli4, "generation_requests", more)
    assert run("gen", "plan", "--set", "pilot").exit_code == 0
    assert run("gen", "smoke").exit_code == 0
    planned = [l.custom_id for l in read_plan(stage4["tmp"] / "pilot" / "requests.jsonl") if l.mode == "batch"]
    by_sha = sorted(planned, key=lambda c: hashlib.sha256(c.encode()).hexdigest())
    assert planned != by_sha  # the plan's order is not already the batch order
    assert run("gen", "submit", "--set", "pilot").exit_code == 0
    (created,) = [c for c in stage4["api"].calls if c[0] == "batch_create"]
    assert created[1] == by_sha


def _one_open_batch(stage4):
    assert run("gen", "plan", "--set", "pilot").exit_code == 0
    assert run("gen", "smoke").exit_code == 0
    assert run("gen", "submit", "--set", "pilot").exit_code == 0
    with stage4["ledger"]() as ledger:
        return next(g for g, info in ledger.state.groups.items() if info["mode"] == "batch")


def test_abandon_refuses_a_batch_that_is_running_or_whose_results_can_still_be_read(stage4, monkeypatch) -> None:
    group = _one_open_batch(stage4)
    monkeypatch.setattr(cli4, "typed_by_developer", lambda expected, prompt: None)
    stage4["api"].polls_until_ended = 99  # still in progress
    r = run("batch", "abandon", group, "--reason", "impatient")
    assert r.exit_code == 1 and "let it end" in r.output
    stage4["api"].polls_until_ended = 1
    r = run("batch", "abandon", group, "--reason", "wrong group")
    assert r.exit_code == 1 and "can still be fetched" in r.output  # collect it instead: never paid twice
    stage4["api"].batch_retrieve = lambda batch_id: (_ for _ in ()).throw(llm.ApiError("server", 502))
    r = run("gen", "collect", "--set", "pilot")
    assert r.exit_code == 1 and "run collect again" in r.output and "abandon" not in r.output  # transient


def test_abandon_refuses_while_a_response_is_in_the_store_and_finishes_when_interrupted(stage4, monkeypatch) -> None:
    group = _one_open_batch(stage4)
    stage4["api"].batch_retrieve = lambda batch_id: (_ for _ in ()).throw(llm.ApiError("client", 404))
    with stage4["ledger"]() as ledger:
        ids = [c for (g, c) in ledger.state.items if g == group]
    monkeypatch.setattr(cli4, "typed_by_developer", lambda expected, prompt: None)
    # A response stored before a crash stopped its settlement: abandoning would lose it, so it is refused.
    from raglab.store import Record

    planned = {l.custom_id: l for l in read_plan(stage4["tmp"] / "pilot" / "requests.jsonl")}
    line = planned[ids[0]]
    Store(stage4["store"].path).add(Record(line.custom_id, line.request_key, line.condition, line.body_sha256,
                                           "pilot", "succeeded", fake_message("x"), None, "msgbatch_x", None, "t",
                                           group))
    r = run("batch", "abandon", group, "--reason", "deleted in the Console")
    assert r.exit_code == 1 and "in the store" in r.output
    path = stage4["store"].path
    path.write_text("".join(l + "\n" for l in path.read_text().splitlines() if ids[0] not in l))
    from raglab.ledger import Ledger

    real_resend = Ledger.resend
    calls = {"n": 0}

    def flaky(self, g, cid, reason):
        calls["n"] += 1
        if calls["n"] == 2:
            raise KeyboardInterrupt
        return real_resend(self, g, cid, reason)

    monkeypatch.setattr(Ledger, "resend", flaky)
    r = run("batch", "abandon", group, "--reason", "deleted in the Console")
    assert r.exit_code != 0  # interrupted between the two passes
    monkeypatch.setattr(Ledger, "resend", real_resend)
    r = run("batch", "abandon", group, "--reason", "deleted in the Console")
    assert r.exit_code == 0, r.output
    with stage4["ledger"]() as ledger:
        assert all(ledger.state.items[(group, c)].status == "unknown" and not ledger.state.blocking(c) for c in ids)


def test_a_lost_response_is_reported_with_its_remedy_never_sent_again(stage4) -> None:
    group = _one_open_batch(stage4)
    assert run("gen", "collect", "--set", "pilot").exit_code == 0
    path = stage4["store"].path
    path.write_text("".join(l + "\n" for l in path.read_text().splitlines() if '"batch_id":null' in l))
    before = len([c for c in stage4["api"].calls if c[0] == "batch_create"])
    r = run("gen", "submit", "--set", "pilot")
    assert r.exit_code == 0 and "--refetch" in r.output
    assert len([c for c in stage4["api"].calls if c[0] == "batch_create"]) == before  # never paid twice


def test_an_interrupted_standard_send_names_batch_orphan(stage4) -> None:
    from raglab.ledger import Planned

    cid = "r" + "8" * 56 + "s00"
    with stage4["ledger"]() as ledger:
        ledger.reserve("pilot", "standard", [Planned(cid, "claude-haiku-5-5", 10, 10, False)], "a" * 64)
        with pytest.raises(cli4.LedgerError, match="batch orphan"):
            ledger.reserve("pilot", "standard", [Planned(cid, "claude-haiku-5-5", 10, 10, False)], "a" * 64)
