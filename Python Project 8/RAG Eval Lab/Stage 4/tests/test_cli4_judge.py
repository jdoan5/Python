"""The pilot's judge commands end to end on the fake API: the order, the gate, retries. No network, no key."""

from __future__ import annotations

import json
import re
from decimal import Decimal
from pathlib import Path

import pytest
from rich.console import Console
from typer.testing import CliRunner

from raglab import cli, cli4, cli4_judge, llm
from raglab.judgeplan import graded_questions
from raglab.judging import read_judge_list
from raglab.ledger import Caps, Ledger, Pricing
from raglab.llm import FakeTransport, fake_message
from raglab.runner import Gate
from raglab.store import Store
from tests.test_judgeplan import build
from tests.test_judgeplan import world as judge_world  # noqa: F401 - the fixture
from tests.test_planted import GRADES

STAGE = Path(__file__).resolve().parents[1]
runner = CliRunner()
_UNIT = re.compile(r"^(U\d+): ", re.M)
_EXCERPT = re.compile(r"^  (U\d+\.E\d+): (.*)$", re.M)
_NUGGET = re.compile(r"^(N\d+) \((?:essential|supporting)\)", re.M)


def judge(body, broken: set[str] | None = None) -> dict:
    """A fake Opus: F supports every unit with a real quote; C finds every nugget conveyed."""
    text = body["messages"][0]["content"][0]["text"]
    if "<units>" in text:
        excerpts = dict(_EXCERPT.findall(text))
        units = []
        for uid in _UNIT.findall(text.split("<units>", 1)[1]):
            mine = [k for k in excerpts if k.startswith(uid + ".")]
            units.append({"id": uid, "claim": True, "support": "supported" if mine else "uncited",
                          "quotes": [{"excerpt": mine[0], "quote": excerpts[mine[0]][:20]}] if mine else [],
                          "reason": "stated"})
        out = {"units": units}
    else:
        out = {"nuggets": [{"id": n, "conveyed": "yes", "answer_quote": ""} for n in _NUGGET.findall(text)],
               "contradiction": {"present": False, "reference_quote": "", "answer_quote": ""},
               "abstention": "none", "resolves": "yes", "reason": "conveyed"}
    if broken and body["output_config"]["effort"] in broken:
        out = {"units": []}  # wrong unit ids: invalid output
    return fake_message(json.dumps(out), model="claude-opus-5-5")


@pytest.fixture
def env(judge_world, tmp_path: Path, monkeypatch):  # noqa: F811
    doc, index, by_id, docs, questions, keys = judge_world
    monkeypatch.setattr(cli4, "console", Console(width=250))
    store = Store(cli4.api_dir() / "store" / "pilot.jsonl")
    stored = build(judge_world)
    graded = graded_questions(questions, keys, by_id, index)
    world = (questions, by_id, index, stored, graded, {"p01": GRADES})
    monkeypatch.setattr(cli4_judge, "pilot_world", lambda: world)
    monkeypatch.setattr(cli4_judge, "_confirmed",
                        lambda w=None: {f"{c.answered}:{c.name}": True for c in cli4_judge._candidates(world)})
    lists = {**cli4.REQUEST_LISTS, "pilot-judging": tmp_path / "pilot" / "judge-requests.jsonl"}
    monkeypatch.setattr(cli4, "REQUEST_LISTS", lists)
    api = FakeTransport(judge)
    caps = Caps(total=Decimal("25.00"), phases={"pilot": Decimal("6.50"), "generation": Decimal("3.00"),
                                                "judging": Decimal("25.00")})
    pricing = Pricing.load(STAGE / "pricing.json")
    monkeypatch.setattr(cli4, "open_ledger", lambda repair=False: Ledger(cli4.api_dir() / "ledger.jsonl", pricing, caps))
    monkeypatch.setattr(cli4, "_live", lambda phase, path, keys: (Gate(lambda: "a" * 64, keys), api, Store(store.path)))
    monkeypatch.setattr(llm, "AnthropicTransport", lambda key: api)
    monkeypatch.setattr(llm, "keychain_key", lambda: "not-a-real-key")
    return {"api": api, "store": store, "lists": lists}


def run(*args: str):
    return runner.invoke(cli.app, list(args))


def sent(api, kind: str) -> list:
    return [c for c in api.calls if c[0] == kind]


def test_the_pilot_judges_in_order_smoke_first_then_batches(env) -> None:
    r = run("judge", "plan", "--set", "pilot")
    assert r.exit_code == 0, r.output
    planned = read_judge_list(env["lists"]["pilot-judging"])
    assert planned[0].item == "smoke"
    r = run("judge", "submit", "--set", "pilot", "--item", "pilot-medium")
    assert r.exit_code == 1 and "smoke" in r.output and not sent(env["api"], "batch_create")
    r = run("judge", "submit", "--set", "pilot", "--item", "smoke")
    assert r.exit_code == 0, r.output
    assert len(sent(env["api"], "create")) == 3  # two identical F calls and one C call, standard API
    r = run("judge", "submit", "--set", "pilot", "--item", "pilot-medium")
    assert r.exit_code == 0, r.output
    batches = sent(env["api"], "batch_create")
    assert batches and all(len(ids) <= 50 for _, ids in batches)
    assert run("judge", "collect", "--set", "pilot").exit_code == 0
    r = run("judge", "status", "--set", "pilot")
    assert r.exit_code == 0 and "pilot-medium" in r.output
    assert "nothing to send" in run("judge", "submit", "--set", "pilot", "--item", "pilot-medium").output


def test_swaps_wait_for_the_originals_and_take_only_what_the_gate_passes(env) -> None:
    assert run("judge", "plan", "--set", "pilot").exit_code == 0
    assert run("judge", "submit", "--set", "pilot", "--item", "smoke").exit_code == 0
    r = run("judge", "submit", "--set", "pilot", "--item", "pilot-swaps-medium")
    assert r.exit_code == 1 and "originals" in r.output
    assert run("judge", "submit", "--set", "pilot", "--item", "pilot-medium").exit_code == 0
    assert run("judge", "collect", "--set", "pilot").exit_code == 0
    before = len(sent(env["api"], "batch_create"))
    r = run("judge", "submit", "--set", "pilot", "--item", "pilot-swaps-medium")
    assert r.exit_code == 0, r.output
    swaps = sent(env["api"], "batch_create")[before:]
    planted = [jl for jl in read_judge_list(env["lists"]["pilot-judging"]) if jl.item == "pilot-swaps-medium"
               and jl.role == "primary"]
    assert swaps and len(swaps[0][1]) <= len(planted)


def test_an_invalid_output_sends_the_listed_retry_once(env) -> None:
    env["api"].responder = lambda body: judge(body, broken={"low"})
    assert run("judge", "plan", "--set", "pilot").exit_code == 0
    assert run("judge", "submit", "--set", "pilot", "--item", "smoke").exit_code == 0
    assert run("judge", "submit", "--set", "pilot", "--item", "pilot-low").exit_code == 0
    assert run("judge", "collect", "--set", "pilot").exit_code == 0
    first = sent(env["api"], "batch_create")[-1][1]
    assert run("judge", "submit", "--set", "pilot", "--item", "pilot-low").exit_code == 0
    retries = sent(env["api"], "batch_create")[-1][1]
    assert set(retries).isdisjoint(first)  # the next sample: new custom_ids, never the same request again
    planned = {jl.line.custom_id: jl for jl in read_judge_list(env["lists"]["pilot-judging"])}
    assert {planned[c].role for c in retries} == {"retry"}
    assert run("judge", "collect", "--set", "pilot").exit_code == 0
    r = run("judge", "submit", "--set", "pilot", "--item", "pilot-low")
    assert "nothing to send" in r.output  # a second failure: unjudged, never a third request
    assert "pilot-low" in run("judge", "status", "--set", "pilot").output


def test_the_pilot_report_applies_the_rules_to_the_finished_pilot(env, judge_world, monkeypatch, tmp_path) -> None:  # noqa: F811
    from raglab import pilot_report, windows
    from raglab.plan4 import write_plan

    world = cli4_judge.pilot_world()
    stored = world[3]
    gen_list = tmp_path / "pilot" / "requests.jsonl"
    write_plan(gen_list, [s.line for s in stored.values() if not s.line.condition.startswith("smoke")])
    env["lists"]["pilot"] = gen_list
    monkeypatch.setattr(cli4, "SMOKE_REPORT", tmp_path / "pilot" / "SMOKE.json")
    monkeypatch.setattr(cli4_judge, "PILOT", tmp_path / "pilot")
    monkeypatch.setattr(cli4_judge, "_corpus", lambda: ([], world[1], world[2]))

    class Whole:  # every window holds its whole answering section
        def __init__(self, docs, index):
            self.index = index

        def window(self, cond, text, answering, target):
            ref = self.index.get(sorted(answering)[0])
            return [(ref.doc_id, ref.start, ref.end)]

    monkeypatch.setattr(windows, "Windows", Whole)
    monkeypatch.setattr(pilot_report, "power_range", lambda m1: [{"e": 0.03, "gamma": 0.5, "runs": 1, "delta": 0.1,
                                                                 "resolvable": 0.17, "power": 0.7}])
    (cli4.api_dir() / "authorization.json").write_text(json.dumps([{"phase": "pilot", "commit": "f" * 40}]))
    assert run("judge", "plan", "--set", "pilot").exit_code == 0
    r = run("pilot", "report")
    assert r.exit_code == 1 and "finished" in r.output  # judging isn't done yet
    for item in ("smoke", "pilot-medium", "pilot-noise", "pilot-low", "pilot-closed"):
        assert run("judge", "submit", "--set", "pilot", "--item", item).exit_code == 0
        assert run("judge", "collect", "--set", "pilot").exit_code == 0
    for item in ("pilot-swaps-medium", "pilot-swaps-low"):
        assert run("judge", "submit", "--set", "pilot", "--item", item).exit_code == 0
        assert run("judge", "collect", "--set", "pilot").exit_code == 0
    r = run("pilot", "report")
    assert r.exit_code == 0, r.output
    report = json.loads((tmp_path / "pilot" / "REPORT.json").read_text())
    assert report["fp1_commit"] == "f" * 40 and report["answer_prompt"] == "answer-v1.md"
    assert set(report["max_tokens"]) == {"haiku", "judge"} and report["max_tokens"]["judge"]["support"] >= 3000
    assert report["rule4_power"]["rejudge_flip_rate"] == 0.0 and report["rule4_power"]["m1_grid"] == 0.85
    assert set(report["rule5_unit_costs"]) >= {"F-medium", "C-medium", "F-low", "C-low"}
    assert not report["rule6_citation_failures"] and not report["stop"]
    md = (tmp_path / "pilot" / "REPORT.md").read_text()
    assert "Rule 3: low judge effort" in md and "f" * 40 in md and "No fix." in md
    # Rule 7's one fix, when made, is read from its committed record and written into the report.
    fix = {"rule": "a list number after a bullet no longer ends a unit", "answers": ["p03 hhr run 1"],
           "reason": "it split one step in two", "commit": "e" * 40, "date": "2026-11-23"}
    (tmp_path / "pilot" / "splitter-fix.json").write_text(json.dumps(fix))
    assert run("pilot", "report").exit_code == 0
    assert json.loads((tmp_path / "pilot" / "REPORT.json").read_text())["rule7_splitter_fix"] == fix
    assert "Fixed once (2026-11-23" in (tmp_path / "pilot" / "REPORT.md").read_text()
    (tmp_path / "pilot" / "splitter-fix.json").write_text(json.dumps({"rule": "x"}))
    assert run("pilot", "report").exit_code == 1  # an incomplete record is refused


def test_a_confirmation_holds_only_for_the_passage_the_worksheet_saw(judge_world, monkeypatch, tmp_path) -> None:  # noqa: F811
    import hashlib

    doc, index, by_id, docs, questions, keys = judge_world
    stored = build(judge_world)
    world = (questions, by_id, index, stored, graded_questions(questions, keys, by_id, index), {"p01": GRADES})
    cands = cli4_judge._candidates(world)
    good = {f"{c.answered}:{c.name}": hashlib.sha256("\n\n".join(c.texts).encode()).hexdigest() for c in cands}
    checks = tmp_path / "checks.json"
    monkeypatch.setattr(cli4_judge, "checks_path", lambda: checks)
    checks.write_text(json.dumps({"confirmed": {k: True for k in good}, "passage_sha256": good}))
    assert cli4_judge._confirmed(world) == {k: True for k in good}
    first = next(iter(good))
    checks.write_text(json.dumps({"confirmed": {k: True for k in good}, "passage_sha256": {**good, first: "0" * 64}}))
    with pytest.raises(ValueError, match="isn't the one the worksheet confirmed"):
        cli4_judge._confirmed(world)


def test_a_lost_judge_record_is_restored_from_its_batch(env) -> None:
    assert run("judge", "plan", "--set", "pilot").exit_code == 0
    assert run("judge", "submit", "--set", "pilot", "--item", "smoke").exit_code == 0
    assert run("judge", "submit", "--set", "pilot", "--item", "pilot-low").exit_code == 0
    assert run("judge", "collect", "--set", "pilot").exit_code == 0
    path = env["store"].path
    before = len(Store(path))
    lines = path.read_text().splitlines()
    kept = [l for l in lines if '"batch_id":null' in l or '"condition":"F:' not in l]  # drop the batched F records
    path.write_text("\n".join(kept) + "\n")
    lost = before - len(Store(path))
    assert lost > 0
    with cli4.open_ledger() as ledger:
        group = next(g for g, info in ledger.state.groups.items()
                     if info["mode"] == "batch" and (info["sent"] or {}).get("batch_id"))
    r = run("judge", "collect", "--set", "pilot", "--refetch", group)
    assert r.exit_code == 0 and f"{lost} lost record(s) restored" in r.output
    assert len(Store(path)) == before


def test_judge_batches_interleave_in_sha256_order_of_custom_id(env) -> None:
    import hashlib

    assert run("judge", "plan", "--set", "pilot").exit_code == 0
    assert run("judge", "submit", "--set", "pilot", "--item", "smoke").exit_code == 0
    assert run("judge", "submit", "--set", "pilot", "--item", "pilot-medium").exit_code == 0
    (created, *_) = [c for c in env["api"].calls if c[0] == "batch_create"]
    ids = created[1]
    assert len(ids) >= 3 and ids == sorted(ids, key=lambda c: hashlib.sha256(c.encode()).hexdigest())
    planned = [jl.line.custom_id for jl in read_judge_list(env["lists"]["pilot-judging"])
               if jl.item == "pilot-medium" and jl.role == "primary"]
    assert [c for c in planned if c in ids] != ids  # the list's own order is not the batch order
