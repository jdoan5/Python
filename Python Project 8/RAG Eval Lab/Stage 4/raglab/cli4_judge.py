"""Stage 4's judge commands for the pilot (PLAN.md, "The pilot", "When a judge fails"). Registered by cli4.

  judge candidates --set pilot   the near-miss swap candidates, from the stored pilot answers (no API)
  judge plan --set pilot         pilot/judge-requests.jsonl: every judge request the pilot may send (no API)
  judge submit --set pilot --item <item>   send what an item needs next (billed: pilot-judging)
  judge collect --set pilot      store and settle ended judge batches (spends nothing)
  judge status --set pilot       where every judgment stands (reads only)

Every billed step runs cli4's checks (freeze point, Keychain, signed
authorization, the gate before every reservation). Items go in PLAN.md's
order: the smoke test's three calls must all give a verdict before any other
item is sent, and the swaps wait for F's medium verdicts on their originals.
"""

from __future__ import annotations

import json
from collections import Counter
from functools import lru_cache
from pathlib import Path

import typer
from rich.markup import escape
from rich.table import Table

from raglab import cli4, params

judge_app = typer.Typer(help="Plan, submit and collect the judges' requests (Claude Opus 5.5).", no_args_is_help=True)

PILOT = cli4.STAGE / "pilot"
SWAP_ITEMS = ("pilot-swaps-medium", "pilot-swaps-low")


def candidates_path() -> Path:
    return PILOT / "candidates.json"


def checks_path() -> Path:
    return PILOT / "checks.json"


def _pilot_only(which: str) -> None:
    if which != "pilot":
        cli4._fail("only --set pilot exists until freeze point 4's golden judge plan is built")


@lru_cache(maxsize=1)
def _corpus():
    return cli4._corpus()


def pilot_world():
    """The stored pilot answers parsed against their windows, the questions with their keys, and the labels."""
    from raglab.judgeplan import graded_questions, stored_answers
    from raglab.sets import pilot_questions
    from raglab.store import Store

    questions = pilot_questions()
    _, by_id, index = _corpus()
    built = {line.custom_id: (line, body, shown) for line, body, shown in cli4.generation_requests("pilot")}
    stored = stored_answers(Store(cli4.api_dir() / "store" / "pilot.jsonl"), built)
    keys = json.loads((PILOT / "keys.json").read_text(encoding="utf-8"))["keys"]
    labels = json.loads((PILOT / "labels.json").read_text(encoding="utf-8"))["labels"]
    return questions, by_id, index, stored, graded_questions(questions, keys, by_id, index), labels


def _candidates(world):
    from raglab.judgeplan import pilot_candidates

    questions, by_id, index, stored, _, labels = world
    return pilot_candidates(stored, questions, labels, by_id, index)


def _confirmed(world=None) -> dict[str, bool]:
    """The worksheet's confirmations, each checked against the passage it confirmed: a candidate whose rebuilt
    block isn't the one the worksheet saw is refused, never silently treated as confirmed."""
    import hashlib

    if not checks_path().exists():
        raise ValueError("pilot/checks.json is missing: confirm the swap candidates by worksheet first "
                         "(analysis/v3_annotations.py pilot-swaps, then pilot-merge-swaps)")
    checks = json.loads(checks_path().read_text(encoding="utf-8"))
    seen = checks.get("passage_sha256", {})
    for c in _candidates(world or pilot_world()):
        key = f"{c.answered}:{c.name}"
        if key in checks["confirmed"] and c.needs_confirmation:
            now = hashlib.sha256("\n\n".join(c.texts).encode("utf-8")).hexdigest()
            if seen.get(key) != now:
                raise ValueError(f"{key}: its passage isn't the one the worksheet confirmed; build the swap "
                                 "worksheet again")
    return checks["confirmed"]


def rebuilt_list(world=None):
    """The pilot's judge requests, (line, body), rebuilt from the store and the code."""
    from raglab.judgeplan import pilot_judge_list

    world = world or pilot_world()
    questions, _, _, stored, graded, _ = world
    return pilot_judge_list(stored, questions, graded, _candidates(world), _confirmed(world))


@judge_app.command("candidates")
def judge_candidates(which: str = typer.Option("pilot", "--set")) -> None:
    """Write pilot/candidates.json: every near-miss swap candidate, by corpus position (no documentation text)."""
    from raglab.judgeplan import write_candidates

    _pilot_only(which)
    try:
        cands = _candidates(pilot_world())
    except (ValueError, OSError, KeyError) as e:
        cli4._fail(str(e))
    write_candidates(candidates_path(), cands)
    by = Counter((c.kind, c.arm) for c in cands)
    cli4.console.print(f"{len(cands)} candidates: " + ", ".join(f"{k} in {a} {n}" for (k, a), n in sorted(by.items())))


@judge_app.command("plan")
def judge_plan(which: str = typer.Option("pilot", "--set")) -> None:
    """Write pilot/judge-requests.jsonl from the stored answers: every request, its retry and truncation re-run."""
    from raglab.judging import write_judge_list

    _pilot_only(which)
    try:
        items = rebuilt_list()
        write_judge_list(cli4.REQUEST_LISTS["pilot-judging"], [jl for jl, _ in items])
    except (ValueError, OSError, KeyError) as e:
        cli4._fail(str(e))
    primaries = Counter(jl.item for jl, _ in items if jl.role == "primary")
    cli4.console.print(f"{len(items)} judge requests ({sum(primaries.values())} primaries): "
                       + ", ".join(f"{i} {n}" for i, n in primaries.items()))


def _validator(jl, world, cands):
    from raglab.judge import check_correctness, check_support
    from raglab.planted import planted_excerpts

    questions, _, _, stored, graded, _ = world
    s = stored[jl.answered]
    if jl.judge == "F":
        planted = planted_excerpts(cands[(jl.answered, jl.planted)]) if jl.planted else None
        return lambda out: check_support(out, s.answer, s.docs, planted)
    g = graded[s.line.question]
    return lambda out: check_correctness(out, g.key, g.reference)


def judgments(lines, world, store, failures):
    """Every judgment in the list, resolved against the store and the ledger's API errors."""
    from raglab.verdicts import groups, resolve

    cands = {(c.answered, c.name): c for c in _candidates(world)}
    return {key: resolve(roles, store, failures, _validator(roles["primary"], world, cands))
            for key, roles in groups(lines).items()}


def originals_supported(all_judgments) -> dict[tuple[str, str], bool] | None:
    """F's medium verdicts on the swaps' originals (hhr run 1 and the baseline), unit by unit;
    None until every one of them is settled."""
    out = {}
    for (judge, answered, item, planted, _), j in all_judgments.items():
        if judge != "F" or item != "pilot-medium" or planted:
            continue
        if j.state in ("send",):
            return None
        if j.state == "verdict":
            for u in j.verdict.units:
                out[(answered, u.id)] = u.supported
    return out


def gated(lines, all_judgments, world):
    """The swap lines the gate passes: per kind and arm, the first confirmed candidates whose original was
    supported (3 lexical and 3 topical per arm)."""
    from raglab.planted import SWAPS, gate

    supported = originals_supported(all_judgments)
    if supported is None:
        raise ValueError("the swaps wait for F's medium verdicts on their originals: finish pilot-medium first")
    from raglab.judgeplan import PILOT_ARMS, PILOT_SWAPS

    needed = {(k, arm): PILOT_SWAPS[k] for k in SWAPS for arm, _ in PILOT_ARMS}
    chosen = gate(_candidates(world), _confirmed(world), supported, needed)
    names = {(c.answered, c.name) for c in chosen}
    return [jl for jl in lines if jl.item not in SWAP_ITEMS or (jl.answered, jl.planted) in names]


@judge_app.command("submit")
def judge_submit(which: str = typer.Option("pilot", "--set"),
                 item: str = typer.Option(..., help="which item to send, in PLAN.md's order")) -> None:
    """Send what an item needs next: primaries not yet answered, then any retry, truncation re-run or
    resubmission its results call for. Billed: needs the pilot-judging authorization."""
    from raglab.judgeplan import PILOT_ITEMS
    from raglab.judging import read_judge_list
    from raglab.ledger import CapExceeded, LedgerError
    from raglab.plan4 import as_request, count_all
    from raglab.runner import RunStopped, send_standard, submit_batch
    from raglab.verdicts import failures_from

    _pilot_only(which)
    if item not in PILOT_ITEMS:
        cli4._fail(f"--item must be one of {', '.join(PILOT_ITEMS)}")
    try:
        planned = read_judge_list(cli4.REQUEST_LISTS["pilot-judging"])
        world = pilot_world()
        bodies = {jl.line.custom_id: body for jl, body in rebuilt_list(world)}
        stray = [jl.line.custom_id for jl in planned if jl.line.custom_id not in bodies]
        if stray:
            raise ValueError(f"{len(stray)} listed judge request(s) can't be rebuilt, e.g. {stray[0]}")
        gate, transport, store = cli4._live("pilot-judging", cli4.REQUEST_LISTS["pilot-judging"],
                                            frozenset(jl.line.custom_id for jl in planned))
        with cli4.open_ledger() as ledger:
            all_j = judgments(planned, world, store, failures_from(ledger.state.items))
            if item != "smoke":
                smoke = [j for k, j in all_j.items() if k[2] == "smoke"]
                if not smoke or any(j.state != "verdict" for j in smoke):
                    raise ValueError("the smoke test's judge calls must all give a verdict first "
                                     "(`judge submit --item smoke`, then `judge status`)")
            lines = [jl for jl in planned if jl.item == item]
            if item in SWAP_ITEMS:
                lines = gated(lines, all_j, world)
            todo = [j.next for k, j in all_j.items() if k[2] == item and j.state == "send"
                    and j.next in lines and j.next.line.custom_id not in store
                    and not ledger.state.blocking(j.next.line.custom_id)]
            if not todo:
                cli4.console.print(f"nothing to send for {item}")
                return
            counted = {l.custom_id: l for l in count_all([jl.line for jl in todo],
                                                         {jl.line.custom_id: bodies[jl.line.custom_id] for jl in todo},
                                                         transport.count_tokens, ledger.pricing)}
            reqs = sorted((as_request(counted[jl.line.custom_id], bodies[jl.line.custom_id]) for jl in todo),
                          key=lambda r: cli4.interleave_key(r.custom_id))
            if item == "smoke":  # the smoke test runs on the standard API, one call at a time
                for r in reqs:
                    rec = send_standard(ledger, store, transport, "pilot", r, gate)
                    cli4.console.print(f"{r.condition[:40]}: {rec.outcome}")
                return
            size = params.BATCH_MAX[params.JUDGE]
            for start in range(0, len(reqs), size):
                try:
                    group = submit_batch(ledger, store, transport, "pilot", reqs[start:start + size], gate)
                except CapExceeded as e:
                    cli4.console.print(f"[yellow]stopping for now:[/yellow] {escape(str(e))}")
                    break
                cli4.console.print(f"submitted {group}: {len(reqs[start:start + size])} judge requests")
    except (RunStopped, LedgerError, ValueError, OSError, KeyError) as e:
        cli4._fail(str(e))
    except Exception as e:  # noqa: BLE001 - FreezeError, AuthorizationError, LlmError: all stop with their message
        cli4._fail(f"{type(e).__name__}: {e}")


@judge_app.command("collect")
def judge_collect(which: str = typer.Option("pilot", "--set"),
                  refetch: str = typer.Option("", help="a settled judge batch group whose stored outputs were lost")
                  ) -> None:
    """Store and settle every ended pilot judge batch. Spends nothing; safe to rerun."""
    from raglab.judging import read_judge_list

    _pilot_only(which)
    try:
        planned = {jl.line.custom_id: jl.line for jl in read_judge_list(cli4.REQUEST_LISTS["pilot-judging"])}
        bodies = {jl.line.custom_id: body for jl, body in rebuilt_list()}
        if refetch:
            cli4.restore_group("pilot", refetch, planned, bodies)
            return
        cli4.collect_groups("pilot", planned, bodies, only=set(planned))
    except Exception as e:  # noqa: BLE001
        cli4._fail(f"{type(e).__name__}: {e}")


@judge_app.command("status")
def judge_status(which: str = typer.Option("pilot", "--set")) -> None:
    """Where every pilot judgment stands, by item: verdicts, waiting to be sent, unjudged. Reads only."""
    from raglab.judging import read_judge_list
    from raglab.store import Store
    from raglab.verdicts import failures_from

    _pilot_only(which)
    try:
        planned = read_judge_list(cli4.REQUEST_LISTS["pilot-judging"])
        world = pilot_world()
        store = Store(cli4.api_dir() / "store" / "pilot.jsonl")
        with cli4.open_ledger() as ledger:
            all_j = judgments(planned, world, store, failures_from(ledger.state.items))
    except Exception as e:  # noqa: BLE001
        cli4._fail(f"{type(e).__name__}: {e}")
    table = Table(title="Pilot judgments")
    for col in ("item", "verdict", "to send", "unjudged"):
        table.add_column(col)
    by = {}
    for (_, _, item, _, _), j in all_j.items():
        by.setdefault(item, Counter())[j.state] += 1
    for item, c in by.items():
        table.add_row(item, str(c["verdict"]), str(c["send"]), str(c["unjudged"]))
    cli4.console.print(table)


# ---------------------------------------------------------------------------------------------------- pilot report

pilot_app = typer.Typer(help="The pilot's rules 1-7, applied before freeze point 2.", no_args_is_help=True)


def fp1_commit() -> str:
    """Freeze point 1's commit: the one the pilot's first authorization was given at."""
    path = cli4.api_dir() / "authorization.json"
    records = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    pilot = [r for r in records if r["phase"] == "pilot"]
    if not pilot:
        raise ValueError("no pilot authorization is recorded, so freeze point 1's commit is unknown")
    return pilot[0]["commit"]


@pilot_app.command("report")
def pilot_report_cmd() -> None:
    """Apply rules 1-7 to the stored pilot answers and verdicts; write pilot/REPORT.json and REPORT.md. No API."""
    from raglab import pilot_report
    from raglab.judgeplan import PILOT_ARMS, PILOT_SWAPS
    from raglab.judging import read_judge_list
    from raglab.planted import SWAPS, gate
    from raglab.plan4 import read_plan
    from raglab.store import Store
    from raglab.verdicts import failures_from
    from raglab.windows import Windows

    try:
        world = pilot_world()
        questions, by_id, index, stored, graded, _ = world
        planned = read_judge_list(cli4.REQUEST_LISTS["pilot-judging"])
        store = Store(cli4.api_dir() / "store" / "pilot.jsonl")
        with cli4.open_ledger() as ledger:
            failures = failures_from(ledger.state.items)
            usage = {e["custom_id"]: e["usage"] for e in ledger.events()
                     if e["event"] == "settle" and e["outcome"] == "succeeded"}
        all_j = judgments(planned, world, store, failures)
        supported = originals_supported(all_j)
        if supported is None:
            raise ValueError("pilot-medium isn't finished, so the swaps' gate can't be read")
        needed = {(k, arm): PILOT_SWAPS[k] for k in SWAPS for arm, _ in PILOT_ARMS}
        chosen = gate(_candidates(world), _confirmed(world), supported, needed)
        names = {(c.answered, c.name) for c in chosen}
        pending = [k for k, j in all_j.items() if j.state == "send" and (k[2] not in SWAP_ITEMS or (k[1], k[3]) in names)]
        if pending:
            raise ValueError(f"{len(pending)} pilot judgment(s) still need a request: finish judging first "
                             "(`judge status --set pilot`)")
        smoke_path = cli4.SMOKE_REPORT
        smoke = json.loads(smoke_path.read_text(encoding="utf-8"))["verdict"] if smoke_path.exists() else {}
        smoke = {**smoke, "judges": {f"{k[0]} sample {k[4]}": j.state for k, j in all_j.items() if k[2] == "smoke"}}
        generation = [l.custom_id for l in read_plan(cli4.REQUEST_LISTS["pilot"]) if l.role == "answer"]
        kinds = {jl.line.custom_id: f"{jl.judge}-{'low' if jl.item.endswith('low') else 'medium'}" for jl in planned}
        docs, _, _ = _corpus()
        fix_path = PILOT / "splitter-fix.json"  # rule 7's one allowed fix, if it was made: its committed record
        fix = json.loads(fix_path.read_text(encoding="utf-8")) if fix_path.exists() else None
        if fix is not None and set(fix) != set(pilot_report.SPLITTER_FIX_FIELDS):
            raise ValueError(f"pilot/splitter-fix.json needs exactly {', '.join(pilot_report.SPLITTER_FIX_FIELDS)}")
        report = pilot_report.assemble(
            stored=stored, questions=questions, graded=graded, judgments=all_j, gated=chosen,
            generation_ids=generation, judge_kinds=kinds, settled_usage=usage, smoke=smoke, fp1_commit=fp1_commit(),
            answer_v2_exists=(cli4.STAGE / "prompts" / "answer-v2.md").exists(), windows=Windows(docs, index),
            index=index, splitter_fix=fix)
        pilot_report.write(report, PILOT)
    except (ValueError, OSError, KeyError) as e:
        cli4._fail(str(e))
    cli4.console.print(f"pilot/REPORT.json and REPORT.md written; low effort {'passed' if report['low_passed'] else 'failed'}"
                       f"; prompt {report['answer_prompt']}" + ("; [red]the stage is stopped[/red]" if report["stop"] else ""))
