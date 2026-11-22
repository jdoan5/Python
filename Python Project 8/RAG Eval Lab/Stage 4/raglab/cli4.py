"""Stage 4's commands (PLAN.md, "Cost"). Registered on Stage 3's command line by cli.py.

Every command that spends money runs the same checks in the same order: the
freeze point it needs is on origin/main; the API key is read from the Keychain
once (a macOS dialog the developer answers); the phase's authorization still
holds and is signed with that key; and, before every reservation, the
authorization is checked again and the request must be on its authorized list.
Collecting results spends nothing and needs no authorization.
"""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from raglab import params
from raglab.authorize import (
    LEDGER_PHASE, PHASES, AuthorizationError, authorize, record_sha256, require, typed_by_developer,
)
from raglab.corpus import project_root
from raglab.ledger import Caps, Ledger, LedgerError, Pricing, default_mirror
from raglab.models import ModelError, ensure_not_synced

STAGE = Path(__file__).resolve().parents[1]
console = Console()
budget_app = typer.Typer(help="Stage 4's spend guard: what has been spent, and the typed authorizations.",
                         no_args_is_help=True)

REQUEST_LISTS = {"pilot": STAGE / "pilot" / "requests.jsonl", "pilot-judging": STAGE / "pilot" / "judge-requests.jsonl",
                 "generation": STAGE / "plan" / "requests.jsonl", "judging": STAGE / "plan" / "judge-requests.jsonl"}
FREEZE_FOR = {"pilot": 1, "pilot-judging": 1, "generation": 2, "judging": 4}


def api_dir() -> Path:
    path = project_root() / "data" / "api"
    try:
        ensure_not_synced(path.parent if path.parent.exists() else project_root() / "data", "API ledger")
    except ModelError as e:
        raise LedgerError(str(e)) from None
    return path


def caps() -> Caps:
    phases = dict(params.PHASE_CAPS)
    phases["judging"] = params.CAP  # judging may use whatever the stage cap leaves (PLAN.md, "Phases")
    return Caps(total=params.CAP, phases=phases)


def responses_stored() -> bool:
    """Whether any store under data/api holds a response: then an empty ledger would mean lost spend."""
    folder = api_dir() / "store"
    return folder.is_dir() and any(p.stat().st_size > 0 for p in folder.glob("*.jsonl"))


def open_ledger(repair: bool = False) -> Ledger:
    return Ledger(api_dir() / "ledger.jsonl", Pricing.load(STAGE / "pricing.json"), caps(), mirror=default_mirror(),
                  repair=repair, expect_history=responses_stored())


def _fail(message: str) -> None:
    console.print(f"[red]Error:[/red] {escape(message)}")
    raise typer.Exit(code=1)


def _usd(x: str | Decimal | None) -> str:
    return "-" if x is None else f"${Decimal(x):,.2f}"


@budget_app.command("status")
def budget_status() -> None:
    """Settled spend, open reservations and every phase against its cap. Reads the ledger only."""
    try:
        with open_ledger() as ledger:
            s = ledger.summary()
    except LedgerError as e:
        _fail(str(e))
    table = Table(title=f"Stage 4 spend (cap {_usd(s['cap'])}; prices as of {s['pricing_as_of']})")
    for col in ("phase", "cap", "committed", "settled"):
        table.add_column(col, justify="right" if col != "phase" else "left")
    for phase, row in s["phases"].items():
        table.add_row(phase, _usd(row["cap"]), _usd(row["committed"]), _usd(row["settled"]))
    console.print(table)
    console.print(f"committed {_usd(s['committed'])} (settled {_usd(s['settled'])}), {s['open_reservations']} open "
                  f"reservations, {s['unknown']} unknown")
    for flag in s["flags"]:
        console.print(f"[yellow]flag:[/yellow] {escape(flag)}")


def phase_cap(phase: str, ledger: Ledger) -> Decimal:
    """What a phase may spend when it is authorized (PLAN.md's phase table)."""
    if phase in ("pilot", "pilot-judging"):
        return params.PHASE_CAPS["pilot"] - ledger.state.committed("pilot")
    if phase == "generation":
        return params.PHASE_CAPS["generation"]
    state = ledger.state
    earlier = [i for (g, c), i in state.items.items() if i.phase in ("pilot", "generation")
               and (i.status in ("reserved", "sent") or (i.status == "unknown" and (g, c) not in state.cleared))]
    if earlier:
        raise LedgerError(f"{len(earlier)} pilot or generation request(s) are still open or unknown: collect, "
                          "reconcile or resend them before judging is authorized, so its cap is what is really left "
                          "(a resent unknown stays counted at its bound)")
    return params.CAP - ledger.state.committed()


@budget_app.command("authorize")
def budget_authorize(phase: str = typer.Option(..., help=f"one of {', '.join(PHASES)}")) -> None:
    """Type the phrase that lets a phase spend. Run it yourself, in your own terminal."""
    from raglab.llm import LlmError, keychain_key

    from raglab import freeze

    if phase not in PHASES:
        _fail(f"phase must be one of {', '.join(PHASES)}")
    try:
        freeze.check_remote(STAGE)  # origin/main must be what GitHub holds before anything is approved
        with open_ledger() as ledger:
            cap = phase_cap(phase, ledger)
        record = authorize(phase, cap, STAGE, api_dir() / "authorization.json", REQUEST_LISTS[phase], keychain_key())
    except (LedgerError, AuthorizationError, LlmError, freeze.FreezeError) as e:
        _fail(str(e))
    console.print(f"[green]Authorized[/green] {record['phase']} up to ${record['cap']} at commit "
                  f"{record['commit'][:8]}.")


@budget_app.command("acknowledge")
def budget_acknowledge(us_inference_phase: str = typer.Option("", "--us-inference",
                                                             help="accept US-only inference for this whole phase"),
                       reason: str = typer.Option(..., help="what you checked")) -> None:
    """Accept flagged settlements so sending can go on. Run it yourself, in your own terminal.

    Without --us-inference it accepts every flag shown. With it, it accepts
    US-only inference (billed at 1.1x) for one phase, past and future; the
    scope rule then projects at 1.1x. Other flags (a cost above its bound,
    another model, another tier) are always accepted one by one.
    """
    try:
        with open_ledger() as ledger:
            if us_inference_phase:
                typed_by_developer(f"accept us inference for {us_inference_phase}",
                                   f"Every {us_inference_phase} call will cost 1.1x. To accept, type exactly:")
                ledger.accept_for_phase(us_inference_phase, "us_inference", reason)
                console.print(f"US-only inference accepted for {us_inference_phase}")
                return
            flagged = ledger.state.unacknowledged()
            if not flagged:
                console.print("no flags to acknowledge")
                return
            for i in flagged:
                console.print(f"{i.group} {i.custom_id}: " + escape("; ".join(i.flags)))
            typed_by_developer(f"acknowledge {len(flagged)} flags", "To accept every flag above, type exactly:")
            for i in flagged:
                ledger.acknowledge(i.group, i.custom_id, reason)
    except (LedgerError, AuthorizationError) as e:
        _fail(str(e))
    console.print(f"{len(flagged)} flag(s) acknowledged")


@budget_app.command("repair")
def budget_repair() -> None:
    """Repair the ledger after a crash: drop a torn last line, or restore a missing copy. Shows what it did."""
    try:
        with open_ledger(repair=True) as ledger:
            for line in ledger.repairs or ["nothing to repair"]:
                console.print(escape(line))
            console.print(json.dumps(ledger.summary(), indent=1))
    except LedgerError as e:
        _fail(str(e))


# ---------------------------------------------------------------------------------------------------- generation

gen_app = typer.Typer(help="Plan, submit and collect the generator's requests (Claude Haiku 5.5).", no_args_is_help=True)
SETS = ("pilot", "golden")


def _corpus():
    from raglab.cli import _load
    from raglab.sections import SectionIndex

    _, report = _load()
    docs = list(report.documents)
    return docs, {d.doc_id: d for d in docs}, SectionIndex(docs)


def generation_requests(which: str):
    """Every generation request of a set, as (plan line, body, documents), rebuilt from the corpus and the code."""
    from raglab.blocks import window_documents
    from raglab.generate import generation_body, prompt
    from raglab.plan4 import line_for
    from raglab.sets import PILOT_CONDITIONS, PILOT_RUNS, golden_questions, pilot_questions
    from raglab.windows import CONDITIONS, Windows

    if which == "pilot":
        questions, conditions, runs = pilot_questions(), PILOT_CONDITIONS, PILOT_RUNS
        max_tokens = {c: params.MAX_TOKENS_PILOT["haiku_xhigh_max_batch"] if CONDITIONS[c].effort in ("xhigh", "max")
                      else params.MAX_TOKENS_PILOT["haiku_batch"] for c in conditions}
    else:
        report = STAGE / "pilot" / "REPORT.json"
        if not report.exists():
            raise LedgerError("golden max_tokens come from the pilot report (pilot rule 1); run `pilot report` first")
        fixed = json.loads(report.read_text(encoding="utf-8"))
        questions, conditions, runs = golden_questions(), params.CONDITIONS, params.RUNS
        max_tokens = {c: fixed["max_tokens"]["haiku"][CONDITIONS[c].effort if CONDITIONS[c].thinking == "adaptive"
                                                      else "off"] for c in conditions}
    docs, by_id, index = _corpus()
    windows = Windows(docs, index)
    answer_prompt = "answer-v1.md" if which == "pilot" else fixed["answer_prompt"]  # answer-v2.md after rule 2
    if answer_prompt not in ("answer-v1.md", "answer-v2.md"):
        raise LedgerError(f"pilot/REPORT.json names an unknown answer prompt {answer_prompt!r}")
    answer, closed = prompt(answer_prompt), prompt("answer-closed-v1.md")
    out = []
    for name in conditions:
        cond = CONDITIONS[name]
        for q in questions:
            spans = windows.window(cond, q.text, q.answering, q.target)
            if spans is None:
                continue  # a masked window needs an answering section to mask
            body = generation_body(cond, q.text, window_documents(spans, by_id, index), max_tokens[name], answer,
                                   closed)
            docs_shown = window_documents(spans, by_id, index)
            for sample in range(runs.get(name, 1)):
                out.append((line_for(name, q.id, sample, body), body, docs_shown))
    if which == "pilot":
        out += _pilot_standard_requests(questions, windows, by_id, index, answer, closed)
    return out


def interleave_key(custom_id: str) -> str:
    """Batch order (PLAN.md, "How the batches are made"): sha256 of the custom_id, so arms, conditions and a
    request's samples (hhr's three runs) are spread across batches rather than kept together."""
    import hashlib

    return hashlib.sha256(custom_id.encode("utf-8")).hexdigest()


SMOKE_HUGE_DOCUMENTS = 300  # the smoke test's very large windows: about the size of the baseline's largest


def _pilot_standard_requests(questions, windows, by_id, index, answer, closed):
    """The pilot's standard-API requests: the smoke test's five answers and the 48-call timing probe."""
    from raglab.blocks import window_documents
    from raglab.generate import generation_body
    from raglab.plan4 import line_for
    from raglab.windows import CONDITIONS

    std = params.MAX_TOKENS_PILOT["haiku_standard"]
    out = []
    answerable = [q for q in questions if q.target]
    first = answerable[0]
    for name in ("hhr-4.8k-med", "base-4.8k-med", "closed-med"):
        cond = CONDITIONS[name]
        spans = windows.window(cond, first.text, first.answering, first.target)
        shown = window_documents(spans, by_id, index)
        body = generation_body(cond, first.text, shown, std, answer, closed)
        out.append((line_for(f"smoke:{name}", first.id, 0, body, mode="standard", role="smoke"), body, shown))
    base = CONDITIONS["base-4.8k-med"]
    for q in answerable[1:3]:  # two very large windows: the baseline's ranking, its first 300 chunks
        shown = window_documents(windows.ranking("base", q.text)[:SMOKE_HUGE_DOCUMENTS], by_id, index)
        body = generation_body(base, q.text, shown, std, answer, closed)
        out.append((line_for("smoke:base-300-documents", q.id, 0, body, mode="standard", role="smoke"), body, shown))
    for name in TIMING_PROBE:
        cond = CONDITIONS[name]
        for q in questions:
            shown = window_documents(windows.window(cond, q.text, q.answering, q.target), by_id, index)
            body = generation_body(cond, q.text, shown, std, answer, closed)
            out.append((line_for(f"timing:{name}", q.id, 0, body, mode="stream", role="timing"), body, shown))
    return out


TIMING_PROBE = ("hhr-4.8k-off", "hhr-4.8k-low", "hhr-4.8k-med", "hhr-4.8k-high")  # 4 settings x 12 questions


def _auth_phase(which: str) -> str:
    return "pilot" if which == "pilot" else "generation"


@gen_app.command("plan")
def gen_plan(which: str = typer.Option("pilot", "--set", help="pilot or golden"),
             count: bool = typer.Option(False, help="count every request's input tokens (free; reads the key) and "
                                                    "record its bound: the golden list is written this way")) -> None:
    """Write the set's generation request list: every request's key, body hash and max_tokens. Spends nothing."""
    from raglab.plan4 import count_all, write_plan

    if which not in SETS:
        _fail(f"--set must be one of {', '.join(SETS)}")
    if which == "golden" and not count:
        _fail("the golden request list carries counted tokens and bounds (PLAN.md, freeze point 2): add --count")
    path = REQUEST_LISTS[_auth_phase(which)]
    try:
        if which == "golden":
            from raglab import freeze

            freeze.forbid(3, "planning golden generation")  # the answers exist once their store is sealed
        items = generation_requests(which)
        lines = [line for line, _, _ in items]
        if count:
            from raglab.llm import AnthropicTransport, keychain_key

            transport = AnthropicTransport(keychain_key())
            lines = count_all(lines, {line.custom_id: body for line, body, _ in items}, transport.count_tokens,
                              Pricing.load(STAGE / "pricing.json"))
        write_plan(path, lines)
    except (LedgerError, ValueError, OSError) as e:
        _fail(str(e))
    except Exception as e:  # noqa: BLE001 - LlmError and the transport's ApiError stop with their message
        _fail(f"{type(e).__name__}: {e}")
    by_condition: dict[str, int] = {}
    for line, _, _ in items:
        by_condition[line.condition] = by_condition.get(line.condition, 0) + 1
    shown = path.relative_to(STAGE.parent) if path.is_relative_to(STAGE.parent) else path.name
    console.print(f"{len(items)} requests written to {shown}: "
                  + ", ".join(f"{c} {n}" for c, n in by_condition.items()))


def _live(auth_phase: str, requests_path: Path, planned_ids: frozenset[str]):
    """Freeze point, key, signed authorization, then the gate that re-checks it before every reservation."""
    from raglab import freeze
    from raglab.llm import AnthropicTransport, keychain_key
    from raglab.runner import Gate
    from raglab.store import Store

    freeze.require(FREEZE_FOR[auth_phase], f"the {auth_phase} phase")
    if auth_phase == "generation":
        freeze.forbid(3, "generation")
    if auth_phase == "judging":
        freeze.forbid(5, "judging")
    key = keychain_key()
    record_path = api_dir() / "authorization.json"
    require(auth_phase, STAGE, record_path, requests_path, key)  # fail early, before anything is counted

    def check() -> str:
        return record_sha256(require(auth_phase, STAGE, record_path, requests_path, key))

    gate = Gate(check, planned_ids)
    transport = AnthropicTransport(key)
    store = Store(api_dir() / "store" / f"{LEDGER_PHASE[auth_phase]}.jsonl")
    return gate, transport, store


@gen_app.command("submit")
def gen_submit(which: str = typer.Option("pilot", "--set", help="pilot or golden"),
               limit: int = typer.Option(0, help="submit at most this many batches now (0: as many as the caps allow)")) -> None:
    """Count, reserve and submit the set's requests not yet answered, in batches. Billed: needs the authorization."""
    from raglab.ledger import CapExceeded
    from raglab.plan4 import as_request, count_all, read_plan
    from raglab.runner import RunStopped, submit_batch

    auth_phase = _auth_phase(which)
    phase = LEDGER_PHASE[auth_phase]
    if which == "pilot" and not smoke_passed():
        _fail("the smoke test hasn't passed: run `gen smoke` first (PLAN.md, 'Smoke test')")
    try:
        planned = read_plan(REQUEST_LISTS[auth_phase])
        bodies = {line.custom_id: body for line, body, _ in generation_requests(which)}
        gate, transport, store = _live(auth_phase, REQUEST_LISTS[auth_phase],
                                       frozenset(line.custom_id for line in planned))
        with open_ledger() as ledger:
            from raglab.verdicts import RESUBMITS, failures_from

            failed = failures_from(ledger.state.items)
            given_up = [line for line in planned if failed.get(line.custom_id, 0) > RESUBMITS]
            if given_up:
                console.print(f"{len(given_up)} request(s) failed twice at the API and are left as status error")
            todo = [line for line in planned if line.mode == "batch" and line.custom_id not in store
                    and not ledger.state.blocking(line.custom_id) and failed.get(line.custom_id, 0) <= RESUBMITS]
            if not todo:
                console.print("nothing to submit: every planned request is answered or in a batch")
                return
            # The golden plan carries counts made before freeze point 2; the pilot's are made now (free).
            uncounted = [line for line in todo if line.input_tokens is None]
            counted = {line.custom_id: line for line in todo if line.input_tokens is not None}
            counted |= {line.custom_id: line for line in count_all(uncounted, bodies, transport.count_tokens,
                                                                  ledger.pricing)}
            reqs = sorted((as_request(counted[l.custom_id], bodies[l.custom_id]) for l in todo),
                          key=lambda r: interleave_key(r.custom_id))
            size = params.BATCH_MAX[params.GENERATOR]
            sent = 0
            for start in range(0, len(reqs), size):
                if limit and sent >= limit:
                    break
                try:
                    group = submit_batch(ledger, store, transport, phase, reqs[start:start + size], gate)
                except CapExceeded as e:
                    console.print(f"[yellow]stopping for now:[/yellow] {escape(str(e))}. Collect results, then submit "
                                  "again: settled batches free their reservations.")
                    break
                sent += 1
                console.print(f"submitted {group}: {len(reqs[start:start + size])} requests")
    except (RunStopped, LedgerError, AuthorizationError, ValueError, OSError) as e:
        _fail(str(e))
    except Exception as e:  # noqa: BLE001 - FreezeError, LlmError: all stop with their message
        _fail(f"{type(e).__name__}: {e}")


@gen_app.command("collect")
def gen_collect(which: str = typer.Option("pilot", "--set", help="pilot or golden"),
                refetch: str = typer.Option("", help="a settled batch group whose stored responses were lost")) -> None:
    """Store and settle every ended batch of the set. Spends nothing, so it needs no authorization. Safe to rerun."""
    from raglab.plan4 import read_plan
    from raglab.runner import RunStopped

    auth_phase = _auth_phase(which)
    try:
        planned = {line.custom_id: line for line in read_plan(REQUEST_LISTS[auth_phase])}
        bodies = {line.custom_id: body for line, body, _ in generation_requests(which)}
        if which == "golden":
            from raglab import freeze

            freeze.forbid(3, "collecting golden answers")  # the sealed store never changes
        if refetch:
            restore_group(LEDGER_PHASE[auth_phase], refetch, planned, bodies)
            return
        collect_groups(LEDGER_PHASE[auth_phase], planned, bodies, only=set(planned))
    except (RunStopped, LedgerError, ValueError, OSError) as e:
        _fail(str(e))
    except Exception as e:  # noqa: BLE001
        _fail(f"{type(e).__name__}: {e}")


def restore_group(phase: str, group: str, planned: dict, bodies: dict) -> None:
    """Fetch an ended batch's results again (the API keeps them 29 days) and restore stored records that were
    lost, only where the response's usage is exactly what the ledger settled. Spends nothing."""
    from raglab.llm import AnthropicTransport, keychain_key
    from raglab.runner import Request, _record
    from raglab.store import Store, body_sha256

    store = Store(api_dir() / "store" / f"{phase}.jsonl")
    with open_ledger() as ledger:
        info = ledger.state.groups.get(group)
        batch_id = ((info or {}).get("sent") or {}).get("batch_id")
        if not batch_id:
            raise LedgerError(f"{group} has no batch to fetch again")
        members = {cid for (g, cid) in ledger.state.items if g == group}
        if not members <= set(planned):
            raise LedgerError(f"{group} isn't one of this request list's batches")
        settled = {e["custom_id"]: e for e in ledger.events() if e["event"] == "settle" and e["group"] == group
                   and e["outcome"] == "succeeded"}
        restored = 0
        for item in AnthropicTransport(keychain_key()).batch_results(batch_id):
            cid, result = item["custom_id"], item["result"]
            if cid in store or cid not in settled or result.get("type") != "succeeded":
                continue
            if body_sha256(bodies[cid]) != planned[cid].body_sha256:
                raise ValueError(f"{cid}: the rebuilt body differs from the planned one")
            message = result["message"]
            if dict(message.get("usage") or {}) != settled[cid]["usage"]:
                raise ValueError(f"{cid}: the fetched usage isn't what the ledger settled; not restored")
            req = Request(planned[cid].condition, bodies[cid], planned[cid].sample, 0)
            store.add(_record(req, phase, message, batch_id, group))
            restored += 1
    console.print(f"{group}: {restored} lost record(s) restored from {batch_id}")


def collect_groups(phase: str, planned: dict, bodies: dict, only: set[str]) -> None:
    """Collect every sent batch of a ledger phase whose requests are all on `planned` (others are left alone:
    the pilot's generation and judge batches share the pilot phase)."""
    from raglab.llm import AnthropicTransport, keychain_key
    from raglab.runner import Request, collect_batch, requests_by_id, settle_from_store
    from raglab.store import Store, body_sha256

    transport = AnthropicTransport(keychain_key())
    store = Store(api_dir() / "store" / f"{phase}.jsonl")
    with open_ledger() as ledger:
        repaired = settle_from_store(ledger, store)
        if repaired:
            console.print(f"settled {repaired} stored responses whose settlement a crash had interrupted")
        groups = sorted({g for (g, _), item in ledger.state.items.items()
                         if item.status == "sent" and ledger.state.groups[g]["phase"] == phase
                         and (ledger.state.groups[g]["sent"] or {}).get("batch_id")})
        for group in groups:
            ids = [cid for (g, cid) in ledger.state.items if g == group]
            if not set(ids) <= only:
                continue  # another request list's batch
            for cid in ids:
                if body_sha256(bodies[cid]) != planned[cid].body_sha256:
                    raise ValueError(f"{cid}: the rebuilt body differs from the planned one")
            # Only the identity of each request matters here: its tokens were reserved when it was sent.
            reqs = requests_by_id(Request(planned[cid].condition, bodies[cid], planned[cid].sample, 0) for cid in ids)
            counts = collect_batch(ledger, store, transport, group, reqs)
            console.print(f"{group}: " + ", ".join(f"{k} {v}" for k, v in counts.items() if v))


SMOKE_REPORT = STAGE / "pilot" / "SMOKE.json"


def smoke_passed() -> bool:
    """The pilot's batches wait for a passed smoke test: a failed citation there is a mapping bug."""
    if not SMOKE_REPORT.exists():
        return False
    return bool(json.loads(SMOKE_REPORT.read_text(encoding="utf-8"))["verdict"]["passed"])


def _send_pilot_lines(role: str, stream: bool) -> tuple[list, dict]:
    """Send the pilot's standard (or streamed) lines of one role, one at a time.

    Returns (line, record, documents) for each line, and the settled cost of each sent one.
    """
    from raglab.plan4 import as_request, count_all, read_plan
    from raglab.runner import send_standard, send_stream

    plan = read_plan(REQUEST_LISTS["pilot"])
    chosen = [line for line in plan if line.role == role]
    if not chosen:
        raise ValueError(f"pilot/requests.jsonl has no {role} lines; it was written by an older `gen plan`")
    built = {line.custom_id: (body, docs) for line, body, docs in generation_requests("pilot")}
    lost = [line.custom_id for line in chosen if line.custom_id not in built]
    if lost:
        raise ValueError(f"{len(lost)} planned {role} request(s) can't be rebuilt (code or corpus changed), "
                         f"e.g. {lost[0]}")
    gate, transport, store = _live("pilot", REQUEST_LISTS["pilot"], frozenset(line.custom_id for line in plan))
    send = send_stream if stream else send_standard
    out = []
    with open_ledger() as ledger:
        todo = [line for line in chosen if line.custom_id not in store]
        counted = {line.custom_id: line for line in count_all(todo, {cid: b for cid, (b, _) in built.items()},
                                                              transport.count_tokens, ledger.pricing)}
        for line in chosen:
            body, docs = built[line.custom_id]
            record = store.get(line.custom_id)
            if record is None:
                record = send(ledger, store, transport, "pilot", as_request(counted[line.custom_id], body), gate)
                console.print(f"{line.condition} {line.question}: {record.outcome}")
            out.append((line, record, docs))
        costs = {i.custom_id: i.cost for i in ledger.state.items.values()
                 if i.status == "settled" and i.custom_id in {line.custom_id for line in chosen}}
    return out, costs


@gen_app.command("smoke")
def gen_smoke() -> None:
    """The smoke test's five answers on the standard API, one at a time, then its checks. Billed (pilot).

    Writes pilot/SMOKE.json: whether every citation mapped, the join rule, the
    usage fields and inference geography, and the settled cost to check
    against the Console. It holds no documentation text.
    """
    from raglab.runner import RunStopped
    from raglab.smoke import answer_check, verdict

    try:
        results, costs = _send_pilot_lines("smoke", stream=False)
    except (RunStopped, LedgerError, AuthorizationError, ValueError, OSError) as e:
        _fail(str(e))
    except Exception as e:  # noqa: BLE001 - FreezeError, LlmError: all stop with their message
        _fail(f"{type(e).__name__}: {e}")
    checks = [{"custom_id": line.custom_id, "condition": line.condition, "question": line.question,
               **answer_check(record.response, docs)} for line, record, docs in results]
    with open_ledger() as ledger:
        us_accepted = ("pilot", "us_inference") in ledger.state.accepted
    v = verdict(checks, us_accepted=us_accepted)
    total = sum((c for c in costs.values() if c is not None), Decimal(0))
    SMOKE_REPORT.write_text(json.dumps({"answers": checks, "verdict": v, "settled": f"{total:.6f}"}, indent=1)
                            + "\n", encoding="utf-8")
    for c in checks:
        console.print(f"{c['condition']} ({c['documents']} documents): {c['status']}, {c['citations']} citations, "
                      f"{c['citations_failed']} failed, rules {c['join_rules']}, geo {c['inference_geo']}")
    console.print(("[green]passed[/green]" if v["passed"] else "[red]failed[/red]") + f": join rule "
                  f"{v['join_rule']!r}; settled {_usd(total)} (check the Console shows the same).")
    for p in v["problems"]:
        console.print(f"[red]problem:[/red] {escape(p)}")
    if not v["passed"]:
        raise typer.Exit(code=1)


@gen_app.command("timing")
def gen_timing() -> None:
    """The pilot's timing probe: 48 streamed standard calls, one at a time. Billed (pilot). Never scored."""
    from raglab.runner import RunStopped

    try:
        results, costs = _send_pilot_lines("timing", stream=True)
    except (RunStopped, LedgerError, AuthorizationError, ValueError, OSError) as e:
        _fail(str(e))
    except Exception as e:  # noqa: BLE001
        _fail(f"{type(e).__name__}: {e}")
    timed = [r.timing for _, r, _ in results if r.timing]
    console.print(f"{len(timed)} of {len(results)} timed; settled "
                  f"{_usd(sum((c for c in costs.values() if c is not None), Decimal(0)))}")


# ---------------------------------------------------------------------------------------------------- batches

batch_app = typer.Typer(help="Loose ends: open batches, lost batch creates, cancels and resends.", no_args_is_help=True)


@batch_app.command("status")
def batch_status() -> None:
    """Every batch the ledger has open, and every unsent or unknown attempt. Reads the ledger only."""
    from raglab.runner import unsent_batches

    try:
        with open_ledger() as ledger:
            state = ledger.state
            pending = set(unsent_batches(ledger))
            table = Table(title="Open ledger groups")
            for col in ("group", "phase", "mode", "batch", "open", "unknown"):
                table.add_column(col)
            for group, info in state.groups.items():
                members = [i for (g, _), i in state.items.items() if g == group]
                still = [i for i in members if i.status in ("reserved", "sent")]
                unknown = [i for i in members if i.status == "unknown" and (group, i.custom_id) not in state.cleared]
                if not still and not unknown:
                    continue
                batch = (info["sent"] or {}).get("batch_id") or ("create got no answer" if group in pending else "-")
                table.add_row(group, info["phase"], info["mode"], batch, str(len(still)), str(len(unknown)))
    except LedgerError as e:
        _fail(str(e))
    console.print(table)


@batch_app.command("reconcile")
def batch_reconcile(release: bool = typer.Option(False, help="release reservations no batch matches (never billed)")) -> None:
    """Match batch creates that got no answer to the batches the API holds, by their custom_ids. Spends nothing."""
    from raglab.llm import AnthropicTransport, keychain_key
    from raglab.runner import reconcile

    try:
        transport = AnthropicTransport(keychain_key())
        with open_ledger() as ledger:
            report = reconcile(ledger, transport, release=release)
    except (LedgerError, ValueError, OSError) as e:
        _fail(str(e))
    except Exception as e:  # noqa: BLE001 - LlmError and the transport's ApiError stop with their message
        _fail(f"{type(e).__name__}: {e}")
    console.print(json.dumps(report, indent=1))


@batch_app.command("cancel")
def batch_cancel(group: str = typer.Argument(..., help="the ledger group whose batch to cancel")) -> None:
    """Cancel a running batch. Its unprocessed items settle at $0 when it is collected. Spends nothing."""
    from raglab.llm import AnthropicTransport, keychain_key

    try:
        with open_ledger() as ledger:
            info = ledger.state.groups.get(group)
            if info is None or not (info["sent"] or {}).get("batch_id"):
                raise LedgerError(f"{group} has no batch to cancel")
            batch_id = info["sent"]["batch_id"]
        result = AnthropicTransport(keychain_key()).batch_cancel(batch_id)
    except (LedgerError, ValueError, OSError) as e:
        _fail(str(e))
    except Exception as e:  # noqa: BLE001
        _fail(f"{type(e).__name__}: {e}")
    console.print(f"{batch_id}: {result.get('processing_status')}; collect it once it has ended")


@batch_app.command("orphan")
def batch_orphan(group: str = typer.Argument(..., help="a standard-API group whose process died mid-call")) -> None:
    """Close a standard request a killed process left behind: "reserved" (never sent) is released at no cost;
    "sent" with no stored response is marked unknown.

    Holding the ledger proves no other process is still waiting for it. An
    unknown one stays counted at its bound; `batch resend` then lets it go out again.
    """
    from raglab.store import Store

    try:
        with open_ledger() as ledger:
            info = ledger.state.groups.get(group)
            if info is None or info["mode"] != "standard":
                raise LedgerError(f"{group} isn't a standard-API group (a batch is reconciled, not orphaned)")
            members = [i for (g, _), i in ledger.state.items.items() if g == group]
            if members and all(i.status == "reserved" for i in members):
                ledger.release(group, "orphaned before it was sent: never billed")
                console.print(f"{group}: never sent, so released at no cost")
                return
            items = [i for i in members if i.status == "sent"]
            if not items:
                raise LedgerError(f"{group} has no request still marked sent or reserved")
            store = Store(api_dir() / "store" / f"{info['phase']}.jsonl")
            for i in items:
                if i.custom_id in store:
                    raise LedgerError(f"{i.custom_id} has a stored response: run `gen collect` or `judge collect`, "
                                      "which settle it from the store")
                ledger.mark_unknown(group, i.custom_id, "orphaned: its process ended before a response was stored")
    except LedgerError as e:
        _fail(str(e))
    console.print(f"{group}: marked unknown; it still counts at its bound")


@batch_app.command("resend")
def batch_resend(group: str = typer.Argument(...), custom_id: str = typer.Argument(...),
                 reason: str = typer.Option(..., help="what you checked, e.g. the Console's usage for that minute")) -> None:
    """Let an unknown attempt go out again. It stays counted at its bound. Run it yourself, in your own terminal."""
    try:
        with open_ledger() as ledger:
            item = ledger.state.items.get((group, custom_id))
            if item is None or item.status != "unknown":
                raise LedgerError(f"{custom_id} in {group} isn't an unknown attempt")
            typed_by_developer(f"resend {custom_id}", f"{custom_id} may already have been billed (it stays counted "
                               f"at {_usd(item.bound)}). To let it be sent again, type exactly:")
            ledger.resend(group, custom_id, reason)
    except (LedgerError, AuthorizationError) as e:
        _fail(str(e))
    console.print(f"{custom_id} may be sent again; it still counts at its bound")


def register(app: typer.Typer) -> None:
    from raglab.cli4_judge import judge_app, pilot_app

    app.add_typer(budget_app, name="budget")
    app.add_typer(gen_app, name="gen")
    app.add_typer(judge_app, name="judge")
    app.add_typer(pilot_app, name="pilot")
    app.add_typer(batch_app, name="batch")
