"""Command line for Stage 3: everything in Stage 2, plus the pinned models and the neural retrievers."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from raglab.chunking import CHUNKERS, chunk_corpus, chunker_config, get_chunker
from raglab.corpus import (
    EXCLUDED_DOCS,
    CorpusError,
    CorpusLock,
    LoadReport,
    default_corpus_dir,
    default_lock_path,
    fetch_corpus,
    load_corpus,
    validate_ref,
    verify_corpus,
)
from raglab.evaluate import (
    build_manifest, load_run, run_configs, save_run, summarize, unjudged,
)
from raglab.experiments import BASELINE, FACTORIAL, GRID, RetrievalConfig, build_retrievers
from raglab.golden import (
    STYLES, GoldenError, GoldenSet, agreement_report, area_of, check_against_corpus, default_golden_path, dump_golden,
    load_golden, merge_judgments, questions_fingerprint, sample_targets, strict_json,
)
from raglab.metrics import BUDGETS, HEADLINE_METRICS, LABELS
from raglab.models import ROLES, ModelError, ModelLock, default_models_lock_path, fetch, local_dir, problems
from raglab.report import budget_chart_svg, markdown_report
from raglab.retrieval import RETRIEVER_NAMES, RetrievalError, build_retriever
from raglab.sections import SectionIndex
from raglab.stats import RESAMPLES, SEED

app = typer.Typer(
    help="RAG Eval Lab, Stage 3 — pinned corpus and models, lexical and neural retrieval, and a golden-set evaluation.",
    no_args_is_help=True,
    add_completion=False,
)
corpus_app = typer.Typer(help="Fetch, verify and inspect the pinned FastAPI docs corpus.", no_args_is_help=True)
golden_app = typer.Typer(help="Write, check and complete the golden question set.", no_args_is_help=True)
eval_app = typer.Typer(help="Score every configuration against the golden set.", no_args_is_help=True)
models_app = typer.Typer(help="Download and verify the pinned embedding and reranking models.", no_args_is_help=True)
app.add_typer(corpus_app, name="corpus")
app.add_typer(golden_app, name="golden")
app.add_typer(eval_app, name="eval")
app.add_typer(models_app, name="models")
console = Console()


def _fail(message: str) -> None:
    console.print(f"[red]Error:[/red] {escape(message)}")
    raise typer.Exit(code=1)


def _load() -> tuple[CorpusLock, LoadReport]:
    """The lock and the corpus — verified against each other on every run."""
    try:
        lock = CorpusLock.load(default_lock_path())
        return lock, load_corpus(default_corpus_dir(lock.ref), lock=lock)
    except CorpusError as e:
        _fail(str(e))
        raise  # unreachable; keeps type checkers honest


@corpus_app.command("fetch")
def corpus_fetch(
    ref: Optional[str] = typer.Option(None, help="FastAPI tag to pin. Defaults to the ref already in corpus.lock.json."),
) -> None:
    """Download the docs at a pinned tag (sparse + shallow), then lock or verify them."""
    lock_path = default_lock_path()
    try:
        existing = CorpusLock.load(lock_path) if lock_path.is_file() else None
        if ref is None:
            if existing is None:
                _fail("No corpus.lock.json yet — pass --ref <tag> to pin one (e.g. --ref 0.165.0).")
            ref = existing.ref
        validate_ref(ref)
        expected = existing if existing is not None and existing.ref == ref else None
        with console.status(f"Fetching FastAPI docs at {escape(ref)}..."):
            lock = fetch_corpus(ref, default_corpus_dir(ref), expected)
        if expected is None:
            lock.save(lock_path)
    except (CorpusError, OSError) as e:
        _fail(str(e))
    if expected is None:
        console.print(f"[green]Pinned[/green] FastAPI {escape(ref)} @ {lock.commit[:12]} — {lock.file_count} files. Wrote {lock_path.name}.")
        if existing is not None:
            console.print(f"[yellow]Note:[/yellow] the pin moved from {escape(existing.ref)}; earlier results no longer apply to this corpus.")
    else:
        console.print(f"[green]Re-fetched[/green] FastAPI {escape(ref)}; matches the lock byte for byte ({lock.file_count} files).")


@corpus_app.command("verify")
def corpus_verify() -> None:
    """Prove the on-disk corpus is byte-identical to corpus.lock.json."""
    try:
        lock = CorpusLock.load(default_lock_path())
        problems = verify_corpus(default_corpus_dir(lock.ref), lock)
    except CorpusError as e:
        _fail(str(e))
    if problems:
        for p in problems[:20]:
            console.print(f"  [red]{escape(p)}[/red]")
        _fail(f"{len(problems)} difference(s) from the lock. Re-run `corpus fetch` to restore it.")
    console.print(f"[green]OK[/green] — {lock.file_count} files match the lock (FastAPI {lock.ref} @ {lock.commit[:12]}).")


@models_app.command("fetch")
def models_fetch() -> None:
    """Download exactly the files models.lock.json pins (safetensors weights only), then check their hashes."""
    try:
        lock = ModelLock.load(default_models_lock_path())
        for role in ROLES:
            spec = lock[role]
            with console.status(f"Fetching {escape(spec.repo)} @ {spec.revision[:12]}..."):
                fetch(spec)
            console.print(f"[green]OK[/green] {role}: {escape(spec.repo)} @ {spec.revision[:12]} — "
                          f"{len(spec.files)} files match the lock.")
    except (ModelError, OSError) as e:
        _fail(str(e))
    except Exception as e:  # network and Hub errors: say what failed, without a traceback
        _fail(f"Download failed: {type(e).__name__}: {e}")


@models_app.command("verify")
def models_verify() -> None:
    """Prove the local copies of both models are byte-identical to models.lock.json."""
    try:
        lock = ModelLock.load(default_models_lock_path())
    except ModelError as e:
        _fail(str(e))
    failed = 0
    for role in ROLES:
        spec = lock[role]
        directory = local_dir(spec)
        found = ["not downloaded"] if directory is None else problems(spec, directory)
        if found:
            failed += 1
            console.print(f"[red]{role}[/red]: {escape(spec.repo)} — {escape('; '.join(found))}")
        else:
            console.print(f"[green]OK[/green] {role}: {escape(spec.repo)} @ {spec.revision[:12]} — "
                          f"{len(spec.files)} files match the lock.")
    if failed:
        _fail("Run `python -m raglab models fetch` to restore the pinned models.")


@corpus_app.command("stats")
def corpus_stats() -> None:
    """What was loaded, what was excluded and why."""
    lock, report = _load()
    docs = report.documents
    sections = [s for d in docs for s in d.sections if s.level > 0]
    console.print(f"FastAPI [bold]{lock.ref}[/bold] @ {lock.commit[:12]}  (fingerprint {lock.fingerprint[:16]}…)")
    console.print(f"  documents loaded : {len(docs)}")
    console.print(f"  cleaned text     : {sum(len(d.text) for d in docs):,} chars")
    console.print(f"  sections         : {len(sections):,} (each one a citable doc#anchor)")
    console.print(f"  unresolved includes: {len(report.unresolved_includes)}")
    for item in report.unresolved_includes:
        console.print(f"    [dim]{escape(item)}[/dim]")
    console.print(f"  excluded ({len(report.excluded)}):")
    for rel in report.excluded:
        console.print(f"    [yellow]{rel}[/yellow] — {EXCLUDED_DOCS.get(rel, '')}")


@app.command("chunk")
def chunk_stats() -> None:
    """Compare the chunking strategies' output on the real corpus."""
    _, report = _load()
    table = Table(title="Chunking strategies (chars per chunk)")
    for col in ("strategy", "params", "chunks", "mean", "median", "p95", "max"):
        table.add_column(col, justify="right" if col not in ("strategy", "params") else "left")
    for name in CHUNKERS:
        chunker = get_chunker(name)
        lengths = sorted(len(c.text) for c in chunk_corpus(list(report.documents), chunker))
        params = ", ".join(f"{k}={v}" for k, v in chunker_config(chunker).items() if k != "name") or "split on blank lines"
        table.add_row(
            name, params, f"{len(lengths):,}", f"{statistics.mean(lengths):.0f}",
            f"{statistics.median(lengths):.0f}", str(lengths[int(len(lengths) * 0.95)]), str(lengths[-1]),
        )
    console.print(table)


def _snippet(text: str, width: int = 160) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= width else flat[: width - 1] + "…"


@app.command("search")
def search(
    query: str = typer.Argument(..., help="Natural-language question about FastAPI."),
    chunker: str = typer.Option("heading", help=f"One of: {', '.join(CHUNKERS)}."),
    retriever: str = typer.Option("bm25", help=f"One of: {', '.join(RETRIEVER_NAMES)}."),
    k: int = typer.Option(5, min=1, max=50, help="How many chunks to return."),
    context: bool = typer.Option(True, help="Prefix each chunk with its heading breadcrumb when indexing."),
) -> None:
    """Retrieve the top-k chunks for a question under one configuration."""
    _, report = _load()
    try:
        chunks = chunk_corpus(list(report.documents), get_chunker(chunker))
        hits = build_retriever(retriever, chunks, with_context=context).search(query, k)
    except (ValueError, RetrievalError, ModelError) as e:
        _fail(str(e))
    config = RetrievalConfig(chunker, retriever, context)
    if not hits:
        console.print(f"No chunk matched under {config.name}.")
        return
    table = Table(title=f"{escape(query)}  —  {config.name}")
    table.add_column("#", justify="right")
    table.add_column("score", justify="right")
    table.add_column("location (doc#anchor)", overflow="fold")
    table.add_column("text")
    for hit in hits:
        table.add_row(str(hit.rank), f"{hit.score:.3f}", escape(hit.chunk.location), escape(_snippet(hit.chunk.text)))
    console.print(table)


@app.command("compare")
def compare(
    query: str = typer.Argument(..., help="Natural-language question about FastAPI."),
    k: int = typer.Option(3, min=1, max=10, help="How many results per configuration."),
) -> None:
    """Run one question through every configuration side by side."""
    _, report = _load()
    try:
        retrievers = build_retrievers(list(report.documents), GRID)
    except (ValueError, RetrievalError, ModelError) as e:
        _fail(str(e))
    table = Table(title=f"{escape(query)}  —  top {k} sections per configuration")
    table.add_column("configuration")
    for rank in range(1, k + 1):
        table.add_column(f"#{rank}", overflow="fold")  # locations are IDs; never hide their ends
    for name, retriever in retrievers.items():
        cells = [escape(h.chunk.location) for h in retriever.search(query, k)]
        table.add_row(escape(name), *(cells + ["—"] * (k - len(cells))))
    console.print(table)
    console.print("[dim]Which row is right? That is the question Stage 2's golden set answers with numbers.[/dim]")


# --- golden set -------------------------------------------------------------------------------

SAMPLE_SEED = 20261120  # the seed the v1 golden set was sampled with


def _golden_context(golden_path: Optional[Path]) -> tuple[CorpusLock, list, SectionIndex, GoldenSet]:
    """Lock, verified corpus, section index and golden set, checked against each other."""
    lock, report = _load()
    docs = list(report.documents)
    index = SectionIndex(docs)
    path = golden_path or default_golden_path()
    try:
        golden = load_golden(path)
    except GoldenError as e:
        _fail(str(e))
    problems = check_against_corpus(golden, lock, index)
    if problems:
        for p in problems[:20]:
            console.print(f"  [red]{escape(p)}[/red]")
        _fail(f"{len(problems)} problem(s) between {path.name} and the corpus.")
    return lock, docs, index, golden


def _excerpt(text: str, limit: int) -> str:
    """Cut long text, and say so: a reader who thinks the text ends there will misjudge it."""
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + f" [truncated: {len(text) - limit:,} more characters; `golden show` prints all]"


@golden_app.command("sample")
def golden_sample(
    n: int = typer.Option(100, min=1, help="How many sections to sample."),
    seed: int = typer.Option(SAMPLE_SEED, help="Sampling seed (the v1 set used the default)."),
    out: Path = typer.Option(Path("golden-worksheet.md"), help="Where to write the authoring worksheet."),
) -> None:
    """Sample target sections (stratified by docs area, seeded) and write a worksheet to author questions from."""
    _, report = _load()
    docs = {d.doc_id: d for d in report.documents}
    index = SectionIndex(docs.values())
    try:
        sampled = sample_targets(index, n, seed)
    except ValueError as e:
        _fail(str(e))
    lines = [f"# Golden set authoring worksheet: seed {seed}, {len(sampled)} sections", ""]
    for i, s in enumerate(sampled, start=1):
        body = docs[s.ref.doc_id].text[s.ref.start:s.ref.end]
        lines += [
            f"## s{i:03d} · {s.style} · {s.ref.location}",
            f"Path: {' > '.join(s.ref.heading_path)}  (area {s.area}, body {s.ref.length} chars)",
            "", _excerpt(body, 1500), "",
        ]
    try:
        out.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    except OSError as e:
        _fail(f"Cannot write {out}: {e}")
    areas: dict[str, int] = {}
    for s in sampled:
        areas[s.area] = areas.get(s.area, 0) + 1
    console.print(f"Sampled [bold]{len(sampled)}[/bold] of {n} requested sections (seed {seed}) -> {escape(str(out))}")
    console.print("  by area : " + ", ".join(f"{a} {c}" for a, c in sorted(areas.items(), key=lambda kv: -kv[1])))
    console.print("  by style: " + ", ".join(f"{st} {sum(s.style == st for s in sampled)}" for st in STYLES))


@golden_app.command("validate")
def golden_validate(
    golden: Optional[Path] = typer.Option(None, help="Golden set file (default: golden/v1.yaml)."),
) -> None:
    """Check the golden set's structure and that every judged section exists in the pinned corpus."""
    lock, _, _, gs = _golden_context(golden)
    grades = {0: 0, 1: 0, 2: 0}
    for q in gs.questions:
        for g in q.judgments.values():
            grades[g] += 1
    answers = [len(q.primary) for q in gs.questions]
    console.print(f"[green]OK[/green] — {escape(gs.path.name)} v{gs.version}: {len(gs.questions)} questions, "
                  f"labelled against FastAPI {gs.corpus_ref} (matches the lock).")
    console.print("  styles   : " + ", ".join(f"{st} {sum(q.style == st for q in gs.questions)}" for st in STYLES))
    areas: dict[str, int] = {}
    for q in gs.questions:
        a = area_of(q.target.split("#", 1)[0])
        areas[a] = areas.get(a, 0) + 1
    console.print("  targets  : " + ", ".join(f"{a} {c}" for a, c in sorted(areas.items(), key=lambda kv: -kv[1])))
    console.print(f"  judgments: {sum(grades.values())} (answers {grades[2]}, related {grades[1]}, rejected {grades[0]})")
    console.print(f"  answering sections per question: mean {statistics.mean(answers):.2f}, max {max(answers)}")
    console.print(f"  questions fingerprint: {questions_fingerprint(gs.questions)}  [dim](unchanged by judging)[/dim]")


@golden_app.command("show")
def golden_show(
    location: str = typer.Argument(..., help="A section location, e.g. tutorial/handling-errors#use-httpexception."),
    chars: Optional[int] = typer.Option(None, min=100, help="Show at most this many characters (default: all)."),
) -> None:
    """Print a section's heading path and whole body: what a judge reads."""
    _, report = _load()
    docs = {d.doc_id: d for d in report.documents}
    index = SectionIndex(docs.values())
    ref = index.get(location)
    if ref is None:
        _fail(f"{location} is not a section with text in the corpus.")
    body = docs[ref.doc_id].text[ref.start:ref.end]
    console.print(f"[bold]{escape(' > '.join(ref.heading_path))}[/bold]  [dim]({escape(location)}, {ref.length} chars)[/dim]")
    console.print(escape(body if chars is None else body[:chars]), soft_wrap=True)
    if chars is not None and len(body) > chars:
        # Say so plainly: a judge who thinks the text ends here will grade it wrongly.
        console.print("[yellow]" + escape(f"[truncated: {len(body) - chars:,} more characters; omit --chars to see all]") + "[/yellow]")


@golden_app.command("pool")
def golden_pool(
    depth: int = typer.Option(10, min=1, max=10, help="Pool every configuration's top N sections (metrics read 10)."),
    golden: Optional[Path] = typer.Option(None, help="Golden set file (default: golden/v1.yaml)."),
    worksheet: Optional[Path] = typer.Option(None, help="Write the unjudged sections, with their text, to this JSON file."),
) -> None:
    """List pooled sections nobody has judged yet. Exits 1 if there are any, so it can gate CI."""
    _, docs, index, gs = _golden_context(golden)
    try:
        results = run_configs(docs, gs.questions, FACTORIAL, index)
    except (ValueError, RetrievalError, ModelError) as e:
        _fail(str(e))
    pool = unjudged(gs.questions, results, depth)
    total = sum(len(v) for v in pool.values())
    if worksheet is not None:
        by_doc = {d.doc_id: d for d in docs}

        def body(location: str) -> str:
            ref = index.get(location)
            return by_doc[ref.doc_id].text[ref.start:ref.end]

        # Blind by construction: no hint of which or how many configurations surfaced a
        # section, so a judge can't be swayed by "everything found it, it must be right".
        items = [
            {
                "id": q.id,
                "question": q.text,
                # Full text, never an excerpt: judges shown cut-off sections under-grade long ones.
                "answers": {loc: body(loc) for loc in sorted(q.primary)},
                "candidates": [
                    {"location": loc, "heading_path": " > ".join(index.get(loc).heading_path), "text": body(loc)}
                    for loc in sorted(pool[q.id])
                ],
            }
            for q in gs.questions if q.id in pool
        ]
        try:
            worksheet.write_text(json.dumps(items, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
        except OSError as e:
            _fail(f"Cannot write {worksheet}: {e}")
        console.print(f"Wrote {len(items)} questions' candidates to {escape(str(worksheet))}")
    if total:
        console.print(f"[yellow]{total} unjudged section(s)[/yellow] across {len(pool)} question(s) in the top {depth} "
                      f"of {len(FACTORIAL)} configurations.")
        raise typer.Exit(code=1)
    console.print(f"[green]Complete[/green] — every section in every configuration's top {depth} is judged "
                  f"({len(gs.questions)} questions, {len(FACTORIAL)} configurations).")


@golden_app.command("merge")
def golden_merge(
    judgments: Path = typer.Argument(..., help='JSON file of {"q001": {"doc#anchor": 0|1|2, ...}, ...}.'),
    golden: Optional[Path] = typer.Option(None, help="Golden set file (default: golden/v1.yaml)."),
    regrade: bool = typer.Option(False, help="Allow changing existing grades (never a target's). For deliberate corrections."),
) -> None:
    """Add new judgments to the golden set. Existing judgments change only with --regrade."""
    _, _, index, gs = _golden_context(golden)
    try:
        additions = strict_json(judgments.read_text(encoding="utf-8"))
        if not isinstance(additions, dict) or not all(isinstance(v, dict) for v in additions.values()):
            raise ValueError("expected an object of question id -> {location: grade}")
        unknown = sorted({loc for v in additions.values() for loc in v if loc not in index})
        if unknown:
            raise ValueError(f"{len(unknown)} location(s) not in the corpus, e.g. {unknown[0]}")
        data, added, changed = merge_judgments(gs.raw, additions, regrade=regrade)
    except (OSError, ValueError, GoldenError) as e:
        _fail(str(e))
    tmp = gs.path.with_name(gs.path.name + ".merge")
    dump_golden(data, tmp)
    try:
        load_golden(tmp)  # the merged file must still be a valid golden set
    except GoldenError as e:
        tmp.unlink(missing_ok=True)
        _fail(str(e))
    tmp.replace(gs.path)
    console.print(f"[green]Merged[/green] {added} new judgment(s) into {escape(gs.path.name)}"
                  + (f"; changed {changed} existing grade(s)." if changed else "."))


@golden_app.command("agreement")
def golden_agreement(
    file: Optional[Path] = typer.Option(None, help="Agreement file (default: golden/v1-agreement.json)."),
) -> None:
    """How often a second, blind judge agreed with the labels (exact agreement and Cohen's kappa)."""
    path = file or default_golden_path().with_name(default_golden_path().stem + "-agreement.json")
    try:
        data = strict_json(path.read_text(encoding="utf-8"))
        r = agreement_report(data["items"], data.get("population"))
    except (OSError, ValueError, KeyError, TypeError) as e:
        _fail(f"Cannot read agreement file {path.name}: {e}")
    console.print(f"{r['items']} judgments re-graded blind (first pass: answers {r['first_counts'][2]}, "
                  f"related {r['first_counts'][1]}, not relevant {r['first_counts'][0]}).")
    console.print(f"  exact agreement : {r['exact']:.0%}")
    console.print(f"  Cohen's kappa   : {r['kappa']:.2f} (linear-weighted {r['kappa_linear']:.2f}); "
                  f"answer vs not-an-answer {r['answer_kappa']:.2f}")
    if r["reweighted"]:
        rw = r["reweighted"]
        console.print(f"  reweighted to the whole label set: agreement {rw['exact']:.0%}, kappa {rw['kappa']:.2f} "
                      f"(linear-weighted {rw['kappa_linear']:.2f}); answer vs not-an-answer {rw['answer_kappa']:.2f}")
    for g, label in ((2, "answers"), (1, "related"), (0, "not relevant")):
        kept = r["agreement_by_first_grade"][g]
        console.print(f"  first pass '{label}' kept by the second judge: {'n/a' if kept is None else f'{kept:.0%}'}")
    if r["targets"]:
        console.print(f"  question targets also graded 'answers': {r['targets_confirmed']} of {r['targets']}")
    table = Table(title="rows: first judge · columns: second judge")
    table.add_column("")
    for label in ("0", "1", "2"):
        table.add_column(label, justify="right")
    for g, row in zip((0, 1, 2), r["confusion"]):
        table.add_row(str(g), *(str(v) for v in row))
    console.print(table)


# --- evaluation ------------------------------------------------------------------------------


def _ci(cell: dict) -> str:
    return f"{cell['mean']:.3f} [dim][{cell['lo']:.2f}, {cell['hi']:.2f}][/dim]"


def _print_summary(summary: dict, manifest: dict) -> None:
    metrics = summary["headline"]
    table = Table(title=f"{summary['questions']} questions · mean [95% bootstrap CI] · run {manifest['run_id']}")
    table.add_column("configuration")
    for m in metrics:
        table.add_column(LABELS[m], justify="right")
    order = sorted(summary["configs"], key=lambda n: -summary["summary"][n][metrics[0]]["mean"])
    for name in order:
        style = "bold" if name == summary["baseline"] else ""
        table.add_row(escape(name), *(_ci(summary["summary"][name][m]) for m in metrics), style=style)
    console.print(table)

    vs = Table(title=f"Paired difference vs {escape(summary['baseline'])} · Δ [95% CI] · Holm-adjusted p")
    vs.add_column("configuration")
    for m in metrics:
        vs.add_column(LABELS[m], justify="right")
    for name in order:
        if name == summary["baseline"]:
            continue
        cells = []
        for m in metrics:
            d = summary["vs_baseline"][name][m]
            mark = "[green]" if d["p_holm"] < 0.05 and d["delta"] > 0 else "[red]" if d["p_holm"] < 0.05 else "[dim]"
            cells.append(f"{mark}{d['delta']:+.3f}[/] [dim][{d['lo']:+.2f}, {d['hi']:+.2f}] p={d['p_holm']:.3g}[/dim]")
        vs.add_row(escape(name), *cells)
    console.print(vs)


@eval_app.command("run")
def eval_run(
    golden: Optional[Path] = typer.Option(None, help="Golden set file (default: golden/v1.yaml)."),
    out: Optional[Path] = typer.Option(None, help="Run directory (default: runs/<run id>)."),
    overwrite: bool = typer.Option(False, help="Replace a run already in --out."),
    resamples: int = typer.Option(RESAMPLES, min=1000, help="Bootstrap and randomisation resamples."),
    seed: int = typer.Option(SEED, help="Seed for every resample."),
) -> None:
    """Score every configuration against the golden set and save a reproducible run."""
    lock, docs, index, gs = _golden_context(golden)
    try:
        manifest = build_manifest(gs, lock, FACTORIAL, resamples=resamples, seed=seed)
    except ModelError as e:
        _fail(str(e))
    run_dir = out or Path("runs") / manifest["run_id"]
    try:
        with console.status(f"Scoring {len(FACTORIAL)} configurations on {len(gs.questions)} questions..."):
            results = run_configs(docs, gs.questions, FACTORIAL, index)
            summary = summarize(gs.questions, FACTORIAL, results, resamples=resamples, seed=seed)
        save_run(run_dir, manifest, summary, gs.questions, results, overwrite=overwrite)
    except (ValueError, RetrievalError, ModelError, OSError) as e:
        _fail(str(e))
    _print_summary(summary, manifest)
    unjudged_rate = 1 - min(d["judged@10"] for d in summary["diagnostics"].values())
    if unjudged_rate > 0:
        console.print(f"[yellow]Note:[/yellow] up to {unjudged_rate:.0%} of a configuration's top-10 sections are "
                      "unjudged (scored as not relevant); `golden pool --depth 10` lists them.")
    code = manifest["code"]
    console.print(f"Saved to {escape(str(run_dir))} · code {code['package_sha256'][:12]}"
                  f" (git {code['git_commit'][:12] if code['git_commit'] else 'unavailable'}"
                  f"{', uncommitted changes' if code['git_dirty'] else ''}) · golden {gs.sha256[:12]}"
                  f" · corpus {lock.fingerprint[:12]}")


@eval_app.command("report")
def eval_report(
    run_dir: Path = typer.Argument(..., help="A directory written by `eval run`."),
    markdown: bool = typer.Option(False, help="Print README-ready Markdown tables instead."),
) -> None:
    """Print a saved run's tables again."""
    try:
        manifest, summary = load_run(run_dir)
    except ValueError as e:
        _fail(str(e))
    try:
        if markdown:
            print(markdown_report(summary, manifest), end="")  # plain print: no Rich wrapping inside tables
            return
        _print_summary(summary, manifest)
    except (KeyError, TypeError, ValueError, RecursionError, ArithmeticError) as e:
        _fail(f"{run_dir} looks damaged ({type(e).__name__}: {e}); re-run `eval run` to regenerate it")


@eval_app.command("chart")
def eval_chart(
    run_dir: Path = typer.Argument(..., help="A directory written by `eval run`."),
    highlight: list[str] = typer.Option(..., help="Configurations to highlight (1-3); repeat the option."),
    out: Path = typer.Option(Path("budget.svg"), help="Where to write the SVG."),
) -> None:
    """Success and coverage against context budget: highlighted configurations over the rest."""
    try:
        _, summary = load_run(run_dir)
        unknown = [h for h in highlight if h not in summary["configs"]]
        if unknown:
            raise ValueError(f"not in this run: {', '.join(unknown)}")
        svg = budget_chart_svg(
            summary, highlight, BUDGETS,
            "Small chunks get something into the prompt more often. Bigger chunks get more of the answer in.",
            f"{summary['questions']} questions · mean over questions · each ranking packed into the budget "
            "as its longest prefix that fits",
        )
        out.write_text(svg, encoding="utf-8", newline="\n")
    except OSError as e:
        _fail(str(e))
    except (ValueError, KeyError, TypeError, RecursionError, ArithmeticError) as e:
        _fail(f"{run_dir} looks damaged ({type(e).__name__}: {e}); re-run `eval run` to regenerate it")
    console.print(f"Wrote {escape(str(out))}")
