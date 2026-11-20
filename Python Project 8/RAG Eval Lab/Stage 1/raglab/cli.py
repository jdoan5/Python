"""Command line for Stage 1: pin the corpus, inspect chunking, run retrieval."""

from __future__ import annotations

import statistics
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
from raglab.experiments import GRID, RetrievalConfig, build_retrievers
from raglab.retrieval import RETRIEVERS, RetrievalError, build_retriever

app = typer.Typer(
    help="RAG Eval Lab, Stage 1 — pinned corpus, chunking strategies, lexical retrieval.",
    no_args_is_help=True,
    add_completion=False,
)
corpus_app = typer.Typer(help="Fetch, verify and inspect the pinned FastAPI docs corpus.", no_args_is_help=True)
app.add_typer(corpus_app, name="corpus")
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
    retriever: str = typer.Option("bm25", help=f"One of: {', '.join(sorted(RETRIEVERS))}."),
    k: int = typer.Option(5, min=1, max=50, help="How many chunks to return."),
    context: bool = typer.Option(True, help="Prefix each chunk with its heading breadcrumb when indexing."),
) -> None:
    """Retrieve the top-k chunks for a question under one configuration."""
    _, report = _load()
    try:
        chunks = chunk_corpus(list(report.documents), get_chunker(chunker))
        hits = build_retriever(retriever, chunks, with_context=context).search(query, k)
    except (ValueError, RetrievalError) as e:
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
    except (ValueError, RetrievalError) as e:
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
