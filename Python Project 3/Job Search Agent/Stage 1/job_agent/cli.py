"""CLI for the Job Search Agent.

    python -m job_agent.cli run <url-or-file>       # full pipeline with HITL gate
    python -m job_agent.cli search "<query>"        # test retrieval directly
    python -m job_agent.cli validate                # lint the inventory file
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import anthropic
import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.prompt import Confirm
from rich.table import Table

from job_agent.fetcher import FetchError, fetch_posting
from job_agent.pipeline import GroundingError, PipelineError, run_pipeline
from job_agent.retrieval import Bm25Index, InventoryError, load_inventory
from job_agent.schemas import JobRequirements, TailoredDraft

app = typer.Typer(help="RAG-grounded, human-approved job application tailoring.")
console = Console()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INVENTORY = PROJECT_ROOT / "data" / "experience_inventory.yaml"
DEFAULT_OUTPUT = PROJECT_ROOT / "output"


def _load_index(inventory: Path) -> Bm25Index:
    try:
        entries = load_inventory(inventory)
        return Bm25Index(entries)
    except InventoryError as e:
        console.print(f"[red]Inventory error:[/red] {e}")
        raise typer.Exit(code=1)


def _get_posting_text(source: str) -> str:
    path = Path(source)
    try:
        is_file = path.exists()
    except OSError:  # e.g. a pasted posting body used as a path (name too long)
        is_file = False
    if is_file:
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            console.print(f"[red]File is empty:[/red] {path}")
            raise typer.Exit(code=1)
        return text
    try:
        with console.status("Fetching posting..."):
            return fetch_posting(source)
    except FetchError as e:
        console.print(f"[red]Fetch error:[/red] {e}")
        raise typer.Exit(code=1)


def _review_and_approve(requirements: JobRequirements, draft: TailoredDraft) -> bool:
    """The human-in-the-loop gate: render everything, then ask.

    All model/posting-derived text goes through escape() — a posting containing
    Rich markup like [red] must not crash or restyle the review screen.
    """
    console.print()
    console.print(Panel(
        f"[bold]{escape(requirements.role)}[/bold] @ {escape(requirements.company)} "
        f"(seniority: {escape(requirements.seniority)})",
        title="Posting", border_style="blue",
    ))

    fit_color = "green" if draft.fit_score >= 7 else "yellow" if draft.fit_score >= 5 else "red"
    console.print(
        f"\n[bold {fit_color}]Fit: {draft.fit_score}/10[/bold {fit_color}] — "
        f"{escape(draft.fit_rationale)}\n"
    )

    table = Table(title="Requirement coverage", show_lines=False)
    table.add_column("Requirement", max_width=50)
    table.add_column("Strength")
    table.add_column("Evidence")
    for match in draft.matches:
        style = {"strong": "green", "partial": "yellow", "none": "red"}.get(match.strength, "")
        table.add_row(escape(match.requirement), f"[{style}]{escape(match.strength)}[/{style}]",
                      escape(", ".join(match.evidence_ids)) or "—")
    console.print(table)

    console.print("\n[bold]Drafted bullets[/bold] (each cites your real inventory entries):")
    for i, bullet in enumerate(draft.bullets, 1):
        console.print(f"  {i}. {escape(bullet.text)}")
        console.print(
            f"     [dim]evidence: {escape(', '.join(bullet.evidence_ids))} | "
            f"targets: {escape(bullet.targets_requirement)}[/dim]"
        )

    if draft.gaps:
        console.print("\n[bold red]Honest gaps[/bold red] (no supporting evidence found):")
        for gap in draft.gaps:
            console.print(f"  - {escape(gap)}")

    if draft.cover_note:
        console.print(Panel(escape(draft.cover_note), title="Cover note", border_style="dim"))

    console.print()
    return Confirm.ask("[bold]Save this draft?[/bold]", default=False)


@app.command()
def run(
    source: str = typer.Argument(..., help="Job posting URL, or path to a saved .txt file."),
    inventory: Path = typer.Option(DEFAULT_INVENTORY, "--inventory", "-i"),
    output_dir: Path = typer.Option(DEFAULT_OUTPUT, "--out", "-o"),
    model: Optional[str] = typer.Option(None, "--model", help="Override the model id."),
    yes: bool = typer.Option(False, "--yes", help="Skip the approval prompt (demo/CI only)."),
) -> None:
    """Run the full pipeline: extract -> evidence -> draft -> approve -> save."""
    index = _load_index(inventory)
    posting_text = _get_posting_text(source)

    approve = (lambda r, d: True) if yes else _review_and_approve
    kwargs = {"model": model} if model else {}

    try:
        result = run_pipeline(
            posting_text=posting_text,
            source=source,
            index=index,
            approve=approve,
            output_dir=output_dir,
            on_stage=lambda name: console.print(f"[dim cyan]→ {name}[/dim cyan]"),
            on_search=lambda q, n: console.print(f"[dim]  searched: {q!r} ({n} hits)[/dim]"),
            **kwargs,
        )
    except GroundingError as e:
        console.print(f"[red]Grounding violation (draft rejected):[/red] {e}")
        console.print("[dim]The model cited experience that retrieval never returned. Re-run.[/dim]")
        raise typer.Exit(code=2)
    except PipelineError as e:
        console.print(f"[red]Pipeline error:[/red] {escape(str(e))}")
        raise typer.Exit(code=1)
    except anthropic.RateLimitError:
        console.print("[red]Rate limited by the API.[/red] Wait a minute and re-run.")
        raise typer.Exit(code=1)
    except anthropic.AuthenticationError:
        console.print("[red]Invalid API key.[/red] Check ANTHROPIC_API_KEY in your .env.")
        raise typer.Exit(code=1)
    except anthropic.APIStatusError as e:
        console.print(f"[red]API error ({e.status_code}):[/red] {escape(e.message)}")
        raise typer.Exit(code=1)
    except anthropic.APIConnectionError:
        console.print("[red]Could not reach the Anthropic API.[/red] Check your connection.")
        raise typer.Exit(code=1)
    except RuntimeError as e:
        console.print(f"[red]{escape(str(e))}[/red]")
        raise typer.Exit(code=1)

    if result.approved:
        console.print(f"\n[green]Saved:[/green] {result.output_path}")
    else:
        console.print("\n[yellow]Draft rejected — nothing written to disk.[/yellow]")


@app.command()
def search(
    query: str = typer.Argument(..., help="Test query against your inventory."),
    inventory: Path = typer.Option(DEFAULT_INVENTORY, "--inventory", "-i"),
    top_k: int = typer.Option(5, "--top-k", "-k", min=1),
) -> None:
    """Run a BM25 search directly — useful for tuning your inventory."""
    index = _load_index(inventory)
    hits = index.search(query, top_k=top_k)
    if not hits:
        console.print("[yellow]No matches.[/yellow] Add tags/keywords to relevant entries.")
        return
    for hit in hits:
        console.print(f"[bold]{escape(hit.entry.entry_id)}[/bold] "
                      f"(score {hit.score:.2f}, {escape(hit.entry.kind)})")
        console.print(f"  {escape(hit.entry.text)}")
        if hit.entry.context:
            console.print(f"  [dim]{escape(hit.entry.context)}[/dim]")


@app.command()
def validate(
    inventory: Path = typer.Option(DEFAULT_INVENTORY, "--inventory", "-i"),
) -> None:
    """Validate the inventory file and show a summary."""
    try:
        entries = load_inventory(inventory)
    except InventoryError as e:
        console.print(f"[red]Invalid:[/red] {e}")
        raise typer.Exit(code=1)
    by_kind: dict = {}
    for entry in entries:
        by_kind[entry.kind] = by_kind.get(entry.kind, 0) + 1
    console.print(f"[green]OK[/green] — {len(entries)} entries: "
                  + ", ".join(f"{count} {kind}" for kind, count in sorted(by_kind.items())))
    untagged = [e.entry_id for e in entries if not e.tags]
    if untagged:
        console.print(f"[yellow]Tip:[/yellow] {len(untagged)} entries have no tags "
                      f"(tags improve retrieval): {', '.join(untagged[:5])}"
                      + ("..." if len(untagged) > 5 else ""))


if __name__ == "__main__":
    app()
