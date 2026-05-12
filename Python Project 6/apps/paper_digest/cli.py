from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from apps.paper_digest.agent import build_agent

app = typer.Typer(help="Generate a weekly digest of recent arXiv papers in a category.")
console = Console()

DEFAULT_DIR = Path.home() / ".paper_digest"


@app.command()
def digest(
    category: str = typer.Argument(..., help="arXiv category, e.g. cs.AI, cs.LG, cs.CL."),
    count: int = typer.Option(10, "--count", "-n", help="Number of papers to fetch (max 25)."),
    out_dir: Path = typer.Option(DEFAULT_DIR, "--out", help="Directory for saved digests."),
    quiet: bool = typer.Option(False, "--quiet", "-q"),
) -> None:
    agent = build_agent(digest_dir=out_dir, verbose=not quiet)
    console.print(f"[dim]Category: {category} | Papers: {count} | Output: {out_dir}[/dim]\n")

    result = agent.run(
        f"Produce a weekly digest for arXiv category {category}. "
        f"Fetch about {count} recent papers, then save the digest."
    )

    console.print(Panel(Markdown(result.final_text), title="Digest", border_style="green"))

    stats = Table(show_header=False, title="Run Stats")
    stats.add_row("Iterations", str(result.iterations))
    stats.add_row("Tool calls", str(len(result.tool_calls)))
    stats.add_row("Input tokens", str(result.usage.get("input_tokens", 0)))
    stats.add_row("Output tokens", str(result.usage.get("output_tokens", 0)))
    console.print(stats)


@app.command()
def list_saved(
    out_dir: Path = typer.Option(DEFAULT_DIR, "--out", help="Directory to list."),
) -> None:
    files = sorted(out_dir.glob("digest_*.md"), reverse=True) if out_dir.exists() else []
    if not files:
        console.print("[yellow]No digests saved yet.[/yellow]")
        return
    table = Table(title="Saved Digests")
    table.add_column("File")
    table.add_column("Size")
    for f in files[:50]:
        table.add_row(f.name, f"{f.stat().st_size} B")
    console.print(table)


if __name__ == "__main__":
    app()
