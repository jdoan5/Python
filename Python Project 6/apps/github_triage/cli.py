from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from apps.github_triage.agent import build_agent

app = typer.Typer(help="Triage a GitHub issue against a real codebase.")
console = Console()


@app.command()
def triage(
    issue: str = typer.Argument(..., help="Issue title + body, or path to a .txt file."),
    repo: Path = typer.Option(
        Path.cwd(),
        "--repo",
        "-r",
        help="Path to the target repository to search.",
    ),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Hide intermediate reasoning."),
    max_iters: Optional[int] = typer.Option(None, "--max-iters", help="Override iteration cap."),
) -> None:
    issue_path = Path(issue)
    issue_text = issue_path.read_text() if issue_path.is_file() else issue

    agent = build_agent(repo_root=repo, verbose=not quiet)
    if max_iters is not None:
        agent.max_iterations = max_iters

    console.print(Panel(issue_text, title="Issue", border_style="blue"))
    console.print(f"[dim]Searching against repo: {repo.resolve()}[/dim]\n")

    result = agent.run(issue_text)

    console.print(Panel(result.final_text, title="Triage Result", border_style="green"))

    table = Table(title="Run Stats", show_header=False)
    table.add_row("Iterations", str(result.iterations))
    table.add_row("Tool calls", str(len(result.tool_calls)))
    table.add_row("Input tokens", str(result.usage.get("input_tokens", 0)))
    table.add_row("Output tokens", str(result.usage.get("output_tokens", 0)))
    table.add_row("Cache reads", str(result.usage.get("cache_read_input_tokens", 0)))
    console.print(table)


if __name__ == "__main__":
    app()
