from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from apps.job_copilot.agent import build_agent

app = typer.Typer(help="Score a job posting against your resume and draft a cover letter.")
console = Console()

DEFAULT_RESUME = Path.home() / ".job_copilot" / "resume.txt"
DEFAULT_DB = Path.home() / ".job_copilot" / "applications.db"


@app.command()
def apply(
    url: str = typer.Argument(..., help="Job posting URL."),
    resume: Path = typer.Option(DEFAULT_RESUME, "--resume", help="Plain-text resume."),
    db: Path = typer.Option(DEFAULT_DB, "--db", help="SQLite tracker location."),
    quiet: bool = typer.Option(False, "--quiet", "-q"),
) -> None:
    resume.parent.mkdir(parents=True, exist_ok=True)
    db.parent.mkdir(parents=True, exist_ok=True)

    agent = build_agent(resume_path=resume, db_path=db, verbose=not quiet)
    console.print(f"[dim]Posting: {url}\nResume: {resume}\nTracker: {db}[/dim]\n")

    result = agent.run(f"Please evaluate this job posting and draft a cover letter: {url}")

    console.print(Panel(result.final_text, title="Application Draft", border_style="green"))

    stats = Table(show_header=False, title="Run Stats")
    stats.add_row("Iterations", str(result.iterations))
    stats.add_row("Tool calls", str(len(result.tool_calls)))
    stats.add_row("Input tokens", str(result.usage.get("input_tokens", 0)))
    stats.add_row("Output tokens", str(result.usage.get("output_tokens", 0)))
    console.print(stats)


@app.command()
def history(
    db: Path = typer.Option(DEFAULT_DB, "--db", help="SQLite tracker location."),
    status: str = typer.Option("", "--status", help="Filter by status."),
) -> None:
    import sqlite3

    if not db.exists():
        console.print("[red]No tracker yet — run `apply` first.[/red]")
        raise typer.Exit(code=1)

    conn = sqlite3.connect(db)
    query = (
        "SELECT id, company, role, fit_score, status, created_at FROM applications"
        + (" WHERE status = ?" if status else "")
        + " ORDER BY created_at DESC"
    )
    rows = conn.execute(query, (status,) if status else ()).fetchall()
    conn.close()

    if not rows:
        console.print("[yellow]No applications recorded.[/yellow]")
        return

    table = Table(title="Application Tracker")
    for col in ("ID", "Company", "Role", "Fit", "Status", "Created"):
        table.add_column(col)
    for row in rows:
        table.add_row(*(str(c) for c in row))
    console.print(table)


if __name__ == "__main__":
    app()
