from __future__ import annotations

import html
import re
import sqlite3
from contextlib import closing
from pathlib import Path

import httpx

from agent_core.tools import Tool, ToolRegistry

MAX_FETCH_BYTES = 500_000
MAX_RESUME_BYTES = 100_000
DEFAULT_TIMEOUT = 10.0


def _html_to_text(html_str: str) -> str:
    text = re.sub(r"<script\b[^>]*>.*?</script>", "", html_str, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style\b[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _init_db(db_path: Path) -> None:
    with closing(sqlite3.connect(db_path)) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS applications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                company TEXT NOT NULL,
                role TEXT NOT NULL,
                url TEXT,
                fit_score INTEGER,
                status TEXT NOT NULL DEFAULT 'drafted',
                notes TEXT,
                created_at TEXT DEFAULT (datetime('now'))
            )
            """
        )
        conn.commit()


def build_tools(resume_path: Path, db_path: Path) -> ToolRegistry:
    _init_db(db_path)
    registry = ToolRegistry()

    def fetch_url(url: str) -> str:
        if not url.startswith(("http://", "https://")):
            return f"Refusing non-http URL: {url}"
        try:
            with httpx.Client(timeout=DEFAULT_TIMEOUT, follow_redirects=True) as client:
                response = client.get(url, headers={"User-Agent": "JobCopilot/0.1"})
                response.raise_for_status()
                content = response.text[:MAX_FETCH_BYTES]
        except httpx.HTTPError as e:
            return f"Fetch failed: {type(e).__name__}: {e}"
        text = _html_to_text(content)
        return text[:30_000]

    registry.register(
        Tool(
            name="fetch_url",
            description=(
                "Fetch a URL (e.g. a job posting) and return its text content. "
                "HTML is stripped to plain text. Returns up to ~30K characters."
            ),
            input_schema={
                "type": "object",
                "properties": {"url": {"type": "string", "description": "HTTP(S) URL."}},
                "required": ["url"],
            },
            fn=fetch_url,
        )
    )

    def read_resume() -> str:
        if not resume_path.exists():
            return f"Resume not found at {resume_path}. Set the path via --resume."
        if resume_path.stat().st_size > MAX_RESUME_BYTES:
            return "Resume too large; use a plain-text version under 100KB."
        return resume_path.read_text(encoding="utf-8", errors="ignore")

    registry.register(
        Tool(
            name="read_resume",
            description="Read the user's resume (plain text). No arguments.",
            input_schema={"type": "object", "properties": {}, "required": []},
            fn=read_resume,
        )
    )

    def save_application(
        company: str,
        role: str,
        fit_score: int,
        url: str = "",
        notes: str = "",
        status: str = "drafted",
    ) -> str:
        if status not in {"drafted", "applied", "interviewing", "rejected", "offered"}:
            raise ValueError(f"Invalid status: {status}")
        if not 1 <= fit_score <= 10:
            raise ValueError(f"fit_score must be 1-10, got {fit_score}")
        with closing(sqlite3.connect(db_path)) as conn:
            cursor = conn.execute(
                "INSERT INTO applications (company, role, url, fit_score, notes, status) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (company, role, url, fit_score, notes, status),
            )
            conn.commit()
            return f"Saved application #{cursor.lastrowid}: {company} — {role} (fit {fit_score}/10)"

    registry.register(
        Tool(
            name="save_application",
            description=(
                "Record an application in the local SQLite tracker. "
                "fit_score is 1-10. status must be one of: drafted, applied, interviewing, "
                "rejected, offered."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "company": {"type": "string"},
                    "role": {"type": "string"},
                    "fit_score": {"type": "integer", "description": "1 (poor) to 10 (perfect)."},
                    "url": {"type": "string", "description": "Job posting URL."},
                    "notes": {"type": "string", "description": "Brief reasoning."},
                    "status": {"type": "string", "description": "Default 'drafted'."},
                },
                "required": ["company", "role", "fit_score"],
            },
            fn=save_application,
        )
    )

    def list_applications(status: str = "") -> str:
        with closing(sqlite3.connect(db_path)) as conn:
            if status:
                rows = conn.execute(
                    "SELECT id, company, role, fit_score, status, created_at "
                    "FROM applications WHERE status = ? ORDER BY created_at DESC",
                    (status,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT id, company, role, fit_score, status, created_at "
                    "FROM applications ORDER BY created_at DESC LIMIT 50"
                ).fetchall()
        if not rows:
            return "No applications recorded yet."
        return "\n".join(f"#{r[0]} {r[1]} — {r[2]} | fit {r[3]}/10 | {r[4]} | {r[5]}" for r in rows)

    registry.register(
        Tool(
            name="list_applications",
            description="List recent applications from the tracker. Optionally filter by status.",
            input_schema={
                "type": "object",
                "properties": {
                    "status": {"type": "string", "description": "Optional status filter."}
                },
                "required": [],
            },
            fn=list_applications,
        )
    )

    return registry
