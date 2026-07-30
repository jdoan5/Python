"""Stage 3 — application tracker.

Ingests the JSON sidecars that Stage 1's save_draft() writes into
`Stage 1/output/` and maintains a status lifecycle for each application in
SQLite. Deliberately stdlib-only (sqlite3, json, pathlib): no coupling to the
job_agent package — the sidecar files on disk are the integration contract,
which is why Stages 1 and 2 needed zero changes for tracking to exist.

Deploy-friendly: every path is injectable (env var or argument), nothing is
hardcoded to this machine.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

STAGE1_OUTPUT = Path(__file__).resolve().parent.parent / "Stage 1" / "output"
DEFAULT_OUTPUT_DIR = Path(os.environ.get("JOB_TRACKER_OUTPUT_DIR", str(STAGE1_OUTPUT)))
DEFAULT_DB = Path(os.environ.get("JOB_TRACKER_DB",
                                 str(Path(__file__).resolve().parent / "tracker.db")))

VALID_STATUSES = ["drafted", "applied", "interviewing", "offer", "rejected", "withdrawn"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS applications (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sidecar_path TEXT NOT NULL UNIQUE,
    company TEXT NOT NULL DEFAULT 'Unknown',
    role TEXT NOT NULL DEFAULT 'Unknown',
    seniority TEXT NOT NULL DEFAULT 'unknown',
    source TEXT NOT NULL DEFAULT '',
    fit_score INTEGER,
    gaps TEXT NOT NULL DEFAULT '[]',
    bullet_count INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'drafted',
    notes TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


class TrackerError(Exception):
    """Raised for invalid tracker operations (bad status, unknown id, ...)."""


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute(_SCHEMA)
    return conn


# save_draft filenames start with "YYYY-MM-DD_HHMMSS[_ffffff]_Company"
# (microseconds were added later, so the middle group is optional).
_STAMP_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})_(\d{6})(?:_(\d{6}))?_")


def _created_at_for(path: Path) -> str:
    """Draft creation time: parse the save_draft filename stamp, else mtime.

    The filename stamp survives file copies/restores; mtime does not.
    """
    match = _STAMP_RE.match(path.name)
    if match:
        date_part, time_part, micro = match.group(1), match.group(2), match.group(3) or "0"
        try:
            stamp = datetime.strptime(f"{date_part} {time_part} {micro}", "%Y-%m-%d %H%M%S %f")
            return stamp.isoformat(timespec="seconds")
        except ValueError:
            pass
    try:
        return datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")
    except OSError:
        return datetime.now().isoformat(timespec="seconds")


def _coerce_fit(fit: object) -> Optional[int]:
    """Guarded fit_score conversion: reject bools, round floats, survive NaN/Inf."""
    if isinstance(fit, bool) or not isinstance(fit, (int, float)):
        return None
    try:
        return round(fit)
    except (ValueError, OverflowError):  # NaN / Infinity
        return None


def parse_sidecar(path: Path) -> Optional[Dict]:
    """Extract tracker-relevant fields from one sidecar. None if unusable."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    # Real sidecars (save_draft) always carry both keys; anything else is a
    # stray JSON file that must not pollute the tracker as "Unknown/Unknown".
    if "requirements" not in data or "draft" not in data:
        return None
    requirements = data.get("requirements") or {}
    draft = data.get("draft") or {}
    if not isinstance(requirements, dict) or not isinstance(draft, dict):
        return None

    gaps = draft.get("gaps") or []
    if not isinstance(gaps, list):
        gaps = []
    bullets = draft.get("bullets") or []

    return {
        "sidecar_path": str(path.resolve()),
        "company": str(requirements.get("company") or "Unknown"),
        "role": str(requirements.get("role") or "Unknown"),
        "seniority": str(requirements.get("seniority") or "unknown"),
        "source": str(data.get("source") or ""),
        "fit_score": _coerce_fit(draft.get("fit_score")),
        "gaps": json.dumps([str(g) for g in gaps]),
        "bullet_count": len(bullets) if isinstance(bullets, list) else 0,
        "created_at": _created_at_for(path),
    }


def sync_output_dir(
    db_path: Path = DEFAULT_DB,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> Tuple[int, List[str]]:
    """Scan output_dir for sidecars and insert any not yet tracked.

    Returns (newly_added_count, list of unreadable sidecar names).
    Re-running is idempotent: sidecar_path is UNIQUE and existing rows —
    including their user-edited status/notes — are never touched.
    """
    added = 0
    skipped: List[str] = []
    if not output_dir.is_dir():
        return 0, []
    with closing(_connect(db_path)) as conn:
        for sidecar in sorted(output_dir.glob("*.json")):
            try:
                parsed = parse_sidecar(sidecar)
            except Exception:  # defense in depth: one bad file must never kill the sync
                parsed = None
            if parsed is None:
                skipped.append(sidecar.name)
                continue
            now = datetime.now().isoformat(timespec="seconds")
            cursor = conn.execute(
                """INSERT OR IGNORE INTO applications
                   (sidecar_path, company, role, seniority, source, fit_score,
                    gaps, bullet_count, created_at, updated_at)
                   VALUES (:sidecar_path, :company, :role, :seniority, :source,
                           :fit_score, :gaps, :bullet_count, :created_at, :updated_at)""",
                {**parsed, "updated_at": now},
            )
            added += cursor.rowcount
        conn.commit()
    return added, skipped


def load_applications(db_path: Path = DEFAULT_DB) -> List[Dict]:
    """All applications, newest first, gaps decoded to lists."""
    with closing(_connect(db_path)) as conn:
        rows = conn.execute(
            "SELECT * FROM applications ORDER BY created_at DESC, id DESC"
        ).fetchall()
    result = []
    for row in rows:
        item = dict(row)
        try:
            item["gaps"] = json.loads(item.get("gaps") or "[]")
        except json.JSONDecodeError:
            item["gaps"] = []
        result.append(item)
    return result


def update_application(
    app_id: int,
    status: Optional[str] = None,
    notes: Optional[str] = None,
    db_path: Path = DEFAULT_DB,
) -> None:
    """Update status and/or notes for one application."""
    if status is not None and status not in VALID_STATUSES:
        raise TrackerError(f"Invalid status {status!r}; expected one of {VALID_STATUSES}")
    sets, params = [], {"id": app_id}
    if status is not None:
        sets.append("status = :status")
        params["status"] = status
    if notes is not None:
        sets.append("notes = :notes")
        params["notes"] = notes
    if not sets:
        return
    sets.append("updated_at = :updated_at")
    params["updated_at"] = datetime.now().isoformat(timespec="seconds")
    with closing(_connect(db_path)) as conn:
        cursor = conn.execute(
            f"UPDATE applications SET {', '.join(sets)} WHERE id = :id", params
        )
        if cursor.rowcount == 0:
            raise TrackerError(f"No application with id {app_id}")
        conn.commit()


def gap_frequencies(applications: List[Dict], top_n: int = 15) -> List[Tuple[str, int]]:
    """Most common gap texts across all tracked applications.

    The 'what should I learn next' signal: gaps that keep appearing across
    postings are the highest-leverage skills to close.
    """
    counts: Dict[str, int] = {}
    for app in applications:
        # Dedupe within one application so the count means "N postings asked
        # for this", not "mentioned N times across drafts".
        unique_gaps = {
            " ".join(str(gap).lower().split())
            for gap in app.get("gaps", [])
        }
        for key in unique_gaps:
            if key:
                counts[key] = counts.get(key, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return ranked[:top_n]
