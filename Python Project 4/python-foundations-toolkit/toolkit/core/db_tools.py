from __future__ import annotations
from pathlib import Path
import sqlite3
from typing import Dict, Any, List, Tuple


def run_query(db_path: Path, sql: str, max_rows: int = 20) -> Dict[str, Any]:
    """
    Run a read-only query against a SQLite database.

    Returns a dict:
        {
            "db_path": str,
            "columns": [str, ...],
            "rows": [tuple, ...],  # up to max_rows
        }
    """
    if not db_path.is_file():
        raise FileNotFoundError(f"Database file not found: {db_path}")

    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute(sql)
        rows: List[Tuple] = cur.fetchall()
        col_names = [d[0] for d in cur.description] if cur.description else []
    finally:
        conn.close()

    rows = rows[:max_rows]

    return {
        "db_path": str(db_path),
        "columns": col_names,
        "rows": rows,
    }