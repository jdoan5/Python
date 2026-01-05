#!/usr/bin/env python3
"""
inspect_duckdb.py
Lightweight DuckDB inspector for the Mini Data Mart (no pandas required).

Examples:
  python inspect_duckdb.py
  python inspect_duckdb.py --db mini_data_mart.duckdb
  python inspect_duckdb.py --limit 10 --no-samples
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import duckdb


DEFAULT_TABLES: tuple[str, ...] = ("dim_customer", "dim_product", "dim_date", "fact_sales")


def _format_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    """Simple aligned table printer (no external deps)."""
    rows_str = [[("" if v is None else str(v)) for v in r] for r in rows]
    widths = [len(h) for h in headers]
    for r in rows_str:
        for i, cell in enumerate(r):
            widths[i] = max(widths[i], len(cell))

    def fmt_row(r: Sequence[str]) -> str:
        return " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(r))

    sep = "-+-".join("-" * w for w in widths)
    out = [fmt_row(list(headers)), sep]
    out.extend(fmt_row(r) for r in rows_str)
    return "\n".join(out)


def _safe_exists(con: duckdb.DuckDBPyConnection, table: str) -> bool:
    q = """
        SELECT COUNT(*) 
        FROM information_schema.tables
        WHERE table_schema = 'main' AND table_name = ?;
    """
    return con.execute(q, [table]).fetchone()[0] == 1


def show_tables(con: duckdb.DuckDBPyConnection) -> List[str]:
    rows = con.execute("SHOW TABLES;").fetchall()
    tables = [r[0] for r in rows]
    print("=== TABLES ===")
    if not tables:
        print("(no tables found)")
    else:
        print("\n".join(f"- {t}" for t in tables))
    return tables


def show_counts(con: duckdb.DuckDBPyConnection, tables: Iterable[str]) -> None:
    print("\n=== ROW COUNTS ===")
    out_rows: List[Tuple[str, object]] = []
    for t in tables:
        if not _safe_exists(con, t):
            out_rows.append((t, "MISSING"))
            continue
        n = con.execute(f"SELECT COUNT(*) FROM {t};").fetchone()[0]
        out_rows.append((t, n))
    print(_format_table(["table", "rows"], out_rows))


def show_schema(con: duckdb.DuckDBPyConnection, tables: Iterable[str]) -> None:
    print("\n=== SCHEMA ===")
    for t in tables:
        print(f"\n-- {t}")
        if not _safe_exists(con, t):
            print("(missing)")
            continue
        # PRAGMA table_info returns: cid, name, type, notnull, dflt_value, pk
        rows = con.execute(f"PRAGMA table_info('{t}');").fetchall()
        headers = ["cid", "name", "type", "notnull", "default", "pk"]
        print(_format_table(headers, rows))


def show_samples(con: duckdb.DuckDBPyConnection, tables: Iterable[str], limit: int) -> None:
    print("\n=== SAMPLE ROWS ===")
    for t in tables:
        print(f"\n-- {t} (LIMIT {limit})")
        if not _safe_exists(con, t):
            print("(missing)")
            continue
        cur = con.execute(f"SELECT * FROM {t} LIMIT ?;", [limit])
        rows = cur.fetchall()
        headers = [d[0] for d in cur.description] if cur.description else []
        if not rows:
            print("(no rows)")
        else:
            print(_format_table(headers, rows))


def _default_db_path() -> Path:
    """
    Prefer MART_DB_PATH if set; otherwise use ./mini_data_mart.duckdb
    """
    env = Path(str(Path.cwd() / "mini_data_mart.duckdb"))
    return Path(env)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inspect a DuckDB file (mini data mart).")
    p.add_argument("--db", type=str, default="mini_data_mart.duckdb", help="Path to .duckdb file")
    p.add_argument("--limit", type=int, default=15, help="Rows to show per table")
    p.add_argument("--tables", type=str, default=",".join(DEFAULT_TABLES), help="Comma-separated table list")
    p.add_argument("--no-counts", action="store_true", help="Skip row counts")
    p.add_argument("--no-schema", action="store_true", help="Skip schema output")
    p.add_argument("--no-samples", action="store_true", help="Skip sample rows")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    db_path = Path(args.db).expanduser().resolve()
    tables = [t.strip() for t in args.tables.split(",") if t.strip()]

    if not db_path.exists():
        print(f"ERROR: DuckDB file not found: {db_path}")
        print("Tip: run your build step first, or pass --db /path/to/mini_data_mart.duckdb")
        return 2

    print(f"Opening DuckDB: {db_path}")
    con = duckdb.connect(db_path.as_posix())
    try:
        existing = show_tables(con)

        # If user passed a default list but DB has different tables, still proceed safely.
        targets = tables or existing

        if not args.no_counts:
            show_counts(con, targets)
        if not args.no_schema:
            show_schema(con, targets)
        if not args.no_samples:
            show_samples(con, targets, args.limit)

        return 0
    finally:
        con.close()


if __name__ == "__main__":
    raise SystemExit(main())