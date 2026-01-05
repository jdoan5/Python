from __future__ import annotations

import argparse
import sys

from .config import load_config
from .build import build as build_mart
from .explore import explore as explore_mart
from .export import export_all as export_all_mart, export_csv as export_csv_mart, export_json as export_json_mart
from .quality import run_checks


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m mart",
        description="Mini Data Mart CLI (DuckDB + CSV-driven inputs)",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    p_build = sub.add_parser("build", help="Build/rebuild DuckDB tables from CSV inputs")
    p_build.add_argument("--no-reset", action="store_true", help="Do not drop/recreate tables (incremental experiments)")

    sub.add_parser("explore", help="Run example analytics queries against the mart")

    p_export = sub.add_parser("export", help="Export mart tables to CSV/JSON outputs")
    p_export.add_argument("--format", choices=["csv", "json", "all"], default="all")

    sub.add_parser("quality", help="Run data quality checks (row counts, nulls, RI)")

    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    cfg = load_config()

    if args.cmd == "build":
        counts = build_mart(cfg, reset=(not args.no_reset))
        print("\nBuild complete. Row counts:")
        for k, v in counts.items():
            print(f"  {k}: {v}")
        return 0

    if args.cmd == "explore":
        explore_mart(cfg)
        return 0

    if args.cmd == "export":
        if args.format == "csv":
            export_csv_mart(cfg)
        elif args.format == "json":
            export_json_mart(cfg)
        else:
            export_all_mart(cfg)
        return 0

    if args.cmd == "quality":
        ok = run_checks(cfg)
        return 0 if ok else 2

    raise RuntimeError("Unhandled command")


if __name__ == "__main__":
    raise SystemExit(main())
