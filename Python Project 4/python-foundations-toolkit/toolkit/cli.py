from __future__ import annotations
import argparse
from pathlib import Path
import textwrap
import logging

from .core.csv_tools import summarize_csv
from .core.text_tools import summarize_text
from .core.db_tools import run_query

logger = logging.getLogger(__name__)


def configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )


def _csv_summary_cmd(args: argparse.Namespace) -> None:
    path = Path(args.path)
    summary = summarize_csv(path, delimiter=args.delimiter, max_preview_rows=args.preview)
    print(f"CSV summary for {summary['path']}")
    print(f"  Columns ({len(summary['columns'])}): {', '.join(summary['columns'])}")
    print(f"  Rows (excluding header): {summary['num_rows']}")
    if summary["preview_rows"]:
        print()
        print(f"Preview (first {len(summary['preview_rows'])} rows):")
        for row in summary["preview_rows"]:
            print("  " + " | ".join(row))


def _text_summary_cmd(args: argparse.Namespace) -> None:
    path = Path(args.path)
    summary = summarize_text(path, top_n=args.top)
    print(f"Text summary for {summary['path']}")
    print(f"  Lines: {summary['num_lines']}")
    print(f"  Characters: {summary['num_chars']}")
    print(f"  Words: {summary['num_words']}")
    print()
    print(f"Top {len(summary['top_words'])} words:")
    for word, count in summary["top_words"]:
        print(f"  {word}: {count}")


def _db_query_cmd(args: argparse.Namespace) -> None:
    db_path = Path(args.db)
    result = run_query(db_path, args.sql, max_rows=args.max_rows)
    print(f"DB query on {result['db_path']}")
    if not result["columns"]:
        print("  (No result columns; query might not be a SELECT.)")
        return

    # Print header
    header = " | ".join(result["columns"])
    print(header)
    print("-" * len(header))

    for row in result["rows"]:
        print(" | ".join(str(v) for v in row))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python-foundations-toolkit",
        description="Small collection of Python foundation tools: CSV, text, and SQLite helpers.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """Examples:
  python -m toolkit.cli csv-summary data/sample.csv
  python -m toolkit.cli text-summary data/sample.txt --top 10
  python -m toolkit.cli db-query data/demo.db "SELECT * FROM items LIMIT 5;"
            """
        ),
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # csv-summary
    p_csv = subparsers.add_parser(
        "csv-summary",
        help="Summarize a CSV file (columns, row count, small preview).",
    )
    p_csv.add_argument("path", help="Path to CSV file.")
    p_csv.add_argument(
        "--delimiter", "-d",
        default=",",
        help="CSV delimiter (default: ',').",
    )
    p_csv.add_argument(
        "--preview",
        type=int,
        default=5,
        help="Number of preview rows to show (default: 5).",
    )
    p_csv.set_defaults(func=_csv_summary_cmd)

    # text-summary
    p_text = subparsers.add_parser(
        "text-summary",
        help="Summarize a text file (line/word counts, top words).",
    )
    p_text.add_argument("path", help="Path to text file.")
    p_text.add_argument(
        "--top",
        type=int,
        default=5,
        help="How many top words to show (default: 5).",
    )
    p_text.set_defaults(func=_text_summary_cmd)

    # db-query
    p_db = subparsers.add_parser(
        "db-query",
        help="Run a SELECT query against a SQLite database.",
    )
    p_db.add_argument("db", help="Path to SQLite .db file.")
    p_db.add_argument("sql", help="SQL query to run (wrap in quotes).")
    p_db.add_argument(
        "--max-rows",
        type=int,
        default=20,
        help="Maximum number of rows to display (default: 20).",
    )
    p_db.set_defaults(func=_db_query_cmd)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(verbose=args.verbose)
    logger.debug("Arguments parsed: %s", args)
    args.func(args)


if __name__ == "__main__":
    main()