"""Simple, dependency-free Python CLI.

Usage examples:
  - As a module:  python -m cli_app greet Alice
  - Installed script: mycli greet Alice
  - With verbosity:   mycli -v sum 1 2 3
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from typing import List

from . import __version__

LOG = logging.getLogger("cli_app")

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mycli",
        description="A clean Python CLI starter with subcommands and logging."
    )
    parser.add_argument(
        "-v", "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (-v=INFO, -vv=DEBUG)."
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # greet
    p_greet = sub.add_parser("greet", help="Say hello to someone.")
    p_greet.add_argument("name", help="Name to greet.")
    p_greet.add_argument("--yell", action="store_true", help="Shout the greeting.")
    p_greet.set_defaults(func=cmd_greet)

    # sum
    p_sum = sub.add_parser("sum", help="Sum a list of numbers.")
    p_sum.add_argument("numbers", nargs="+", type=float, help="Numbers to add.")
    p_sum.set_defaults(func=cmd_sum)

    # version
    p_ver = sub.add_parser("version", help="Show CLI version.")
    p_ver.add_argument("--json", action="store_true", help="Output JSON.")
    p_ver.set_defaults(func=cmd_version)

    return parser

def configure_logging(verbosity: int) -> None:
    # Map -v to INFO and -vv to DEBUG; default WARNING
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG

    logging.basicConfig(
        level=level,
        format="%(levelname)s: %(message)s"
    )

def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.verbose)

    LOG.debug("Parsed args: %s", args)

    try:
        return args.func(args)
    except KeyboardInterrupt:
        LOG.error("Interrupted.")
        return 130

# --- command handlers ---

def cmd_greet(args: argparse.Namespace) -> int:
    msg = f"Hello, {args.name}!"
    if args.yell:
        msg = msg.upper()
    print(msg)
    LOG.info("Greeted %s", args.name)
    return 0

def cmd_sum(args: argparse.Namespace) -> int:
    total = sum(args.numbers)
    print(total)
    LOG.debug("Summed %s -> %s", args.numbers, total)
    return 0

def cmd_version(args: argparse.Namespace) -> int:
    if args.json:
        import json
        print(json.dumps({"version": __version__}))
    else:
        print(__version__)
    return 0
