"""Nothing key-shaped in Stage 4's files: the repository is public."""

from __future__ import annotations

import gzip
import re
from pathlib import Path

STAGE = Path(__file__).resolve().parents[1]
KEY_SHAPED = re.compile(r"sk-ant-[A-Za-z0-9_\-]{8,}")
SKIP_DIRS = {"__pycache__", ".pytest_cache"}
SUFFIXES = (".py", ".md", ".json", ".jsonl", ".yaml", ".yml", ".txt", ".toml", ".html", ".svg", ".gz")


def stage_files():
    for path in sorted(STAGE.rglob("*")):
        if path.is_file() and not SKIP_DIRS & set(path.parts) and path.suffix in SUFFIXES:
            yield path


def test_no_file_in_stage_4_holds_anything_shaped_like_an_api_key():
    found = []
    for path in stage_files():
        data = path.read_bytes()
        if path.suffix == ".gz":
            data = gzip.decompress(data)
        if KEY_SHAPED.search(data.decode("utf-8", errors="replace")):
            found.append(str(path.relative_to(STAGE)))
    assert found == []


def test_the_scan_would_catch_a_key():
    example = "sk-" + "ant-" + "api03-abcdefghijkl"  # assembled here so this file holds no key-shaped string
    assert KEY_SHAPED.search(f"ANTHROPIC_API_KEY={example}")
    assert not KEY_SHAPED.search("not-a-real-key-0000")
