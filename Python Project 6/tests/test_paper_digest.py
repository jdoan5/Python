from __future__ import annotations

from pathlib import Path

import pytest

from apps.paper_digest.tools import _parse_arxiv_feed, build_tools

SAMPLE_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2401.12345v1</id>
    <published>2024-01-15T10:00:00Z</published>
    <title>A Sample Paper on Agents</title>
    <summary>This is a summary of the work
on agents and tool use.</summary>
    <author><name>Alice Researcher</name></author>
    <author><name>Bob Co-author</name></author>
    <category term="cs.AI" />
    <category term="cs.LG" />
  </entry>
</feed>"""


def test_parse_arxiv_feed() -> None:
    papers = _parse_arxiv_feed(SAMPLE_FEED)
    assert len(papers) == 1
    p = papers[0]
    assert p["id"] == "2401.12345v1"
    assert "A Sample Paper" in p["title"]
    assert "Alice Researcher" in p["authors"]
    assert "Bob Co-author" in p["authors"]
    assert "cs.AI" in p["categories"]
    assert p["published"] == "2024-01-15"


def test_fetch_arxiv_rejects_unknown_category(tmp_path: Path) -> None:
    tools = build_tools(tmp_path)
    out, err = tools.execute("fetch_arxiv_recent", {"category": "cs.NONSENSE"})
    assert err
    assert "Unknown category" in out


def test_fetch_arxiv_rejects_out_of_range_count(tmp_path: Path) -> None:
    tools = build_tools(tmp_path)
    out, err = tools.execute(
        "fetch_arxiv_recent", {"category": "cs.AI", "max_results": 999}
    )
    assert err


def test_save_and_list_digests(tmp_path: Path) -> None:
    tools = build_tools(tmp_path)
    out, err = tools.execute(
        "save_digest", {"category": "cs.AI", "content": "# Test\nbody"}
    )
    assert not err
    assert "Saved digest" in out

    out, err = tools.execute("list_digests", {})
    assert not err
    assert "digest_cs-AI_" in out


def test_save_digest_creates_dir(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "subdir"
    tools = build_tools(target)
    assert target.exists()
