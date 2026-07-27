from __future__ import annotations

from pathlib import Path

import pytest

from job_agent.retrieval import (
    Bm25Index,
    InventoryEntry,
    InventoryError,
    load_inventory,
    tokenize,
)


def make_entries() -> list:
    return [
        InventoryEntry("py-api", "project", "Built a FastAPI REST service in Python.",
                       tags=["python", "fastapi", "api"]),
        InventoryEntry("node-ats", "project", "Node.js ATS resume parsing pipeline.",
                       tags=["node", "javascript", "ats"]),
        InventoryEntry("ml-churn", "project", "Churn classifier with scikit-learn TF-IDF.",
                       tags=["ml", "sklearn"]),
    ]


def test_tokenize_keeps_tech_tokens() -> None:
    tokens = tokenize("Expert in C++ and .NET")
    assert "c++" in tokens
    # Leading punctuation is stripped — ".NET" and "NET" both become "net",
    # so query and document tokenize consistently and still match.
    assert "net" in tokens
    assert tokenize("") == []
    assert tokenize("!!! ???") == []


def test_tokenize_strips_sentence_punctuation() -> None:
    # A sentence-final "Python." must match the query "python".
    assert tokenize("Built pipelines in Python.") == ["built", "pipelines", "in", "python"]
    assert tokenize("c++") == ["c++"]  # trailing '+' survives
    assert tokenize("c#") == ["c#"]    # trailing '#' survives


def test_sentence_final_word_is_searchable() -> None:
    entries = [InventoryEntry("e1", "skill", "Built ETL pipelines in Python.")]
    hits = Bm25Index(entries).search("python")
    assert hits and hits[0].entry.entry_id == "e1"


def test_search_ranks_relevant_entry_first() -> None:
    index = Bm25Index(make_entries())
    hits = index.search("python fastapi service")
    assert hits, "expected at least one hit"
    assert hits[0].entry.entry_id == "py-api"


def test_search_uses_tags() -> None:
    index = Bm25Index(make_entries())
    hits = index.search("javascript")
    assert hits and hits[0].entry.entry_id == "node-ats"


def test_search_no_match_returns_empty() -> None:
    index = Bm25Index(make_entries())
    assert index.search("kubernetes helm istio") == []


def test_search_empty_query() -> None:
    index = Bm25Index(make_entries())
    assert index.search("   ") == []


def test_empty_index_raises() -> None:
    with pytest.raises(InventoryError):
        Bm25Index([])


def test_load_inventory_roundtrip(tmp_path: Path) -> None:
    f = tmp_path / "inv.yaml"
    f.write_text(
        "entries:\n"
        "  - id: a1\n    kind: skill\n    text: Python testing with pytest.\n"
        "    tags: [python, pytest]\n"
        "  - id: a2\n    text: SQL reporting queries.\n",
        encoding="utf-8",
    )
    entries = load_inventory(f)
    assert len(entries) == 2
    assert entries[0].entry_id == "a1"
    assert entries[1].kind == "achievement"  # default kind


def test_load_inventory_duplicate_ids(tmp_path: Path) -> None:
    f = tmp_path / "inv.yaml"
    f.write_text(
        "entries:\n"
        "  - id: dup\n    text: one\n"
        "  - id: dup\n    text: two\n",
        encoding="utf-8",
    )
    with pytest.raises(InventoryError, match="Duplicate"):
        load_inventory(f)


def test_load_inventory_missing_file(tmp_path: Path) -> None:
    with pytest.raises(InventoryError, match="not found"):
        load_inventory(tmp_path / "nope.yaml")


def test_load_inventory_missing_text(tmp_path: Path) -> None:
    f = tmp_path / "inv.yaml"
    f.write_text("entries:\n  - id: x\n", encoding="utf-8")
    with pytest.raises(InventoryError, match="needs both"):
        load_inventory(f)


def test_load_inventory_null_entries(tmp_path: Path) -> None:
    # 'entries:' present but null must raise InventoryError, not TypeError.
    f = tmp_path / "inv.yaml"
    f.write_text("entries:\n", encoding="utf-8")
    with pytest.raises(InventoryError, match="entries"):
        load_inventory(f)


def test_load_inventory_non_list_entries(tmp_path: Path) -> None:
    f = tmp_path / "inv.yaml"
    f.write_text("entries: 5\n", encoding="utf-8")
    with pytest.raises(InventoryError):
        load_inventory(f)


def test_load_inventory_empty_list(tmp_path: Path) -> None:
    f = tmp_path / "inv.yaml"
    f.write_text("entries: []\n", encoding="utf-8")
    with pytest.raises(InventoryError, match="empty"):
        load_inventory(f)


def test_load_inventory_null_tags(tmp_path: Path) -> None:
    # 'tags:' present but null must be treated as no tags, not crash.
    f = tmp_path / "inv.yaml"
    f.write_text("entries:\n  - id: x\n    text: something\n    tags:\n", encoding="utf-8")
    entries = load_inventory(f)
    assert entries[0].tags == []
