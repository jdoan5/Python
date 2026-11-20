from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from rich.console import Console
from typer.testing import CliRunner

from raglab import cli
from raglab import corpus as corpus_mod
from raglab.corpus import CorpusLock, fingerprint_of, hash_tree
from test_corpus import FakeGit

runner = CliRunner()


@pytest.fixture(autouse=True)
def wide_console(monkeypatch):
    # Rich folds long cells at the default 80 columns; tests assert on whole IDs.
    monkeypatch.setattr(cli, "console", Console(width=250))


@pytest.fixture
def home(tmp_path, monkeypatch, mini_corpus) -> Path:
    """A project home holding a pinned mini corpus, as `corpus fetch` would leave it."""
    root = tmp_path / "home"
    target = root / "data" / "corpus" / "fastapi-0.1.0"
    shutil.copytree(mini_corpus, target)
    files = hash_tree(target)
    CorpusLock("repo", "0.1.0", "c" * 40, fingerprint_of(files), files).save(root / "corpus.lock.json")
    monkeypatch.setenv("RAGLAB_HOME", str(root))
    return root


def run(*args: str):
    return runner.invoke(cli.app, list(args))


def test_verify_passes_on_an_untouched_corpus(home):
    result = run("corpus", "verify")
    assert result.exit_code == 0, result.output
    assert "match the lock" in result.output


def test_verify_fails_loudly_when_a_file_changes(home):
    (home / "data" / "corpus" / "fastapi-0.1.0" / "docs" / "index.md").write_text("tampered", encoding="utf-8")
    result = run("corpus", "verify")
    assert result.exit_code == 1
    assert "modified: docs/index.md" in result.output


def test_stats_explains_every_exclusion(home):
    result = run("corpus", "stats")
    assert result.exit_code == 0, result.output
    assert "release-notes.md" in result.output and "dominate term statistics" in result.output


def test_chunk_reports_all_three_strategies(home):
    result = run("chunk")
    assert result.exit_code == 0, result.output
    for name in ("paragraph", "fixed", "heading"):
        assert name in result.output


def test_search_returns_the_section_that_answers(home):
    result = run("search", "return a 404 with HTTPException", "--k", "3")
    assert result.exit_code == 0, result.output
    assert "tutorial/first-steps#raise-errors" in result.output


def test_compare_runs_every_configuration(home):
    result = run("compare", "interactive API docs")
    assert result.exit_code == 0, result.output
    for name in ("baseline (RAG Mini Chat)", "fixed+bm25", "heading+tfidf", "paragraph+bm25"):
        assert name in result.output


def test_commands_without_a_corpus_explain_how_to_fetch_one(tmp_path, monkeypatch):
    monkeypatch.setenv("RAGLAB_HOME", str(tmp_path))
    result = run("corpus", "stats")
    assert result.exit_code == 1
    assert "corpus fetch" in result.output


def test_fetch_without_a_lock_needs_a_ref(tmp_path, monkeypatch):
    monkeypatch.setenv("RAGLAB_HOME", str(tmp_path))
    result = run("corpus", "fetch")
    assert result.exit_code == 1
    assert "--ref" in result.output


def test_unknown_chunker_is_a_clean_error_not_a_traceback(home):
    result = run("search", "anything", "--chunker", "semantic")
    assert result.exit_code == 1
    assert "unknown chunker" in result.output


def test_fetch_pins_then_reverifies_then_warns_when_the_pin_moves(tmp_path, monkeypatch):
    monkeypatch.setenv("RAGLAB_HOME", str(tmp_path))
    monkeypatch.setattr(corpus_mod, "_git", FakeGit())

    first = run("corpus", "fetch", "--ref", "0.1.0")
    assert first.exit_code == 0, first.output
    assert "Pinned" in first.output and (tmp_path / "corpus.lock.json").is_file()

    again = run("corpus", "fetch")  # no --ref: re-fetch whatever the lock pins
    assert again.exit_code == 0, again.output
    assert "matches the lock" in again.output

    moved = run("corpus", "fetch", "--ref", "0.2.0")
    assert moved.exit_code == 0, moved.output
    assert "pin moved from 0.1.0" in moved.output
    assert CorpusLock.load(tmp_path / "corpus.lock.json").ref == "0.2.0"


# ---------------------------------------------------------------- regressions from the Stage 1 review


@pytest.mark.parametrize("content", ['{"repo": "r"', "[]", '{"repo": "r", "ref": "0.1.0"}'])
def test_a_corrupt_lock_is_a_clean_error_not_a_traceback(home, content):
    (home / "corpus.lock.json").write_text(content, encoding="utf-8")
    result = run("corpus", "stats")
    assert result.exit_code == 1
    assert "Error:" in result.output and "Traceback" not in result.output


def test_rich_markup_in_a_ref_cannot_crash_the_cli(tmp_path, monkeypatch):
    monkeypatch.setenv("RAGLAB_HOME", str(tmp_path))
    result = run("corpus", "fetch", "--ref", "v1[/]")
    assert result.exit_code == 1
    assert "not a release tag" in result.output


def test_every_command_verifies_the_corpus_before_using_it(home):
    # An edited page (or an iCloud " 2" copy) must stop the run, not shift its numbers.
    (home / "data" / "corpus" / "fastapi-0.1.0" / "docs" / "index.md").write_text("# Edited\n", encoding="utf-8")
    result = run("search", "anything")
    assert result.exit_code == 1
    assert "does not match corpus.lock.json" in result.output
