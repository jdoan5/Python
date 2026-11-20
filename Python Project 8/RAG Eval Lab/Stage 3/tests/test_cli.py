from __future__ import annotations

import json
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


# ---------------------------------------------------------------- Stage 2: golden set and evaluation

RAISE = "tutorial/first-steps#raise-errors"
DOCS = "tutorial/first-steps#interactive-api-docs"


@pytest.fixture
def golden_home(home) -> Path:
    """The mini project home plus a golden set labelled against its corpus."""
    from raglab.golden import dump_golden

    lock = CorpusLock.load(home / "corpus.lock.json")
    (home / "golden").mkdir()
    dump_golden({
        "version": 1,
        "corpus": {"ref": lock.ref, "fingerprint": lock.fingerprint},
        "questions": [
            {"id": "q001", "question": "How do I return a 404 when an item is not found?", "style": "paraphrase",
             "target": RAISE, "judgments": {RAISE: 2}},
            {"id": "q002", "question": "Where is the Swagger UI docs page?", "style": "keyword",
             "target": DOCS, "judgments": {DOCS: 2}},
        ],
    }, home / "golden" / "v2.yaml")
    return home


def test_golden_validate_summarises_a_good_set(golden_home):
    result = run("golden", "validate")
    assert result.exit_code == 0, result.output
    assert "2 questions" in result.output and "questions fingerprint" in result.output


def test_golden_validate_rejects_labels_for_another_corpus(golden_home):
    path = golden_home / "golden" / "v2.yaml"
    fingerprint = CorpusLock.load(golden_home / "corpus.lock.json").fingerprint
    path.write_text(path.read_text(encoding="utf-8").replace(fingerprint, "f" * 64), encoding="utf-8")
    result = run("golden", "validate")
    assert result.exit_code == 1
    assert "re-judge" in result.output


def test_golden_show_prints_a_section_body(golden_home):
    result = run("golden", "show", RAISE)
    assert result.exit_code == 0, result.output
    assert "HTTPException" in result.output and "truncated" not in result.output
    assert run("golden", "show", "tutorial/first-steps#nope").exit_code == 1


def test_golden_show_says_plainly_when_it_truncates(golden_home):
    # A judge once read a cut-off section as broken data; truncation must be explicit.
    short = run("golden", "show", RAISE, "--chars", "100")  # a 70-char body: nothing to cut
    assert short.exit_code == 0 and "truncated" not in short.output
    cut = run("golden", "show", "tutorial/first-steps#first-steps", "--chars", "100")
    assert "[truncated: 118 more characters; omit --chars to see all]" in cut.output


def test_golden_pool_gates_on_unjudged_sections_until_they_are_merged(golden_home, tmp_path):
    sheet = tmp_path / "pool.json"
    first = run("golden", "pool", "--depth", "3", "--worksheet", str(sheet))
    assert first.exit_code == 1, first.output
    assert "unjudged" in first.output
    items = json.loads(sheet.read_text(encoding="utf-8"))
    # Judges stay blind: a candidate carries its location, heading path and text, and nothing else.
    assert all(set(c) == {"location", "heading_path", "text"} for item in items for c in item["candidates"])
    # ...and the text is the whole section: excerpts made judges under-grade long sections.
    from raglab.corpus import load_corpus
    from raglab.sections import SectionIndex

    docs = {d.doc_id: d for d in load_corpus(golden_home / "data" / "corpus" / "fastapi-0.1.0").documents}
    index = SectionIndex(docs.values())
    for item in items:
        for c in item["candidates"]:
            ref = index.get(c["location"])
            assert c["text"] == docs[ref.doc_id].text[ref.start:ref.end]
    judged = {item["id"]: {c["location"]: 0 for c in item["candidates"]} for item in items}
    (tmp_path / "judged.json").write_text(json.dumps(judged), encoding="utf-8")
    merged = run("golden", "merge", str(tmp_path / "judged.json"))
    assert merged.exit_code == 0, merged.output
    again = run("golden", "pool", "--depth", "3")
    assert again.exit_code == 0, again.output
    assert "Complete" in again.output


def test_golden_merge_refuses_unknown_sections_and_regrades(golden_home, tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"q001": {"tutorial/first-steps#nope": 1}}), encoding="utf-8")
    assert "not in the corpus" in run("golden", "merge", str(bad)).output
    regrade = tmp_path / "regrade.json"
    regrade.write_text(json.dumps({"q001": {RAISE: 1}}), encoding="utf-8")
    result = run("golden", "merge", str(regrade))
    assert result.exit_code == 1 and "already judged 2" in result.output


def test_golden_sample_writes_a_worksheet(golden_home, tmp_path):
    out = tmp_path / "sheet.md"
    result = run("golden", "sample", "--n", "1", "--out", str(out))  # the mini corpus has one eligible section
    assert result.exit_code == 0, result.output
    assert out.read_text(encoding="utf-8").count("\n## s") == 1
    assert "can sample between 1 and 1" in run("golden", "sample", "--n", "2", "--out", str(out)).output


def test_eval_run_saves_a_run_that_report_can_read(golden_home, tmp_path):
    out = tmp_path / "run"
    result = run("eval", "run", "--out", str(out), "--resamples", "1000")
    assert result.exit_code == 0, result.output
    assert (out / "manifest.json").is_file() and (out / "questions.jsonl").is_file()
    assert "baseline (RAG Mini Chat)" in result.output
    again = run("eval", "run", "--out", str(out), "--resamples", "1000")
    assert again.exit_code == 1 and "already holds a run" in again.output
    report = run("eval", "report", str(out))
    assert report.exit_code == 0, report.output
    assert "Paired difference" in report.output
    markdown = run("eval", "report", str(out), "--markdown")
    assert markdown.exit_code == 0, markdown.output
    for heading in ("### Ranking quality", "### What a 4,800-character prompt receives",
                    "### Against the baseline: ranking", "### What each switch is worth: the prompt"):
        assert heading in markdown.output
    assert "| **baseline (RAG Mini Chat)** |" in markdown.output



def test_cli_errors_are_clean_messages_not_tracebacks(golden_home, tmp_path):
    missing_dir = run("golden", "sample", "--n", "1", "--out", str(tmp_path / "no" / "such" / "sheet.md"))
    assert missing_dir.exit_code == 1 and "Cannot write" in missing_dir.output
    dup = tmp_path / "dup.json"
    dup.write_text('{"q001": {"index#home": 1}, "q001": {"index#home": 0}}', encoding="utf-8")
    merged = run("golden", "merge", str(dup))
    assert merged.exit_code == 1 and "duplicate key" in merged.output


@pytest.mark.parametrize(
    ("manifest", "summary"),
    [("[]", "{}"), ('{"schema": 2}', "{}"), ('{"schema": 1, "run_id": "x"}', "{}"), ("not json", "{}")],
)
def test_report_on_a_broken_run_is_a_clean_error(tmp_path, manifest, summary):
    (tmp_path / "manifest.json").write_text(manifest, encoding="utf-8")
    (tmp_path / "summary.json").write_text(summary, encoding="utf-8")
    for args in (("eval", "report", str(tmp_path)), ("eval", "report", str(tmp_path), "--markdown")):
        result = run(*args)
        assert result.exit_code == 1 and "Error:" in result.output, result.output
