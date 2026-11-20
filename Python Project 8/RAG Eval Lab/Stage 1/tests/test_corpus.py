from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Callable

import pytest

from conftest import make_doc
from raglab import corpus as corpus_mod
from raglab.corpus import (
    CorpusError,
    CorpusLock,
    fetch_corpus,
    fingerprint_of,
    hash_tree,
    load_corpus,
    make_include_resolver,
    verify_corpus,
)
from raglab.markdown import fence_closes, fence_open

SHA = "c0ffee" * 6 + "abcd"  # 40 hex chars, shaped like a git commit


def lock_for(root: Path, ref: str = "0.1.0", commit: str = SHA) -> CorpusLock:
    files = hash_tree(root)
    return CorpusLock("repo", ref, commit, fingerprint_of(files), files)


# ---------------------------------------------------------------- lock + verify


def test_fingerprint_is_order_independent_and_content_sensitive():
    a = {"docs/a.md": "1", "docs/b.md": "2"}
    assert fingerprint_of(a) == fingerprint_of(dict(reversed(list(a.items()))))
    assert fingerprint_of(a) != fingerprint_of({"docs/a.md": "1", "docs/b.md": "3"})
    assert fingerprint_of(a) != fingerprint_of({"docs/a.md": "1", "docs/c.md": "2"})


def test_lock_round_trips_through_json(tmp_path: Path):
    files = {"docs/a.md": "h"}
    lock = CorpusLock("repo", "0.1.0", SHA, fingerprint_of(files), files, ["x.py"])
    lock.save(tmp_path / "lock.json")
    assert CorpusLock.load(tmp_path / "lock.json") == lock


def test_the_lock_is_written_with_unix_newlines_on_every_os(tmp_path: Path):
    files = {"docs/a.md": "h"}
    CorpusLock("repo", "0.1.0", SHA, fingerprint_of(files), files).save(tmp_path / "lock.json")
    assert b"\r" not in (tmp_path / "lock.json").read_bytes()


def test_loading_a_missing_lock_explains_how_to_create_one(tmp_path: Path):
    with pytest.raises(CorpusError, match="corpus fetch"):
        CorpusLock.load(tmp_path / "nope.json")


def _valid_lock_data() -> dict:
    files = {"docs/a.md": "h"}
    return {"repo": "r", "ref": "0.1.0", "commit": SHA, "fingerprint": fingerprint_of(files),
            "files": files, "missing_includes": [], "file_count": 1}


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda d: "{ truncated", "not valid JSON"),
        (lambda d: [], "JSON object"),
        (lambda d: {k: v for k, v in d.items() if k != "files"}, "malformed"),
        (lambda d: {**d, "surprise": 1}, "malformed"),
        (lambda d: {**d, "files": ["docs/a.md"]}, "'files' must map"),
        (lambda d: {**d, "ref": "../escape"}, "not a release tag"),
        (lambda d: {**d, "commit": "[red]abc[/red]"}, "full git commit SHA"),
        (lambda d: {**d, "fingerprint": "0" * 64}, "internally inconsistent"),
        (lambda d: {**d, "file_count": 999}, "internally inconsistent"),
    ],
)
def test_a_damaged_lock_is_a_clear_corpus_error(tmp_path: Path, mutate: Callable, message: str):
    data = mutate(_valid_lock_data())
    (tmp_path / "lock.json").write_text(data if isinstance(data, str) else json.dumps(data), encoding="utf-8")
    with pytest.raises(CorpusError, match=message):
        CorpusLock.load(tmp_path / "lock.json")


def test_verify_reports_missing_unexpected_and_modified_files(mini_corpus: Path):
    lock = lock_for(mini_corpus)
    assert verify_corpus(mini_corpus, lock) == []

    (mini_corpus / "docs" / "index.md").write_text("# Changed\n", encoding="utf-8")
    (mini_corpus / "docs" / "new.md").write_text("# New\n", encoding="utf-8")
    (mini_corpus / "docs" / "release-notes.md").unlink()
    assert verify_corpus(mini_corpus, lock) == [
        "missing: docs/release-notes.md",
        "unexpected: docs/new.md",
        "modified: docs/index.md",
    ]


def test_verify_ignores_files_the_os_drops_into_folders(mini_corpus: Path):
    lock = lock_for(mini_corpus)
    (mini_corpus / "docs" / ".DS_Store").write_bytes(b"\x00\x01finder")
    assert verify_corpus(mini_corpus, lock) == []


# ---------------------------------------------------------------- fetch (fake git)


class FakeGit:
    """Stands in for `git`: `checkout` materialises a tiny upstream tree.

    `plant` lets a test add hostile content to that tree, the way a compromised
    or malicious upstream could.
    """

    def __init__(self, commit: str = SHA, plant: Callable[[Path], None] | None = None) -> None:
        self.commit = commit
        self.plant = plant
        self.calls: list[list[str]] = []

    def __call__(self, args: list[str], cwd: Path) -> str:
        self.calls.append(args)
        if args[0] == "checkout":
            docs = cwd / "docs" / "en" / "docs"
            (docs / "tutorial").mkdir(parents=True)
            (docs / "index.md").write_text("# Home\n\n{* ../../docs_src/a/app.py *}\n", encoding="utf-8")
            (docs / "tutorial" / "x.md").write_text("# X\n\n{* ../../docs_src/gone.py *}\n", encoding="utf-8")
            (cwd / "docs_src" / "a").mkdir(parents=True)
            (cwd / "docs_src" / "a" / "app.py").write_text("app = 1\n", encoding="utf-8")
            (cwd / "docs_src" / "unused.py").write_text("never included\n", encoding="utf-8")
            if self.plant:
                self.plant(cwd)
        if args[0] == "rev-parse":
            return self.commit
        return ""


def test_fetch_keeps_the_docs_and_only_the_examples_they_include(tmp_path, monkeypatch):
    monkeypatch.setattr(corpus_mod, "_git", FakeGit())
    dest = tmp_path / "data" / "fastapi-0.1.0"
    lock = fetch_corpus("0.1.0", dest)

    assert set(lock.files) == {"docs/index.md", "docs/tutorial/x.md", "docs_src/a/app.py"}
    assert lock.missing_includes == ["gone.py"]
    assert verify_corpus(dest, lock) == []
    # No scratch directories are left behind next to the corpus.
    assert sorted(p.name for p in dest.parent.iterdir()) == ["fastapi-0.1.0"]


def test_fetch_asks_git_for_the_tag_namespace_only(tmp_path, monkeypatch):
    git = FakeGit()
    monkeypatch.setattr(corpus_mod, "_git", git)
    fetch_corpus("0.1.0", tmp_path / "fastapi-0.1.0")
    fetch = next(c for c in git.calls if c[0] == "fetch")
    assert fetch[-1] == "refs/tags/0.1.0"


@pytest.mark.parametrize("ref", ["--upload-pack=touch /tmp/x", "refs/pull/1/head", "v1[/]", "../escape", "", " 0.1.0"])
def test_refs_that_are_not_release_tags_are_refused_before_git_runs(tmp_path, monkeypatch, ref):
    git = FakeGit()
    monkeypatch.setattr(corpus_mod, "_git", git)
    with pytest.raises(CorpusError, match="not a release tag"):
        fetch_corpus(ref, tmp_path / "corpus")
    assert git.calls == []


def test_git_runs_with_line_ending_conversion_disabled(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(corpus_mod.shutil, "which", lambda name: "/usr/bin/git")

    def fake_run(argv, **kwargs):
        seen["argv"] = argv
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(corpus_mod.subprocess, "run", fake_run)
    corpus_mod._git(["status"], tmp_path)
    assert seen["argv"] == ["git", "-c", "core.autocrlf=false", "-c", "core.eol=lf", "status"]


def test_refetching_from_the_lock_succeeds_when_nothing_changed(tmp_path, monkeypatch):
    monkeypatch.setattr(corpus_mod, "_git", FakeGit())
    dest = tmp_path / "fastapi-0.1.0"
    lock = fetch_corpus("0.1.0", dest)
    assert fetch_corpus("0.1.0", dest, expected=lock) == lock


def test_fetch_refuses_a_tag_that_now_points_at_another_commit(tmp_path, monkeypatch):
    monkeypatch.setattr(corpus_mod, "_git", FakeGit())
    dest = tmp_path / "fastapi-0.1.0"
    good = fetch_corpus("0.1.0", dest)

    monkeypatch.setattr(corpus_mod, "_git", FakeGit(commit="f" * 40))
    with pytest.raises(CorpusError, match="refusing to mix corpora"):
        fetch_corpus("0.1.0", dest, expected=good)
    # The good corpus that was already on disk survives a failed fetch.
    assert verify_corpus(dest, good) == []


def test_fetch_refuses_files_that_differ_from_the_lock(tmp_path, monkeypatch):
    monkeypatch.setattr(corpus_mod, "_git", FakeGit())
    dest = tmp_path / "fastapi-0.1.0"
    good = fetch_corpus("0.1.0", dest)
    tampered = CorpusLock(good.repo, good.ref, good.commit, "0" * 64, good.files)
    with pytest.raises(CorpusError, match="do not match"):
        fetch_corpus("0.1.0", dest, expected=tampered)
    assert verify_corpus(dest, good) == []


# ---------------------------------------------------------------- hostile upstream trees


def test_the_reviewers_symlink_escape_cannot_write_outside_the_corpus(tmp_path, monkeypatch):
    """docs_src/lnk -> a/b/c/d/e plus an include of lnk/../../../../../README.md
    used to pass the source check and overwrite the project's README."""
    project_readme = tmp_path / "README.md"
    project_readme.write_text("mine", encoding="utf-8")

    def plant(cwd: Path) -> None:
        (cwd / "docs_src" / "a" / "b" / "c" / "d" / "e").mkdir(parents=True)
        (cwd / "docs_src" / "lnk").symlink_to("a/b/c/d/e")
        (cwd / "docs_src" / "README.md").write_text("pwned", encoding="utf-8")
        (cwd / "docs" / "en" / "docs" / "evil.md").write_text(
            "# Evil\n\n{* ../../docs_src/lnk/../../../../../README.md *}\n", encoding="utf-8"
        )

    monkeypatch.setattr(corpus_mod, "_git", FakeGit(plant=plant))
    with pytest.raises(CorpusError, match="symlink"):
        fetch_corpus("0.1.0", tmp_path / "data" / "corpus" / "fastapi-0.1.0")
    assert project_readme.read_text(encoding="utf-8") == "mine"


def test_a_symlinked_page_is_refused_and_the_good_corpus_survives(tmp_path, monkeypatch):
    secret = tmp_path / "secret.env"
    secret.write_text("API_KEY=do-not-index", encoding="utf-8")
    monkeypatch.setattr(corpus_mod, "_git", FakeGit())
    dest = tmp_path / "fastapi-0.1.0"
    good = fetch_corpus("0.1.0", dest)

    plant = lambda cwd: (cwd / "docs" / "en" / "docs" / "leak.md").symlink_to(secret)
    monkeypatch.setattr(corpus_mod, "_git", FakeGit(plant=plant))
    with pytest.raises(CorpusError, match="symlink"):
        fetch_corpus("0.1.0", dest, expected=good)
    assert verify_corpus(dest, good) == []
    assert "do-not-index" not in "".join(p.read_text() for p in dest.rglob("*.md"))


def test_a_symlinked_docs_directory_is_refused(tmp_path, monkeypatch):
    def plant(cwd: Path) -> None:
        docs = cwd / "docs" / "en" / "docs"
        docs.rename(cwd / "elsewhere")
        docs.symlink_to(cwd / "elsewhere")

    monkeypatch.setattr(corpus_mod, "_git", FakeGit(plant=plant))
    with pytest.raises(CorpusError, match="symlink"):
        fetch_corpus("0.1.0", tmp_path / "fastapi-0.1.0")


@pytest.mark.parametrize("include", ["a/../../../escape.txt", "/etc/passwd", "a\\..\\..\\escape.txt", "C:/escape.txt"])
def test_include_paths_that_could_leave_docs_src_are_never_copied(tmp_path, monkeypatch, include):
    plant = lambda cwd: (cwd / "docs" / "en" / "docs" / "evil.md").write_text(
        f"# Evil\n\n{{* ../../docs_src/{include} *}}\n", encoding="utf-8"
    )
    monkeypatch.setattr(corpus_mod, "_git", FakeGit(plant=plant))
    lock = fetch_corpus("0.1.0", tmp_path / "data" / "fastapi-0.1.0")
    assert include in lock.missing_includes
    assert not any("escape" in p.name for p in tmp_path.rglob("*"))


def test_an_oversized_upstream_file_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(corpus_mod, "MAX_FILE_BYTES", 1_000)
    plant = lambda cwd: (cwd / "docs" / "en" / "docs" / "huge.md").write_text("x" * 5_000, encoding="utf-8")
    monkeypatch.setattr(corpus_mod, "_git", FakeGit(plant=plant))
    with pytest.raises(CorpusError, match="not documentation"):
        fetch_corpus("0.1.0", tmp_path / "fastapi-0.1.0")


# ---------------------------------------------------------------- the swap


def test_a_failed_swap_puts_the_previous_corpus_back(tmp_path):
    dest = tmp_path / "fastapi-0.1.0"
    dest.mkdir()
    (dest / "keep.md").write_text("good", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        corpus_mod._swap_into_place(tmp_path / "no-such-staging", dest)
    assert (dest / "keep.md").read_text(encoding="utf-8") == "good"
    assert not (tmp_path / "fastapi-0.1.0.previous").exists()


def test_a_swap_interrupted_between_its_renames_is_recovered_on_the_next_fetch(tmp_path, monkeypatch):
    dest = tmp_path / "fastapi-0.1.0"
    previous = tmp_path / "fastapi-0.1.0.previous"
    previous.mkdir()
    (previous / "keep.md").write_text("good", encoding="utf-8")
    corpus_mod._recover_interrupted_swap(dest)
    assert (dest / "keep.md").read_text(encoding="utf-8") == "good" and not previous.exists()

    # And a full fetch over a recovered corpus still ends with exactly one, new corpus.
    monkeypatch.setattr(corpus_mod, "_git", FakeGit())
    lock = fetch_corpus("0.1.0", dest)
    assert verify_corpus(dest, lock) == [] and not previous.exists()


# ---------------------------------------------------------------- loading


def test_load_corpus_excludes_listed_pages_and_inlines_example_code(mini_corpus: Path):
    report = load_corpus(mini_corpus)
    ids = [d.doc_id for d in report.documents]
    assert ids == ["index", "tutorial/first-steps"]  # sorted, release notes excluded
    assert report.excluded == ("release-notes.md",)
    first_steps = report.documents[1]
    assert "app = FastAPI()" in first_steps.text
    assert first_steps.title == "First Steps"


def test_loading_against_the_lock_refuses_an_icloud_conflict_copy(mini_corpus: Path):
    lock = lock_for(mini_corpus)
    assert load_corpus(mini_corpus, lock=lock).documents
    (mini_corpus / "docs" / "index 2.md").write_text("# Home\n\nA sync conflict copy.\n", encoding="utf-8")
    with pytest.raises(CorpusError, match=re.escape("unexpected: docs/index 2.md")):
        load_corpus(mini_corpus, lock=lock)


def test_an_unreadable_page_is_a_corpus_error_not_a_crash(mini_corpus: Path):
    (mini_corpus / "docs" / "index.md").write_bytes(b"# Home\n\n\xff\xfe not utf-8\n")
    with pytest.raises(CorpusError, match="Cannot read docs/index.md"):
        load_corpus(mini_corpus)


def test_load_corpus_without_a_fetched_corpus_says_how_to_fetch(tmp_path: Path):
    with pytest.raises(CorpusError, match="corpus fetch"):
        load_corpus(tmp_path / "missing")


def test_include_resolver_refuses_to_leave_docs_src(mini_corpus: Path):
    (mini_corpus / "secret.txt").write_text("do not read", encoding="utf-8")
    resolve = make_include_resolver(mini_corpus)
    assert resolve("../../docs_src/first/app.py") is not None
    assert resolve("../../docs_src/../secret.txt") is None
    assert resolve("../../docs_src/missing.py") is None
    assert resolve("../../fastapi/openapi/docs.py") is None
    assert resolve("../../docs_src/first/app\x00.py") is None  # a NUL byte is unresolved, not a crash


# ---------------------------------------------------------------- sections


def test_sections_tile_the_document_with_no_gaps_or_overlaps():
    doc = make_doc("# A { #a }\n\none\n\n## B { #b }\n\ntwo\n\n### C\n\nthree\n\n## D { #d }\n\nfour\n")
    sections = doc.sections
    assert sections[0].start == 0 and sections[-1].end == len(doc.text)
    assert all(x.end == y.start for x, y in zip(sections, sections[1:]))
    assert [s.anchor for s in sections] == ["a", "b", "c", "d"]


def test_heading_paths_follow_the_nesting():
    doc = make_doc("# A\n\nx\n\n## B\n\ny\n\n### C\n\nz\n\n## D\n\nw\n")
    assert [s.heading_path for s in doc.sections] == [
        ("A",), ("A", "B"), ("A", "B", "C"), ("A", "D"),
    ]


def test_text_before_the_first_heading_gets_its_own_section():
    doc = make_doc("Intro paragraph.\n\n# Title\n\nBody.\n", doc_id="page")
    first = doc.sections[0]
    assert (first.anchor, first.level, first.start) == ("", 0, 0)
    assert first.heading_path == ("Title",)


def test_section_at_finds_the_section_containing_an_offset():
    doc = make_doc("# A { #a }\n\none\n\n## B { #b }\n\ntwo\n")
    assert doc.section_at(0).anchor == "a"
    assert doc.section_at(doc.text.index("two")).anchor == "b"
    assert doc.section_at(len(doc.text) - 1).anchor == "b"


# ---------------------------------------------------------------- the real corpus


@pytest.mark.corpus
def test_no_mkdocs_syntax_survives_cleaning(real_docs):
    text = "\n".join(d.text for d in real_docs)
    assert not re.search(r"\{ *#[\w-]+ *\}", text)        # heading anchors
    assert not re.search(r"(?m)^\s*\{\*\s*\S", text)       # {* include *} directives
    assert not re.search(r"(?m)^/{3,}", text)              # /// admonition and tab fences
    assert not re.search(r"(?m)^:::\s", text)              # mkdocstrings directives


@pytest.mark.corpus
def test_no_markdown_link_urls_survive_in_prose(real_docs):
    # The review found 81 links like [`Depends`](...) whose URLs leaked into the text.
    leaks = [d.doc_id for d in real_docs for line in _prose_lines(d.text) if re.search(r"\]\((?:https?:|/|\.)", line)]
    assert leaks == []


def _prose_lines(text: str):
    fence = None
    for line in text.split("\n"):
        if fence is None and (opened := fence_open(line)):
            fence = (opened[1], opened[2])
        elif fence is not None and fence_closes(line, *fence):
            fence = None
        elif fence is None:
            yield re.sub(r"`+[^`]*`+", "", line)  # code spans may legitimately show link syntax


@pytest.mark.corpus
def test_real_heading_offsets_and_sections_are_exact(real_docs):
    for doc in real_docs:
        for h in doc.headings:
            assert doc.text[h.offset:].startswith("#" * h.level + " "), (doc.doc_id, h.title)
        s = doc.sections
        assert s[0].start == 0 and s[-1].end == len(doc.text), doc.doc_id
        assert all(a.end == b.start for a, b in zip(s, s[1:])), doc.doc_id
        anchors = [x.anchor for x in s if x.anchor]
        assert len(anchors) == len(set(anchors)), doc.doc_id
