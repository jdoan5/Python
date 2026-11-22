"""Freeze points are read from origin/main's contents, never from dates or tags. Uses a throwaway repo."""

from __future__ import annotations

from pathlib import Path

import pytest

from raglab.freeze import FreezeError, current, file_state, forbid, origin_sha256, require
from tests.test_authorize import git

FILES = {1: ("PLAN.md",), 2: ("plan/requests.jsonl", "../golden/v3.yaml"), 3: ("store/SEALED.json",)}


@pytest.fixture
def stage(tmp_path: Path) -> Path:
    origin = tmp_path / "origin.git"
    git(tmp_path, "init", "--bare", "-b", "main", str(origin))
    repo = tmp_path / "repo"
    stage = repo / "Project" / "Stage 4"
    (stage / "plan").mkdir(parents=True)
    (repo / "Project" / "golden").mkdir()
    (stage / "PLAN.md").write_text("# plan\n", encoding="utf-8")
    git(repo, "init", "-b", "main")
    git(repo, "config", "user.email", "test@example.com")
    git(repo, "config", "user.name", "Test")
    git(repo, "add", ".")
    git(repo, "commit", "-m", "plan")
    git(repo, "remote", "add", "origin", str(origin))
    git(repo, "push", "-u", "origin", "main")
    return stage


def push(stage: Path, message: str) -> None:
    repo = stage.parents[1]
    git(repo, "add", ".")
    git(repo, "commit", "-m", message)
    git(repo, "push", "origin", "main")


def test_a_freeze_point_is_its_files_on_origin_main(stage: Path) -> None:
    assert current(stage, FILES) == 1 and file_state(stage, "PLAN.md", ) == "pushed"
    require(1, "the pilot", stage, FILES)
    with pytest.raises(FreezeError, match="freeze point 2"):
        require(2, "generation", stage, FILES)
    (stage / "plan" / "requests.jsonl").write_text("{}\n", encoding="utf-8")
    (stage.parent / "golden" / "v3.yaml").write_text("questions: []\n", encoding="utf-8")
    repo = stage.parents[1]
    git(repo, "add", ".")
    git(repo, "commit", "-m", "inputs, committed but not pushed")
    assert current(stage, FILES) == 1  # committed isn't enough: it must be on origin/main
    git(repo, "push", "origin", "main")
    assert current(stage, FILES) == 2
    assert origin_sha256(stage, "../golden/v3.yaml") is not None


def test_a_file_changed_after_its_push_breaks_the_freeze_point(stage: Path) -> None:
    (stage / "PLAN.md").write_text("# plan, edited\n", encoding="utf-8")
    assert file_state(stage, "PLAN.md") == "changed" and current(stage, FILES) == 0
    with pytest.raises(FreezeError, match="changed since"):
        require(1, "the pilot", stage, FILES)


def test_forbid_closes_a_step_as_soon_as_its_freeze_point_begins(stage: Path) -> None:
    forbid(3, "generation", stage, FILES)  # not begun: fine
    (stage / "store").mkdir()
    (stage / "store" / "SEALED.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FreezeError, match="closed"):  # sealed here, not pushed yet: already closed
        forbid(3, "generation", stage, FILES)
    push(stage, "sealed")
    (stage / "store" / "SEALED.json").unlink()  # deleting the local file can't reopen it once pushed
    with pytest.raises(FreezeError, match="closed"):
        forbid(3, "generation", stage, FILES)


def test_a_stale_origin_ref_is_refused(stage: Path, tmp_path: Path) -> None:
    # Someone else pushes to GitHub; this checkout's origin/main is now behind what GitHub holds.
    other = tmp_path / "other"
    git(tmp_path, "clone", str(stage.parents[1] / ".." / "origin.git"), str(other))
    git(other, "config", "user.email", "test@example.com")
    git(other, "config", "user.name", "Test")
    (other / "Project" / "note.txt").write_text("x", encoding="utf-8")
    git(other, "add", ".")
    git(other, "commit", "-m", "elsewhere")
    git(other, "push", "origin", "main")
    with pytest.raises(FreezeError, match="isn't what GitHub holds"):
        require(1, "the pilot", stage, FILES)
    git(stage.parents[1], "fetch", "origin")
    require(1, "the pilot", stage, FILES)
