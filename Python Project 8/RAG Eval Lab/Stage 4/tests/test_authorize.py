"""Spend authorization: typed, tied to the plan and to a pushed commit. Uses a throwaway git repo."""

from __future__ import annotations

import subprocess
from decimal import Decimal
from pathlib import Path

import pytest

from raglab.authorize import AuthorizationError, authorize, file_sha256, phrase, require

KEY = "not-a-real-key-0000"


def git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


@pytest.fixture
def stage(tmp_path: Path) -> Path:
    """A repo shaped like this one (a stage folder beside golden/), pushed to a local bare 'origin'."""
    origin = tmp_path / "origin.git"
    git(tmp_path, "init", "--bare", "-b", "main", str(origin))
    repo = tmp_path / "repo"
    stage = repo / "Stage 4"
    (stage / "raglab").mkdir(parents=True)
    (repo / "golden").mkdir()
    (stage / "PLAN.md").write_text("# plan\n", encoding="utf-8")
    (stage / "raglab" / "x.py").write_text("x = 1\n", encoding="utf-8")
    (stage / "pilot").mkdir()
    (stage / "pilot" / "requests.jsonl").write_text('{"custom_id": "r1"}\n', encoding="utf-8")
    (repo / "golden" / "v3.yaml").write_text("questions: []\n", encoding="utf-8")
    git(repo, "init", "-b", "main")
    git(repo, "config", "user.email", "test@example.com")
    git(repo, "config", "user.name", "Test")
    git(repo, "add", ".")
    git(repo, "commit", "-m", "plan")
    git(repo, "remote", "add", "origin", str(origin))
    git(repo, "push", "-u", "origin", "main")
    return stage


def typed(text: str):
    return lambda prompt: text


def ok_phrase(stage: Path, phase: str = "pilot", cap: str = "4") -> str:
    return phrase(phase, Decimal(cap), file_sha256(stage / "PLAN.md"), file_sha256(stage / "pilot" / "requests.jsonl"))


def run(stage: Path, rec: Path, text: str, isatty: bool = True, env: dict | None = None, phase: str = "pilot"):
    return authorize(phase, Decimal("4"), stage, rec, stage / "pilot" / "requests.jsonl", KEY, read=typed(text),
                     isatty=lambda: isatty, env={} if env is None else env)


def test_the_phrase_must_be_typed_in_a_terminal_outside_claude_code(stage: Path, tmp_path: Path) -> None:
    rec = tmp_path / "auth.json"
    good = ok_phrase(stage)
    assert good.startswith("authorize pilot up to $4.00 plan ") and " requests " in good
    with pytest.raises(AuthorizationError, match="terminal"):
        run(stage, rec, good, isatty=False)
    with pytest.raises(AuthorizationError, match="Claude Code"):
        run(stage, rec, good, env={"CLAUDECODE": "1"})
    with pytest.raises(AuthorizationError, match="didn't match"):
        run(stage, rec, "yes")
    assert not rec.exists()
    record = run(stage, rec, good)
    assert record["phase"] == "pilot" and record["cap"] == "4.00"
    requests = stage / "pilot" / "requests.jsonl"
    assert require("pilot", stage, rec, requests, KEY)["commit"] == record["commit"]
    with pytest.raises(AuthorizationError, match="isn't authorized"):
        require("generation", stage, rec, requests, KEY)


def test_an_authorization_lapses_when_the_plan_code_requests_or_commit_change(stage: Path, tmp_path: Path) -> None:
    rec = tmp_path / "auth.json"
    requests = stage / "pilot" / "requests.jsonl"
    run(stage, rec, ok_phrase(stage))
    (stage / "raglab" / "x.py").write_text("x = 2\n", encoding="utf-8")
    with pytest.raises(AuthorizationError, match="uncommitted"):
        require("pilot", stage, rec, requests, KEY)
    git(stage.parent, "commit", "-am", "change the code")
    with pytest.raises(AuthorizationError, match="authorize again"):
        require("pilot", stage, rec, requests, KEY)
    requests.write_text('{"custom_id": "r2"}\n', encoding="utf-8")
    with pytest.raises(AuthorizationError, match="requests.jsonl changed"):
        require("pilot", stage, rec, requests, KEY)
    (stage / "PLAN.md").write_text("# plan, edited\n", encoding="utf-8")
    with pytest.raises(AuthorizationError, match="PLAN.md changed"):
        require("pilot", stage, rec, requests, KEY)


def test_an_unpushed_commit_cant_be_authorized(stage: Path, tmp_path: Path) -> None:
    (stage / "raglab" / "y.py").write_text("y = 1\n", encoding="utf-8")
    git(stage.parent, "add", ".")
    git(stage.parent, "commit", "-m", "not pushed yet")
    with pytest.raises(AuthorizationError, match="push first"):
        run(stage, tmp_path / "a.json", ok_phrase(stage))


def test_an_untracked_file_in_a_guarded_folder_blocks_authorization(stage: Path, tmp_path: Path) -> None:
    (stage / "raglab" / "new_module.py").write_text("x = 1\n", encoding="utf-8")
    with pytest.raises(AuthorizationError, match="commit these first"):
        run(stage, tmp_path / "a.json", ok_phrase(stage))


def test_golden_annotation_work_doesnt_block_the_pilot(stage: Path, tmp_path: Path) -> None:
    # FP2's inputs are written while the pilot spends; only the files that decide what the pilot sends are guarded
    (stage.parent / "golden" / "v3-sufficiency.json").write_text("{}", encoding="utf-8")
    run(stage, tmp_path / "a.json", ok_phrase(stage))
    require("pilot", stage, tmp_path / "a.json", stage / "pilot" / "requests.jsonl", KEY)


def test_a_record_not_signed_with_the_key_is_refused(stage: Path, tmp_path: Path) -> None:
    import json

    rec = tmp_path / "auth.json"
    requests = stage / "pilot" / "requests.jsonl"
    run(stage, rec, ok_phrase(stage))
    with pytest.raises(AuthorizationError, match="isn't signed with this API key"):
        require("pilot", stage, rec, requests, "another-key")
    forged = json.loads(rec.read_text(encoding="utf-8"))
    forged[0]["cap"] = "25.00"  # an edited record keeps its old signature, which no longer matches
    rec.write_text(json.dumps(forged), encoding="utf-8")
    with pytest.raises(AuthorizationError, match="isn't signed"):
        require("pilot", stage, rec, requests, KEY)
    forged[0].pop("hmac")  # and a record written by hand has none
    rec.write_text(json.dumps(forged), encoding="utf-8")
    with pytest.raises(AuthorizationError, match="isn't signed"):
        require("pilot", stage, rec, requests, KEY)


def test_a_typed_confirmation_needs_the_developers_own_terminal() -> None:
    from raglab.authorize import typed_by_developer

    typed_by_developer("resend x", "type:", read=lambda p: "resend x\n", isatty=lambda: True, env={})
    for kwargs, match in (({"env": {"CLAUDECODE": "1"}}, "own terminal"),
                          ({"isatty": lambda: False}, "typed, not piped"),
                          ({"read": lambda p: "resend y"}, "didn't match")):
        args = {"read": lambda p: "resend x", "isatty": lambda: True, "env": {}} | kwargs
        with pytest.raises(AuthorizationError, match=match):
            typed_by_developer("resend x", "type:", **args)


def test_authorizations_are_kept_and_the_newest_holds(stage: Path, tmp_path: Path) -> None:
    import json

    rec = tmp_path / "auth.json"
    first = run(stage, rec, ok_phrase(stage))
    (stage / "raglab" / "x.py").write_text("x = 3\n", encoding="utf-8")
    git(stage.parent, "commit", "-am", "a fix")
    git(stage.parent, "push")
    second = run(stage, rec, ok_phrase(stage))
    records = json.loads(rec.read_text())
    assert [r["commit"] for r in records] == [first["commit"], second["commit"]]  # the first is still readable
    assert require("pilot", stage, rec, stage / "pilot" / "requests.jsonl", KEY)["commit"] == second["commit"]


def test_a_deleted_guarded_file_counts_as_a_change(stage: Path, tmp_path: Path) -> None:
    rec = tmp_path / "auth.json"
    run(stage, rec, ok_phrase(stage))
    (stage / "raglab" / "x.py").unlink()  # as iCloud might
    with pytest.raises(AuthorizationError, match="uncommitted"):
        require("pilot", stage, rec, stage / "pilot" / "requests.jsonl", KEY)
