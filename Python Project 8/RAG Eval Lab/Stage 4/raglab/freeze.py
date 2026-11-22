"""Which freeze point the repository has reached, read from what is on origin/main (PLAN.md, "Freeze points").

A freeze point is reached when each of its files is on origin/main with exactly
the bytes the working tree holds. Commit dates and tags don't count: dates are
set by whoever commits, and tags were never made in Stage 3. Commands call
`require` before acting and `forbid` once acting would break an order the plan
promises (no generation after the answers are sealed; no judging before the
grades are pushed, or after the verdicts are sealed). Both first check that the
local origin/main is what GitHub holds (`git ls-remote`), since this repository
lives in iCloud Drive, where refs have been lost before.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Sequence

STAGE = Path(__file__).resolve().parents[1]

# Paths relative to the stage folder ("../golden/..." for the shared golden set).
FREEZE_POINTS: dict[int, tuple[str, ...]] = {
    1: ("PLAN.md", "pilot/questions.yaml", "pilot/labels.json", "pilot/sufficiency.json", "pilot/keys.json",
        "pilot/requests.jsonl"),
    2: ("pilot/REPORT.md", "pilot/REPORT.json", "plan/requests.jsonl", "../golden/v3.yaml",
        "../golden/v3-sufficiency.json", "../golden/v3-keys.json", "../golden/v3-provenance.json"),
    3: ("store/generation/SEALED.json", "store/generation/ledger.jsonl", "store/generation/authorizations.json",
        "human/draw.json"),
    4: ("human/grades.json", "plan/judge-requests.jsonl", "plan/scope.json", "../golden/v3-cited.yaml",
        "../golden/v3-checks.json", "../golden/v3-rekeys.json"),
    5: ("store/judging/SEALED.json", "store/judging/ledger.jsonl", "store/judging/authorizations.json"),
}


class FreezeError(RuntimeError):
    pass


def _git(stage: Path, *args: str, text: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=stage, capture_output=True, text=text, timeout=60)


def _repo_path(stage: Path, rel: str) -> str:
    """The path git uses on origin/main: relative to the repository root, with '..' resolved."""
    prefix = _git(stage, "rev-parse", "--show-prefix").stdout.strip()
    parts: list[str] = [p for p in prefix.split("/") if p]
    for piece in rel.split("/"):
        if piece == "..":
            parts.pop()
        elif piece and piece != ".":
            parts.append(piece)
    return "/".join(parts)


def origin_sha256(stage: Path, rel: str) -> str | None:
    """SHA-256 of the file's bytes on origin/main, or None if it isn't there."""
    shown = _git(stage, "show", f"origin/main:{_repo_path(stage, rel)}", text=False)
    return hashlib.sha256(shown.stdout).hexdigest() if shown.returncode == 0 else None


def file_state(stage: Path, rel: str) -> str:
    """'pushed' (on origin/main, same bytes), 'changed' (there, but the working copy differs), or 'absent'."""
    remote = origin_sha256(stage, rel)
    if remote is None:
        return "absent"
    local = stage / rel
    if not local.is_file() or hashlib.sha256(local.read_bytes()).hexdigest() != remote:
        return "changed"
    return "pushed"


def check_remote(stage: Path = STAGE) -> None:
    """The local origin/main must be what GitHub holds: a stale or iCloud-damaged ref would read the wrong state."""
    local = _git(stage, "rev-parse", "origin/main")
    remote = _git(stage, "ls-remote", "origin", "refs/heads/main")
    if local.returncode != 0:
        raise FreezeError("no local origin/main: fetch from GitHub first (GitHub Desktop: Fetch origin)")
    if remote.returncode != 0 or not remote.stdout.strip():
        raise FreezeError("couldn't read main from GitHub (git ls-remote origin); check the network and try again")
    if remote.stdout.split()[0] != local.stdout.strip():
        raise FreezeError("the local origin/main isn't what GitHub holds: fetch first (GitHub Desktop: Fetch origin)")


def reached(fp: int, stage: Path = STAGE, files: dict[int, Sequence[str]] | None = None) -> bool:
    return all(file_state(stage, rel) == "pushed" for rel in (files or FREEZE_POINTS)[fp])


def current(stage: Path = STAGE, files: dict[int, Sequence[str]] | None = None) -> int:
    """The highest freeze point reached with every earlier one reached too (0 for none)."""
    fps = files or FREEZE_POINTS
    n = 0
    for fp in sorted(fps):
        if not reached(fp, stage, fps):
            break
        n = fp
    return n


def require(fp: int, what: str, stage: Path = STAGE, files: dict[int, Sequence[str]] | None = None) -> None:
    fps = files or FREEZE_POINTS
    check_remote(stage)
    missing = [rel for rel in fps[fp] if file_state(stage, rel) != "pushed"]
    if missing:
        raise FreezeError(f"{what} needs freeze point {fp} on origin/main; not there yet (or changed since): "
                          + ", ".join(missing))


def forbid(fp: int, what: str, stage: Path = STAGE, files: dict[int, Sequence[str]] | None = None) -> None:
    """Refuse once freeze point fp has begun: any of its files exists here, or it is on origin/main.

    Monotone on purpose: a stale ref, or a freeze point not pushed yet, never reopens a closed step.
    """
    fps = files or FREEZE_POINTS
    check_remote(stage)
    begun = [rel for rel in fps[fp] if (stage / rel).exists() or origin_sha256(stage, rel) is not None]
    if begun:
        raise FreezeError(f"{what} is closed: freeze point {fp} has begun ({', '.join(begun)} exists here or on "
                          "origin/main). A change now is a deviation; log it in the README first.")
