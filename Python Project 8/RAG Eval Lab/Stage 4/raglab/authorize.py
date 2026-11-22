"""Spend authorization: a typed approval, signed with the API key, tied to the plan, a request list and a pushed commit.

No billed command runs without an authorization for its phase (PLAN.md, "Phases"):
- `pilot`: the smoke test, the pilot's generation and its timing probe (pilot/requests.jsonl), after freeze point 1;
- `pilot-judging`: the pilot's judge requests (pilot/judge-requests.jsonl), once they are built from its answers;
- `generation`: every golden generation request (plan/requests.jsonl), after freeze point 2;
- `judging`: every golden judge request (plan/judge-requests.jsonl), after freeze point 4.
The two pilot authorizations share the pilot's cap.

The developer runs `raglab budget authorize --phase <phase>` in their own
terminal and types the phrase it shows. The phrase names the phase, its cap
and the first 8 hex digits of the SHA-256 of the plan and of the request list
it approves. It is read from /dev/tty, and the command refuses to run with
CLAUDECODE set or without a terminal. That stops accidents; it can't stop a
determined program. The human gate is the macOS Keychain: the key is stored
trusting no application, so reading it raises a dialog the developer must
answer, and every authorization record carries an HMAC made with the key. A
record written by anything that never got past that dialog fails `require`.

An authorization holds only while:
- PLAN.md and the request list are unchanged;
- HEAD is the commit it was given at, so the code that spends is the code approved;
- the files that decide what this phase sends have no uncommitted changes;
- that commit is on origin/main, so the public record predates the spend.
Every reservation in the ledger records the SHA-256 of the authorization it was made under. The
record file is append-only, so every authorization ever given stays readable beside the reservations
it covered; the newest one for a phase is the one `require` checks.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Callable, Sequence


class AuthorizationError(RuntimeError):
    pass


PHASES = ("pilot", "pilot-judging", "generation", "judging")
LEDGER_PHASE = {"pilot": "pilot", "pilot-judging": "pilot", "generation": "generation", "judging": "judging"}

# What decides what each phase sends and what it may cost, relative to the stage folder. Narrow on purpose:
# golden-set annotation (FP2's inputs) can go on while the pilot spends.
_CODE = ("raglab", "prompts", "PLAN.md", "pricing.json", "pyproject.toml", "requirements-lock.txt",
         "../corpus.lock.json", "../models.lock.json")
GUARDED = {
    "pilot": _CODE + ("pilot/questions.yaml", "pilot/labels.json", "pilot/sufficiency.json", "pilot/keys.json",
                      "pilot/requests.jsonl"),
    "pilot-judging": _CODE + ("pilot/questions.yaml", "pilot/labels.json", "pilot/sufficiency.json",
                              "pilot/keys.json", "pilot/requests.jsonl", "pilot/judge-requests.jsonl",
                              "pilot/candidates.json", "pilot/checks.json"),
    "generation": _CODE + ("plan/requests.jsonl", "pilot/REPORT.json", "../golden/v3.yaml",
                           "../golden/v3-sufficiency.json", "../golden/v3-keys.json", "../golden/v3-provenance.json"),
    "judging": _CODE + ("plan", "human", "../golden/v3.yaml", "../golden/v3-sufficiency.json",
                        "../golden/v3-keys.json", "../golden/v3-provenance.json", "../golden/v3-cited.yaml",
                        "../golden/v3-checks.json", "../golden/v3-rekeys.json"),
}


def _git(stage_dir: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=stage_dir, capture_output=True, text=True, timeout=60)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class GitState:
    head: str
    dirty: tuple[str, ...]
    pushed: bool


def git_state(stage_dir: Path, guarded: Sequence[str]) -> GitState:
    head = _git(stage_dir, "rev-parse", "HEAD")
    if head.returncode != 0:
        raise AuthorizationError(f"not a git checkout: {head.stderr.strip()}")
    # Every guarded path, present or not: a tracked file that was deleted (iCloud evicts files) is a change too,
    # and a pathspec that matches nothing is simply quiet.
    status = _git(stage_dir, "status", "--porcelain", "--untracked-files=all", "--", *guarded) if guarded else None
    if status is not None and status.returncode != 0:
        raise AuthorizationError(f"git status failed: {status.stderr.strip()}")
    dirty = tuple(line for line in (status.stdout.splitlines() if status else []) if line.strip())
    pushed = _git(stage_dir, "merge-base", "--is-ancestor", "HEAD", "origin/main").returncode == 0
    return GitState(head.stdout.strip(), dirty, pushed)


def phrase(phase: str, cap: Decimal, plan_sha: str, requests_sha: str) -> str:
    return f"authorize {phase} up to ${cap:.2f} plan {plan_sha[:8]} requests {requests_sha[:8]}"


def _read_tty(prompt: str) -> str:
    """Read one line from the controlling terminal itself, never from stdin."""
    with open("/dev/tty", "r+", encoding="utf-8") as tty:
        tty.write(prompt)
        tty.flush()
        return tty.readline()


def _canonical(record: dict) -> bytes:
    return json.dumps({k: v for k, v in record.items() if k != "hmac"}, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def sign(record: dict, key: str) -> str:
    return hmac.new(key.encode("utf-8"), _canonical(record), hashlib.sha256).hexdigest()


def record_sha256(record: dict) -> str:
    return hashlib.sha256(json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _load(record_path: Path) -> list[dict]:
    if not record_path.exists():
        return []
    return json.loads(record_path.read_text(encoding="utf-8"))


def typed_by_developer(expected: str, prompt: str, read: Callable[[str], str] = _read_tty,
                       isatty: Callable[[], bool] = sys.stdin.isatty, env: dict | None = None) -> None:
    """Refuse unless the developer types `expected` in their own terminal (never inside Claude Code)."""
    env = dict(os.environ) if env is None else env
    if env.get("CLAUDECODE"):
        raise AuthorizationError("run this in your own terminal, not inside Claude Code")
    if not isatty():
        raise AuthorizationError("run this yourself in a terminal: it has to be typed, not piped")
    if read(f"{prompt}\n  {expected}\n> ").strip() != expected:
        raise AuthorizationError("not done: what was typed didn't match")


def authorize(phase: str, cap: Decimal, stage_dir: Path, record_path: Path, requests_path: Path, key: str,
              read: Callable[[str], str] = _read_tty, isatty: Callable[[], bool] = sys.stdin.isatty,
              env: dict | None = None) -> dict:
    """Ask the developer to type the phrase; on a match, record the authorization, signed with the key."""
    env = dict(os.environ) if env is None else env
    if phase not in PHASES:
        raise AuthorizationError(f"phase must be one of {PHASES}")
    if env.get("CLAUDECODE"):
        raise AuthorizationError("run this in your own terminal, not inside Claude Code")
    if not isatty():
        raise AuthorizationError("run this yourself in a terminal: the phrase has to be typed, not piped")
    if not requests_path.is_file():
        raise AuthorizationError(f"{requests_path} doesn't exist: build and commit the request list first")
    state = git_state(stage_dir, GUARDED[phase])
    if state.dirty:
        raise AuthorizationError("commit these first:\n  " + "\n  ".join(state.dirty))
    if not state.pushed:
        raise AuthorizationError("push first: HEAD is not on origin/main")
    plan_sha, requests_sha = file_sha256(stage_dir / "PLAN.md"), file_sha256(requests_path)
    expected = phrase(phase, cap, plan_sha, requests_sha)
    typed = read(f"Commit {state.head[:8]}. To let phase {phase!r} spend up to ${cap:.2f} on the requests in "
                 f"{requests_path.name}, type exactly:\n  {expected}\n> ")
    if typed.strip() != expected:
        raise AuthorizationError("not authorized: the phrase didn't match")
    record = {"phase": phase, "ledger_phase": LEDGER_PHASE[phase], "cap": f"{cap:.2f}", "plan_sha256": plan_sha,
              "requests": requests_path.name, "requests_sha256": requests_sha, "commit": state.head,
              "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    record["hmac"] = sign(record, key)
    records = _load(record_path) + [record]  # append-only: older reservations' authorizations stay readable
    record_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = record_path.with_name(record_path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:  # a crash mid-write never loses an earlier authorization
        f.write(json.dumps(records, indent=1) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, record_path)
    return record


def require(phase: str, stage_dir: Path, record_path: Path, requests_path: Path, key: str) -> dict:
    """The authorization for this phase, if it still holds; otherwise why not. Run before every reservation."""
    match = [r for r in _load(record_path) if r["phase"] == phase]
    if not match:
        raise AuthorizationError(f"phase {phase!r} isn't authorized: run `raglab budget authorize --phase {phase}` "
                                 "in your own terminal")
    record = match[-1]  # the newest authorization for the phase is the one that holds
    if not hmac.compare_digest(record.get("hmac", ""), sign(record, key)):
        raise AuthorizationError(f"the {phase} authorization isn't signed with this API key: it wasn't made by "
                                 "`budget authorize`, or it was edited")
    if file_sha256(stage_dir / "PLAN.md") != record["plan_sha256"]:
        raise AuthorizationError("PLAN.md changed since this phase was authorized")
    if not requests_path.is_file() or file_sha256(requests_path) != record["requests_sha256"]:
        raise AuthorizationError(f"{requests_path.name} changed since this phase was authorized")
    state = git_state(stage_dir, GUARDED[phase])
    if state.head != record["commit"]:
        raise AuthorizationError(f"HEAD is {state.head[:8]}, but the authorization was given at "
                                 f"{record['commit'][:8]}: authorize again for the new commit")
    if state.dirty:
        raise AuthorizationError("uncommitted changes to files that decide what is sent:\n  " + "\n  ".join(state.dirty))
    if not state.pushed:
        raise AuthorizationError("HEAD is not on origin/main")
    return record
