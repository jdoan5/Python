"""PLAN.md's parameters are the code's, and Stage 4 runs Stage 3's retrieval exactly."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from raglab import evaluate, params
from raglab.ledger import Pricing

STAGE = Path(__file__).resolve().parents[1]
STAGE3 = STAGE.parent / "Stage 3"

# Stage 3's modules that Stage 4 runs unchanged: the windows every arm is built from are Stage 3's windows.
UNCHANGED = ("bench.py", "chunking.py", "corpus.py", "evaluate.py", "experiments.py", "golden.py", "markdown.py",
             "metrics.py", "models.py", "neural.py", "retrieval.py", "sections.py", "stats.py")


def plan_parameters() -> dict:
    text = (STAGE / "PLAN.md").read_text(encoding="utf-8")
    section = text.split("## Parameters (machine-readable)", 1)[1]
    return json.loads(re.search(r"```json\n(.*?)\n```", section, re.S).group(1))


def test_the_plan_s_parameters_are_the_code_s():
    assert plan_parameters() == params.as_plan_block()


def test_every_run_records_which_plan_it_follows():
    assert evaluate.plan_path() == STAGE / "PLAN.md"


def test_stage_4_runs_stage_3_s_retrieval_code_byte_for_byte():
    for name in UNCHANGED:
        ours, theirs = (d / "raglab" / name for d in (STAGE, STAGE3))
        assert hashlib.sha256(ours.read_bytes()).digest() == hashlib.sha256(theirs.read_bytes()).digest(), name


def test_the_ledger_and_budget_use_the_plan_s_prices_and_caps():
    pricing = Pricing.load(STAGE / "pricing.json")
    assert set(pricing.models) == {params.GENERATOR, params.JUDGE}
    assert pricing.batch_multiplier == 0.5
    import sys
    sys.path.insert(0, str(STAGE / "analysis"))
    import budget

    assert (budget.CAP, budget.CEILING) == (params.CAP, params.PLANNING_CEILING)


def _pins(path: Path) -> dict[str, str]:
    pins = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("\\")[0].strip()
        if line and not line.startswith(("#", "--")) and "==" in line:
            name, version = line.split("==")
            pins[name.lower().replace("_", "-")] = version.strip()
    return pins


def test_stage_4_s_lock_keeps_every_stage_3_pin_and_pins_every_wheel_by_hash():
    # All stages share one environment, so Stage 4's install must not move anything Stage 3 locked.
    stage3, stage4 = _pins(STAGE3 / "requirements-lock.txt"), _pins(STAGE / "requirements-lock.txt")
    assert {name: stage4.get(name) for name in stage3} == stage3
    text = (STAGE / "requirements-lock.txt").read_text(encoding="utf-8")
    assert text.count("--hash=sha256:") == len(stage4)
    assert stage4["anthropic"] == params.as_plan_block()["sdk"]
    assert set(stage4) - set(stage3) == {"anthropic", "docstring-parser", "jiter", "opentelemetry-api", "sniffio"}


def test_the_installed_sdk_is_the_locked_one():
    import importlib.metadata as md

    stage4 = _pins(STAGE / "requirements-lock.txt")
    for name in ("anthropic", "jiter", "opentelemetry-api", "docstring-parser", "sniffio"):
        assert md.version(name) == stage4[name], name


# Dicts keyed by data (a model id, a condition name), so their keys are read through a variable, never spelled out.
KEYED_BY_DATA = {"BATCH_MAX", "RUNS"}


def test_every_parameter_is_read_by_the_code_that_applies_it():
    """A frozen value no module reads would freeze a rule nothing applies. A constant counts as read where a
    module uses `params.NAME` or imports NAME from raglab.params; a dict key where such a module subscripts it."""
    sources = {p.name: p.read_text(encoding="utf-8") for p in sorted((STAGE / "raglab").glob("*.py"))
               + sorted((STAGE / "analysis").glob("*.py")) if p.name != "params.py"}
    unread = []
    for name in (n for n in dir(params) if n.isupper()):
        readers = [f for f, src in sources.items() if re.search(rf"\bparams\.{name}\b", src)
                   or re.search(rf"from raglab\.params import[^\n]*\b{name}\b", src)]
        if not readers:
            unread.append(name)
            continue
        value = getattr(params, name)
        if isinstance(value, dict) and name not in KEYED_BY_DATA:
            unread += [f"{name}[{k!r}]" for k in value if isinstance(k, str)
                       and not any(f'["{k}"]' in sources[f] or f"['{k}']" in sources[f] for f in readers)]
    assert unread == []


def test_the_plan_is_the_one_frozen_at_freeze_point_1():
    """After freeze point 1 the plan never changes: a change is a deviation, logged in the README."""
    import pytest
    import subprocess

    report = STAGE / "pilot" / "REPORT.json"
    if not report.exists():
        pytest.skip("freeze point 1's commit is recorded in pilot/REPORT.json, which the pilot writes")
    commit = json.loads(report.read_text(encoding="utf-8"))["fp1_commit"]
    prefix = subprocess.run(["git", "rev-parse", "--show-prefix"], cwd=STAGE, capture_output=True, text=True,
                            check=True).stdout.strip()
    frozen = subprocess.run(["git", "show", f"{commit}:{prefix}PLAN.md"], cwd=STAGE, capture_output=True, check=True)
    assert frozen.stdout == (STAGE / "PLAN.md").read_bytes()
