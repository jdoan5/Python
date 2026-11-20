"""PLAN.md's parameters are the code's: a setting can't change in one without the other."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from conftest import REAL_MODELS_LOCK
from raglab import evaluate, metrics, neural, stats
from raglab.experiments import CHAIN, EVALUATED, EXTRAS
from raglab.golden import default_golden_path, load_golden, questions_fingerprint
from raglab.models import ModelLock

STAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STAGE / "analysis"))

import v2_labels  # noqa: E402


def plan_parameters() -> dict:
    text = (STAGE / "PLAN.md").read_text(encoding="utf-8")
    section = text.split("## Parameters (machine-readable)", 1)[1]
    return json.loads(re.search(r"```json\n(.*?)\n```", section, re.S).group(1))


def test_the_plan_s_parameters_are_the_code_s():
    p = plan_parameters()
    lock = ModelLock.load(REAL_MODELS_LOCK)
    assert p["questions_fingerprint"] == questions_fingerprint(load_golden(default_golden_path(1)).questions)
    assert p["embedder"] == {"repo": lock["embedder"].repo, "revision": lock["embedder"].revision}
    assert p["reranker"] == {"repo": lock["reranker"].repo, "revision": lock["reranker"].revision}
    assert p["query_instruction"] == neural.QUERY_INSTRUCTION
    assert (p["rrf_k"], p["rerank_depth"], p["tie_seed"], p["threads"], p["max_tokens"]) == \
        (neural.RRF_K, neural.RERANK_DEPTH, neural.TIE_SEED, neural.THREADS, neural.MAX_TOKENS)
    assert p["configurations"] == len(EVALUATED) and p["extras"] == [c.name for c in EXTRAS]
    assert (p["primary_metric"], p["alpha"]) == (evaluate.PRIMARY_METRIC, evaluate.ALPHA)
    assert p["chain"] == [[a.name, b.name, predicted] for a, b, predicted in CHAIN]
    assert p["supporting_metrics"] == list(evaluate.SUPPORTING_METRICS)
    assert p["interaction_metric"] == evaluate.INTERACTION_METRICS[0]
    assert (p["budgets"], p["headline_budget"]) == (list(metrics.BUDGETS), metrics.HEADLINE_BUDGET)
    assert (p["resamples"], p["seed"]) == (stats.RESAMPLES, stats.SEED)
    assert p["pool"] == {"depth": 10, "window_budget": metrics.HEADLINE_BUDGET,
                         "window_chunkers": list(v2_labels.WINDOW_CHUNKERS)}
    assert p["calibration"] == v2_labels.CALIBRATION
    assert p["agreement"] == {str(k): v for k, v in v2_labels.AGREEMENT.items()}
    assert (p["paraphrase_share"], p["bridge"], p["batch_items"], p["judge_model"]) == \
        (v2_labels.PARAPHRASE_SHARE, v2_labels.BRIDGE, v2_labels.BATCH_ITEMS, v2_labels.JUDGE_MODEL)


def test_every_run_records_which_plan_it_follows():
    assert evaluate.plan_path() == STAGE / "PLAN.md"


def test_the_prompts_the_plan_names_exist_and_v2_differs_from_v1_by_one_sentence():
    prompts = STAGE.parent / "golden" / "prompts"
    for kind in ("pool", "blind"):
        v1 = (prompts / f"v1-{kind}-judge.md").read_text(encoding="utf-8").splitlines()
        v2 = (prompts / f"v2-{kind}-judge.md").read_text(encoding="utf-8").splitlines()
        changed = [(a, b) for a, b in zip(v1, v2) if a != b]
        assert len(v1) == len(v2) and len(changed) == 1
        assert changed[0][1] == "- Open no file other than the input file. " + changed[0][0].removeprefix("- ")
    assert "{input}" in (prompts / "v2-quote-judge.md").read_text(encoding="utf-8")


def _pins(path: Path) -> dict[str, str]:
    pins = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("\\")[0].strip()
        if line and not line.startswith(("#", "--")) and "==" in line:
            name, version = line.split("==")
            pins[name.lower().replace("_", "-")] = version.strip()
    return pins


def test_stage_3_s_lock_keeps_every_stage_2_pin_and_pins_every_wheel_by_hash():
    # Both stages share one environment, so Stage 3's install must not move anything Stage 2 locked.
    stage2, stage3 = _pins(STAGE.parent / "Stage 2" / "requirements-lock.txt"), _pins(STAGE / "requirements-lock.txt")
    assert {name: stage3.get(name) for name in stage2} == stage2
    text = (STAGE / "requirements-lock.txt").read_text(encoding="utf-8")
    assert text.count("--hash=sha256:") == len(stage3) >= len(stage2)
    for name in ("torch", "sentence-transformers", "transformers", "tokenizers", "safetensors"):
        assert name in stage3
