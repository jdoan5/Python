"""The two question sets Stage 4 asks: the pilot's 12 made-up questions and the 98 golden ones.

Both give the same view: a question's text, its target, and its grade-2
sections (for the masked and oracle windows and for the keys). The pilot's
labels come from pilot/labels.json; the golden set's from golden/v3.yaml once
it exists, and v2.yaml before.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import yaml

from raglab.golden import default_golden_path, load_golden

STAGE = Path(__file__).resolve().parents[1]
PILOT_CONDITIONS = ("hhr-4.8k-med", "base-4.8k-med", "base-4.8k-med-notitle", "closed-med", "hhr-4.8k-off",
                    "hhr-4.8k-low", "hhr-4.8k-high", "hhr-4.8k-xhigh", "hhr-4.8k-max", "hhr-9.6k-med")
PILOT_RUNS = {"hhr-4.8k-med": 2}


@dataclass(frozen=True)
class AskedQuestion:
    id: str
    text: str
    style: str
    target: str | None
    answering: frozenset[str]


def pilot_questions(stage: Path = STAGE) -> list[AskedQuestion]:
    data = yaml.safe_load((stage / "pilot" / "questions.yaml").read_text(encoding="utf-8"))
    labels_path = stage / "pilot" / "labels.json"
    labels = json.loads(labels_path.read_text(encoding="utf-8"))["labels"] if labels_path.exists() else {}
    out = []
    for q in data["questions"]:
        grades = labels.get(q["id"], {})
        answering = frozenset(loc for loc, g in grades.items() if g == 2)
        out.append(AskedQuestion(q["id"], q["question"], q["style"], q.get("target"), answering))
    return out


def golden_questions(allow_v2: bool = False) -> list[AskedQuestion]:
    """The 98 golden questions with v3's labels. Golden requests are never planned on v2's labels (the masked
    and oracle windows depend on them); only exploratory code may ask for v2 explicitly."""
    v3 = default_golden_path(2).with_name("v3.yaml")
    if not v3.exists() and not allow_v2:
        raise FileNotFoundError("golden/v3.yaml doesn't exist yet: golden requests wait for v3's labels")
    golden = load_golden(v3 if v3.exists() else default_golden_path(2))
    return [AskedQuestion(q.id, q.text, q.style, q.target, frozenset(q.primary)) for q in golden.questions]


def plan_path(which: str, stage: Path = STAGE) -> Path:
    return stage / "pilot" / "requests.jsonl" if which == "pilot" else stage / "plan" / "requests.jsonl"
