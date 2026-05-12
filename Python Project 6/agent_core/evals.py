from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable

from agent_core.agent import AgentResult
from agent_core.llm import get_client


@dataclass
class EvalCase:
    name: str
    input: str
    rubric: str


@dataclass
class EvalResult:
    case: EvalCase
    score: int
    rationale: str
    agent_output: str
    iterations: int


JUDGE_SYSTEM = """You are a strict evaluator. Score the agent's output from 1-5 against the rubric.

- 5: Fully meets every rubric criterion with high quality.
- 4: Meets all criteria with minor issues.
- 3: Meets most criteria; one significant gap.
- 2: Meets some criteria; major gaps.
- 1: Fails to address the rubric.

Respond with JSON only: {"score": <int>, "rationale": "<one sentence>"}
"""


def judge(rubric: str, agent_output: str) -> tuple[int, str]:
    client = get_client()
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        system=JUDGE_SYSTEM,
        messages=[
            {
                "role": "user",
                "content": f"Rubric:\n{rubric}\n\nAgent output:\n{agent_output}",
            }
        ],
    )
    text = next((b.text for b in response.content if b.type == "text"), "{}")
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        parsed = json.loads(text[start:end])
        return int(parsed.get("score", 0)), str(parsed.get("rationale", ""))
    except (ValueError, json.JSONDecodeError):
        return 0, f"Could not parse judge response: {text}"


def run_evals(
    cases: list[EvalCase],
    agent_fn: Callable[[str], AgentResult],
) -> list[EvalResult]:
    results = []
    for case in cases:
        agent_result = agent_fn(case.input)
        score, rationale = judge(case.rubric, agent_result.final_text)
        results.append(
            EvalResult(
                case=case,
                score=score,
                rationale=rationale,
                agent_output=agent_result.final_text,
                iterations=agent_result.iterations,
            )
        )
    return results
