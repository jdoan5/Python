"""The pilot's rules, applied by `pilot report` before freeze point 2 (PLAN.md, "The pilot").

Fixed at freeze point 1. No pilot result changes a test, a metric or a
decision rule; these rules set operating values (max_tokens, the one allowed
prompt fix, whether low judge effort is allowed, unit costs) and the power row
the README quotes.
"""

from __future__ import annotations

import math
from decimal import Decimal
from typing import Mapping, Sequence

from raglab import params
from raglab.units import Answer, looks_like_abstention


def nearest_rank(values: Sequence[float], pct: float) -> float:
    """The pct-th percentile by nearest rank: an observed value, never an interpolation."""
    if not values:
        raise ValueError("no values")
    ordered = sorted(values)
    return ordered[max(0, math.ceil(pct / 100 * len(ordered)) - 1)]


def clamp(x: float, bounds: Sequence[int]) -> int:
    return int(min(max(x, bounds[0]), bounds[1]))


def max_tokens_rule(haiku_outputs: Mapping[str, Sequence[int]], judge_outputs: Mapping[str, Sequence[int]]) -> dict:
    """Rule 1: 3 x the pilot's 99th-percentile output, within the plan's bounds.

    haiku_outputs maps an effort level ("off" for thinking disabled) to output
    token counts; judge_outputs maps "support" and "correctness" to theirs.
    """
    b = params.MAX_TOKENS_BOUNDS
    haiku = {effort: clamp(3 * nearest_rank(v, 99), b["haiku_xhigh_max"] if effort in ("xhigh", "max") else b["haiku"])
             for effort, v in haiku_outputs.items()}
    judges = {kind: clamp(3 * nearest_rank(v, 99), b["judge"]) for kind, v in judge_outputs.items()}
    return {"haiku": haiku, "judge": judges}


def format_failures(answers: Sequence[Answer], labels: Sequence[str] | None = None) -> list[str]:
    """Rule 2's format checks, one line per failed check, each named by its answer's label (condition, question
    and run in the pilot report; its index otherwise). One answer can fail two checks."""
    out = []
    for n, a in enumerate(answers):
        name = labels[n] if labels is not None else f"answer {n}"
        if a.status != "ok":
            continue
        if not a.abstained and looks_like_abstention(a.text) and not any(u.citations for u in a.units):
            out.append(f"{name}: an abstention not in the prescribed words")
        elif not a.abstained and not any(u.citations for u in a.units):
            out.append(f"{name}: answers without citing anything")
        if len(a.text.split()) > 300:
            out.append(f"{name}: {len(a.text.split())} words")
    return out


def code_citation_share(answers: Sequence[Answer]) -> float | None:
    code = [u for a in answers if a.status == "ok" for u in a.units if u.is_code]
    return sum(bool(u.citations) for u in code) / len(code) if code else None


def prose_citation_share(answers: Sequence[Answer]) -> float | None:
    """The share of units of 40 or more non-whitespace characters, headings aside, that carry a citation:
    the code gate's units (units.code_gate), among answers that don't abstain."""
    long = [u for a in answers if a.status == "ok" and not a.abstained for u in a.units
            if not u.text.lstrip().startswith("#") and len("".join(u.text.split())) >= params.CODE_GATE_MIN_CHARS]
    return sum(bool(u.citations) for u in long) / len(long) if long else None


def prompt_fix_needed(by_condition: Mapping[str, Sequence[Answer]], hhr: Sequence[str] = ("hhr-4.8k-med",),
                      baseline: Sequence[str] = ("base-4.8k-med",),
                      labels: Mapping[str, Sequence[str]] | None = None) -> dict:
    """Rule 2: the answer prompt may be fixed once, and only on these four grounds.

    `by_condition` holds the answer-v1 answers by condition, closed-book left
    out; only answers with status ok are checked (truncations are reported
    separately). The citation-share grounds are read in the hhr and baseline
    answers, each arm on its own.
    """
    answers = [a for name, group in by_condition.items() if name != "closed-med" for a in group]
    failures = []
    for name, group in by_condition.items():
        if name == "closed-med":
            continue
        names = [f"{name} {x}" for x in labels[name]] if labels else [f"{name} answer {n}" for n in range(len(group))]
        failures += format_failures(group, names)
    failing = {line.split(":", 1)[0] for line in failures}  # one answer may fail two checks
    code = code_citation_share(answers)
    shares = {"hhr": prose_citation_share([a for c in hhr for a in by_condition.get(c, ())]),
              "baseline": prose_citation_share([a for c in baseline for a in by_condition.get(c, ())])}
    need = params.PILOT["unit_citation_share"]
    reasons = []
    if len(failing) > params.PILOT["format_failures_allowed"]:
        reasons.append(f"{len(failing)} answers fail a format check")
    if code is not None and code < params.PILOT["code_citation_share"]:
        reasons.append(f"only {code:.0%} of code units carry a citation")
    for arm, share in shares.items():
        if share is not None and share < need:
            reasons.append(f"only {share:.0%} of the {arm} answers' units of {params.CODE_GATE_MIN_CHARS}+ "
                           "characters carry a citation")
    if None not in shares.values() and abs(shares["hhr"] - shares["baseline"]) > params.PILOT["unit_citation_gap"]:
        reasons.append(f"the arms' cited shares differ by {abs(shares['hhr'] - shares['baseline']):.0%}")
    truncated = sum(a.status == "truncated" for a in answers)
    return {"needed": bool(reasons), "reasons": reasons, "format_failures": failures, "failing_answers": len(failing),
            "code_citation_share": code, "prose_citation_share": shares, "truncated": truncated}


def uncited_claim_share(units_by_arm: Mapping[str, Sequence[tuple[bool, bool]]]) -> dict[str, float | None]:
    """Reported with rule 2: per arm, the share of uncited units that F labels claims.
    Each item is (cited, F says claim) for one unit."""
    out = {}
    for arm, units in units_by_arm.items():
        uncited = [claim for cited, claim in units if not cited]
        out[arm] = sum(uncited) / len(uncited) if uncited else None
    return out


def agreement(a: Sequence, b: Sequence) -> float | None:
    pairs = list(zip(a, b, strict=True))
    return sum(x == y and x is not None for x, y in pairs) / len(pairs) if pairs else None


def low_effort_rule(arms: Mapping[str, Mapping[str, tuple[Sequence, Sequence]]],
                    swaps_low: Mapping[str, Mapping[str, tuple[int, int]]]) -> dict:
    """Rule 3: low effort passes only if, in each arm separately, it agrees with medium on at least 11 of the 12
    SC verdicts and on 90% of unit labels and of nugget labels, and it catches at least 11 of the 12 swaps
    overall and 5 of the 6 in each arm.

    `arms[arm]` maps "units", "nuggets" and "sc" to (medium, low) label lists in
    the same order; an unjudged verdict is None and never agrees. `swaps_low[arm]`
    maps each kind (swap-lexical, swap-topical) to (caught, judged). An arm with
    fewer than 6 judged swaps fails: the check can't be made.
    """
    p = params.PILOT
    per_arm, passes = {}, True
    for arm, labels in arms.items():
        units, nuggets = agreement(*labels["units"]), agreement(*labels["nuggets"])
        sc_medium, sc_low = labels["sc"]
        sc_agree = sum(x == y and x is not None for x, y in zip(sc_medium, sc_low, strict=True))
        kinds = swaps_low.get(arm, {})
        caught = sum(c for c, _ in kinds.values())
        judged = sum(j for _, j in kinds.values())
        ok = (units is not None and units >= p["low_effort_agreement"]
              and nuggets is not None and nuggets >= p["low_effort_agreement"]
              and sc_agree >= p["low_effort_sc_agree"] and judged >= p["swaps_per_arm"]
              and caught >= p["swap_gate_per_arm"])
        per_arm[arm] = {"units": units, "nuggets": nuggets, "sc_agree": sc_agree, "sc_compared": len(sc_medium),
                        "swaps": {k: list(v) for k, v in kinds.items()}, "caught": caught, "judged": judged,
                        "passes": ok}
        passes = passes and ok
    total = sum(v["caught"] for v in per_arm.values())
    judged_all = sum(v["judged"] for v in per_arm.values())
    passes = passes and total >= p["swap_gate"] and judged_all >= p["swaps"] and len(per_arm) == 2
    return {"arms": per_arm, "swaps_caught": total, "swaps_judged": judged_all, "passes": passes}


def flip_rate(first: Sequence[int], second: Sequence[int]) -> float | None:
    """Rule 4: how often judging the same answer again changes its SC."""
    pairs = list(zip(first, second, strict=True))
    return sum(x != y for x, y in pairs) / len(pairs) if pairs else None


def unit_costs(usages: Mapping[str, Sequence[Mapping]]) -> dict[str, dict]:
    """Rule 5: for each call kind ("F-medium", "C-low", ...), the mean output tokens per call from the pilot's
    settled usage, thinking included. The scope rule prices an item as its counted input at the batch input rate
    plus n x this mean at the batch output rate, so input is never counted twice. A sum of hundreds of calls sits
    near n x the mean, so it projects with the mean (x its factor), never a high percentile."""
    out = {}
    for kind, rows in usages.items():
        if not rows:
            continue
        mean = Decimal(sum(int(u.get("output_tokens") or 0) for u in rows)) / len(rows)
        out[kind] = {"calls": len(rows), "mean_output_tokens": str(mean.quantize(Decimal("0.1")))}
    return out


def integrity_stop(answers: Sequence[Answer]) -> list[str]:
    """Rule 6: any citation that failed its check in the pilot stops the stage."""
    return [f"answer {n}: {f}" for n, a in enumerate(answers) for f in a.failures]
