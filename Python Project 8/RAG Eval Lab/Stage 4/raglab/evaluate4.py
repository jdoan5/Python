"""Stage 4's confirmatory tests and the instrument check (PLAN.md, "The confirmatory tests").

Everything statistical is Stage 3's, unchanged: `paired_test` (two-sided paired
sign-flip test, per question and per target page), `_verdict` and `holm`. What
is new: the inputs (each answer's outcome, per judged run), the rules for
unjudged answers in both directions, the quote-failure rule, S2's per-run
eligibility, and the instrument check that decides how the README words link 1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from raglab import params
from raglab.evaluate import ALPHA, _verdict, paired_test
from raglab.stats import RESAMPLES, SEED, bootstrap_indices, cohen_kappa, holm


@dataclass(frozen=True)
class ArmScores:
    """One arm's per-answer outcomes, shaped (questions, judged runs). NaN marks "not applicable"."""

    sc: np.ndarray
    correct: np.ndarray
    support_rate: np.ndarray       # NaN where the answer made no claim or F didn't judge it
    penalized: np.ndarray
    sc_quote_relaxed: np.ndarray   # SC counting a failed quote as supported where F said "supported"
    unjudged_f: np.ndarray         # bool
    unjudged_c: np.ndarray         # bool
    lost_citation: np.ndarray      # bool: a citation failed the integrity check and was dropped
    sc_rekeyed: np.ndarray | None = None  # SC with the reference check's re-keyed C verdicts where a key was rebuilt

    @property
    def runs(self) -> int:
        return self.sc.shape[1]


def _per_question(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, float).mean(axis=1)


def _test(questions, a, b, predicted=1) -> dict:
    n = len(questions)
    t = paired_test(questions, np.asarray(a, float), np.asarray(b, float), predicted,
                    bootstrap_indices(n, RESAMPLES, SEED), RESAMPLES, SEED)
    t["verdict"] = _verdict(t["delta"], t["hi"], t["p"], predicted)
    return t


def _set(x: np.ndarray, mask: np.ndarray, value: float) -> np.ndarray:
    out = np.asarray(x, float).copy()
    out[mask] = value
    return out


REFERENCE_CHECK_STATES = ("not needed", "run", "not run")


def confirmatory(questions: Sequence, arms: Mapping[str, ArmScores], reference_check: str = "not needed") -> dict:
    """The chain and its supporting family, with every pre-registered sensitivity rule applied.

    `reference_check` is "not needed" when no key was rebuilt, "run" when the
    re-keyed C verdicts are in each arm's `sc_rekeyed`, and "not run" when the
    scope rule left the item out (the README then says so).
    """
    if reference_check not in REFERENCE_CHECK_STATES:
        raise ValueError(f"reference_check must be one of {REFERENCE_CHECK_STATES}")
    (base_name, hhr_name, _), (bm25_name, _, _) = params.CHAIN
    base, hhr = arms[base_name], arms[hhr_name]
    if base.runs != hhr.runs:
        raise ValueError("link 1's arms must average the same number of judged runs")
    sc_hit_b = base.unjudged_f | base.unjudged_c | base.lost_citation
    sc_hit_h = hhr.unjudged_f | hhr.unjudged_c | hhr.lost_citation
    n_hit = int(sc_hit_b.sum() + sc_hit_h.sum())

    link1 = _test(questions, _per_question(base.sc), _per_question(hhr.sc))
    link1.update({"metric": params.PRIMARY_METRIC, "significant": link1["p"] < ALPHA, "judged_runs": base.runs,
                  "unjudged": {base_name: int(sc_hit_b.sum()), hhr_name: int(sc_hit_h.sum())}})
    if link1["verdict"] == "supported" and n_hit:
        # The worst case for the prediction: every unjudged baseline answer was SC 1, every unjudged hhr answer 0.
        worst = _test(questions, _per_question(_set(base.sc, sc_hit_b, 1)), _per_question(_set(hhr.sc, sc_hit_h, 0)))
        link1["worst_case"] = {"delta": worst["delta"], "p": worst["p"]}
        if worst["p"] >= ALPHA or worst["delta"] <= 0:
            link1["verdict"] = f"supported, sensitive to {n_hit} unjudged answers"
    if link1["verdict"] == "contradicted" and n_hit:
        # The mirror: the worst case against the contradiction.
        best = _test(questions, _per_question(_set(base.sc, sc_hit_b, 0)), _per_question(_set(hhr.sc, sc_hit_h, 1)))
        link1["best_case"] = {"delta": best["delta"], "p": best["p"]}
        if best["verdict"] != "contradicted":
            link1["verdict"] = f"contradicted, sensitive to {n_hit} unjudged answers"
    if link1["verdict"].startswith("supported"):
        relaxed = _test(questions, _per_question(base.sc_quote_relaxed), _per_question(hhr.sc_quote_relaxed))
        link1["quote_relaxed"] = {"delta": relaxed["delta"], "p": relaxed["p"]}
        if relaxed["p"] >= ALPHA:
            link1["verdict"] += ", sensitive to quote failures"
    link1["reference_check"] = {"state": reference_check}
    if reference_check == "run":
        if base.sc_rekeyed is None or hhr.sc_rekeyed is None:
            raise ValueError("the reference check ran, but an arm has no re-keyed SC")
        rekeyed = _test(questions, _per_question(base.sc_rekeyed), _per_question(hhr.sc_rekeyed))
        link1["reference_check"].update({"delta": rekeyed["delta"], "p": rekeyed["p"]})
        if link1["verdict"].startswith("supported") and rekeyed["p"] >= ALPHA:
            link1["verdict"] += ", sensitive to the reference top-up"
    chain = [link1]
    if bm25_name in arms:
        link2 = _test(questions, _per_question(arms[bm25_name].sc), _per_question(hhr.sc))
        link2["tested"] = bool(link1["significant"])
        if not link2["tested"]:
            link2["verdict"] = "not reached: link 1 was not significant, so this is exploratory"
        chain.append(link2)
    else:
        chain.append({"tested": False, "verdict": "not run: heading+bm25 was not judged (the scope rule)"})

    supporting = _supporting(questions, base, hhr, link1["significant"])
    return {"alpha": ALPHA, "chain": chain, "supporting": supporting}


def _support_rate_differences(base: ArmScores, hhr: ArmScores, a=None, b=None) -> tuple[np.ndarray, np.ndarray, dict]:
    """S2: per run, eligible only where both arms made a claim; each question averages its eligible runs.

    A question drops out when no run is eligible. The count is reported, with
    the ineligible runs of those questions split by which arm made no claim.
    """
    a = np.asarray(base.support_rate if a is None else a, float)
    b = np.asarray(hhr.support_rate if b is None else b, float)
    eligible = ~(np.isnan(a) | np.isnan(b))
    diffs = np.where(eligible, b - a, np.nan)
    keep = eligible.any(axis=1)
    per_q = np.array([np.nanmean(row) for row in diffs[keep]]) if keep.any() else np.array([])
    out_a, out_b = np.isnan(a)[~keep], np.isnan(b)[~keep]
    dropped = {"questions": int((~keep).sum()),
               "runs with no claim in the baseline only": int((out_a & ~out_b).sum()),
               "runs with no claim in hhr only": int((out_b & ~out_a).sum()),
               "runs with no claim in either": int((out_a & out_b).sum())}
    return per_q, keep, dropped


def _supporting(questions, base: ArmScores, hhr: ArmScores, link1_significant: bool) -> list[dict]:
    rows = []
    # The worst case covers every unjudged or citation-losing answer (PLAN.md, "Sensitivity rules").
    specs = (("correct", base.unjudged_c | base.lost_citation, hhr.unjudged_c | hhr.lost_citation, 1.0, 0.0),
             ("penalized", base.unjudged_f | base.unjudged_c | base.lost_citation,
              hhr.unjudged_f | hhr.unjudged_c | hhr.lost_citation, 1.0, -1.0))
    for metric, hit_b, hit_h, best_b, worst_h in specs:
        a, b = getattr(base, metric), getattr(hhr, metric)
        t = _test(questions, _per_question(a), _per_question(b))
        worst = _test(questions, _per_question(_set(a, hit_b, best_b)), _per_question(_set(b, hit_h, worst_h)))
        t.update({"metric": metric, "questions": len(questions), "worst_p": worst["p"]})
        rows.append(t)
    per_q, keep, dropped = _support_rate_differences(base, hhr)
    kept = [q for q, k in zip(questions, keep) if k]
    t = _test(kept, np.zeros(len(per_q)), per_q) if len(kept) > 1 else {
        "delta": float("nan"), "lo": float("nan"), "hi": float("nan"), "p": 1.0, "page_level": {"p": 1.0},
        "verdict": "inconclusive"}
    # F-unjudged answers are already out (their rate is NaN). A citation-losing answer that F judged is set to
    # the worst case: rate 1 in the baseline, 0 in hhr, where it is eligible.
    worst_a = np.where(base.lost_citation & ~np.isnan(np.asarray(base.support_rate, float)), 1.0, base.support_rate)
    worst_b = np.where(hhr.lost_citation & ~np.isnan(np.asarray(hhr.support_rate, float)), 0.0, hhr.support_rate)
    w_per_q, w_keep, _ = _support_rate_differences(base, hhr, worst_a, worst_b)
    w_kept = [q for q, k in zip(questions, w_keep) if k]
    worst_p = _test(w_kept, np.zeros(len(w_per_q)), w_per_q)["p"] if len(w_kept) > 1 else 1.0
    t.update({"metric": "claim_support_rate", "questions": len(kept), "dropped": dropped, "worst_p": worst_p})
    rows.insert(1, t)
    order = {m: i for i, m in enumerate(params.SUPPORTING_METRICS)}
    rows.sort(key=lambda r: order[r["metric"]])
    for t, p_holm, worst_holm, page_holm in zip(rows, holm([r["p"] for r in rows]), holm([r["worst_p"] for r in rows]),
                                                holm([r["page_level"]["p"] for r in rows])):
        t["p_holm"], t["worst_p_holm"], t["page_level"]["p_holm"] = p_holm, worst_holm, page_holm
        t["fragile"] = bool(p_holm < ALPHA <= page_holm)
        if not link1_significant:
            t["verdict"] = "exploratory: link 1 was not significant"
            continue
        t["verdict"] = _verdict(t["delta"], t["hi"], p_holm, 1)
        if t["verdict"] == "supported" and worst_holm >= ALPHA:
            t["verdict"] = "supported, sensitive to unjudged answers"
    return rows


# ---------------------------------------------------------------------------------------------------- the check


def kappa_or_none(a: Sequence[int], b: Sequence[int]) -> float | None:
    """Cohen's kappa on two binary label lists; None when either rater used only one value (kappa is undefined)."""
    if not a or len(set(a)) < 2 or len(set(b)) < 2:
        return None
    return cohen_kappa(list(a), list(b), categories=(0, 1))


SWAP_GROUPS = ("hhr-4.8k-med", "base-4.8k-med")        # per arm
PLANT_GROUPS = ("plant-cross", "plant-shape")            # per kind


def instrument_check(swaps_caught: Mapping[str, tuple[int, int]], plants_kept: Mapping[str, tuple[int, int]],
                     judge_sc: Sequence[int], human_sc: Sequence[int], paired_gap_ci: tuple[float, float]) -> dict:
    """PLAN.md's instrument check.

    swaps_caught: per arm, (caught, planted) for the near-miss swaps (leniency, both arms).
    plants_kept: per kind, (kept supported, planted) for the true-support plants (strictness).
    judge_sc / human_sc: SC on the hand-graded answers, quotes off on the judges' side (part 3).
    paired_gap_ci: the judges' paired link-1 difference minus the grader's, 95% interval (part 4).
    """
    g = params.INSTRUMENT_CHECK
    kappa = kappa_or_none(list(judge_sc), list(human_sc))
    parts = {
        "near_miss_swaps": gate_passes(swaps_caught, g["swaps"], g["swaps_gate"], g["swaps_gate_per_arm"],
                                       SWAP_GROUPS),
        "true_support_plants": gate_passes(plants_kept, g["plants"], g["plants_gate"], g["plants_gate_per_kind"],
                                           PLANT_GROUPS),
        "kappa": kappa is not None and kappa >= g["kappa"],
        "bias": bool(paired_gap_ci[0] <= 0 <= paired_gap_ci[1]),
    }
    return {"swaps_caught": dict(swaps_caught), "plants_kept": dict(plants_kept), "kappa": kappa,
            "paired_gap_ci": list(paired_gap_ci), "parts": parts, "passes": all(parts.values())}


SHORTFALL_MINIMUM = 6  # fewer judged in an arm or kind than this, and a gate can't be read: it fails


def gate_passes(groups: Mapping[str, tuple[int, int]], planned: int, gate: int, gate_per_group: int,
                expected: Sequence[str] | None = None) -> bool:
    """A planted gate: at least `gate` of the `planned` items right overall and `gate_per_group` of each group's
    share. If fewer qualified (PLAN.md, "Checks on the judges"), the same shares hold on those judged
    (90% overall, 80% per group), with at least 6 in every group; below that the gate fails, not computable.
    `expected` names every group the gate needs (both arms, or both plant kinds): a missing one counts as none
    judged, so a gate can't pass on one arm alone."""
    groups = {g: groups.get(g, (0, 0)) for g in (expected or groups)} | dict(groups)
    judged = [n for _, n in groups.values()]
    right = sum(r for r, _ in groups.values())
    total = sum(judged)
    if not groups or any(n < SHORTFALL_MINIMUM for n in judged):
        return False
    per_group = planned // len(groups)
    if total >= planned and all(n >= per_group for n in judged):
        return right >= gate and all(r >= gate_per_group for r, _ in groups.values())
    return right >= total * gate / planned and all(r >= n * gate_per_group / per_group for r, n in groups.values())


def citation_failure_stop(arms: Mapping[str, tuple[int, int]]) -> list[str]:
    """The golden arms where more than 0.5% of citations failed their check: a mapping bug stops the stage.
    `arms` maps an arm to (failed, total) citations."""
    return [arm for arm, (failed, total) in arms.items() if total and failed / total > params.CITATION_FAILURE_STOP]


def lenient(sc_on_no_windows: Sequence[int]) -> dict:
    """The no-answer-window check: SC = 1 on a window S calls "no" is an error of the judges or the labels. The
    judges are flagged lenient if that share, among judged answers there that don't abstain, has a 95% Wilson
    lower bound above 0.05."""
    from raglab.stats import wilson_ci

    if not sc_on_no_windows:
        return {"n": 0, "share": None, "lower": None, "lenient": False}
    share, lo, _ = wilson_ci([float(x) for x in sc_on_no_windows])
    return {"n": len(sc_on_no_windows), "share": share, "lower": lo, "lenient": lo > params.LENIENCY_WILSON_LOWER}


def stated_verdict(link1_verdict: str, check: Mapping) -> str:
    """How the README words link 1 (PLAN.md, the instrument check's table). The bias line is printed beside it
    whatever the outcome: a passing check never reads as confirming the absence of an arm-specific error."""
    if link1_verdict != "supported":
        return link1_verdict  # anything else, a qualified "supported, sensitive to ..." included, stands as it is
    if check["passes"]:
        return link1_verdict + " (no gross judge error found)"
    failed = ", ".join(name for name, ok in check["parts"].items() if not ok)
    if check["kappa"] is None:
        failed += " (kappa not computable)"
    return f"{link1_verdict} on the Opus judges' grades, not confirmed by the instrument check ({failed} failed)"
