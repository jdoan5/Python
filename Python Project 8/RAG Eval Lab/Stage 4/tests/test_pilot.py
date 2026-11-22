"""The pilot's rules: max_tokens, the one prompt fix, low judge effort, flip rate, unit costs, integrity."""

from __future__ import annotations

from decimal import Decimal

from raglab.pilot import (
    code_citation_share, flip_rate, integrity_stop, low_effort_rule, max_tokens_rule, nearest_rank, prompt_fix_needed,
    uncited_claim_share, unit_costs,
)
from raglab.units import Answer, Unit


def answer(text, cited=True, code=False, abstained=False, failures=()):
    unit = Unit("U1", 0, len(text), text, code, ("c",) if cited else ())
    return Answer(text, "ok", None, (unit,), 1 if cited else 0, 0, abstained, tuple(failures))


def test_max_tokens_is_three_times_the_99th_percentile_within_bounds():
    out = max_tokens_rule({"medium": [800, 1000, 1200, 5000], "max": [9000, 15000], "off": [200, 300]},
                          {"support": [1500, 2000], "correctness": [900, 5000]})
    assert out["haiku"] == {"medium": 15000, "max": 32000, "off": 4000}  # 15,000; 45,000 capped; 900 floored
    assert out["judge"] == {"support": 6000, "correctness": 12000}
    assert nearest_rank([1, 2, 3, 4], 50) == 2


FINE = [answer("Use `Depends` for that.")] * 10 + [answer("The provided documentation does not answer this question.",
                                                       cited=False, abstained=True)]
LONG = "Use the `Depends` function to declare the dependency here."  # 40+ non-whitespace characters


def test_rule_2_ground_1_format_failures() -> None:
    assert not prompt_fix_needed({"hhr-4.8k-med": FINE, "base-4.8k-med": FINE})["needed"]
    two = FINE + [answer("Use a middleware.", cited=False), answer("The docs don't mention it.", cited=False)]
    out = prompt_fix_needed({"hhr-4.8k-off": two, "hhr-4.8k-med": FINE, "base-4.8k-med": FINE})
    assert out["needed"] and "2 answers fail a format check" in out["reasons"][0]


def test_rule_2_ground_2_code_units_must_carry_citations() -> None:
    code = [answer("```py\nx\n```", cited=False, code=True)] * 2 + [answer("```py\ny\n```", code=True)] * 8
    assert code_citation_share(code) == 0.8
    out = prompt_fix_needed({"hhr-4.8k-high": code, "hhr-4.8k-med": FINE, "base-4.8k-med": FINE})
    assert out["needed"] and any("code units" in r for r in out["reasons"])


def _with_long_units(cited: int, uncited: int) -> list:
    return [answer(LONG)] * cited + [Answer(LONG, "ok", None, (Unit("U1", 0, len(LONG), LONG, False, ("c",)),
                                                                Unit("U2", 0, len(LONG), LONG, False, ())), 1, 0,
                                            False)] * uncited


def test_rule_2_ground_3_long_units_in_each_arm_must_carry_citations() -> None:
    hhr = _with_long_units(8, 2)  # 2 uncited of 12 long units: 83%
    out = prompt_fix_needed({"hhr-4.8k-med": hhr, "base-4.8k-med": _with_long_units(5, 2)})
    assert out["needed"] and any("hhr" in r for r in out["reasons"])
    assert round(out["prose_citation_share"]["hhr"], 2) == 0.83


def test_rule_2_ground_4_the_arms_may_not_differ_by_more_than_10_points() -> None:
    hhr, base = _with_long_units(40, 0), _with_long_units(36, 4)  # 100% against 91%: no single arm under 90%
    assert not prompt_fix_needed({"hhr-4.8k-med": hhr, "base-4.8k-med": base})["needed"]
    base = _with_long_units(0, 30)  # 50%
    out = prompt_fix_needed({"hhr-4.8k-med": hhr, "base-4.8k-med": base})
    assert any("differ by" in r for r in out["reasons"])


def test_rule_2_leaves_out_closed_book_and_counts_truncations() -> None:
    closed = [answer("Use a middleware.", cited=False)] * 5
    truncated = [Answer("cut", "truncated", None, (), 0, 0, False)]
    out = prompt_fix_needed({"closed-med": closed, "hhr-4.8k-med": FINE + truncated, "base-4.8k-med": FINE})
    assert not out["needed"] and out["truncated"] == 1


def test_uncited_claims_are_reported_per_arm() -> None:
    assert uncited_claim_share({"hhr": [(True, True), (False, True), (False, False)], "baseline": [(True, True)]}) == \
        {"hhr": 0.5, "baseline": None}


def arm(units=0.95, nuggets=1.0, sc_disagree=0):
    n = 20
    med = ["supported"] * n
    low = ["partial"] * round(n * (1 - units)) + ["supported"] * (n - round(n * (1 - units)))
    sc = [1] * 12
    return {"units": (med, low), "nuggets": (["yes"] * 10, ["yes"] * round(10 * nuggets) + ["no"] * (10 - round(10 * nuggets))),
            "sc": (sc, [0] * sc_disagree + sc[sc_disagree:])}


SWAPS_OK = {"swap-lexical": (3, 3), "swap-topical": (3, 3)}


def test_rule_3_low_effort_passes_only_in_both_arms() -> None:
    assert low_effort_rule({"hhr": arm(), "baseline": arm()}, {"hhr": SWAPS_OK, "baseline": SWAPS_OK})["passes"]
    assert not low_effort_rule({"hhr": arm(units=0.85), "baseline": arm()},
                               {"hhr": SWAPS_OK, "baseline": SWAPS_OK})["passes"]
    assert not low_effort_rule({"hhr": arm(), "baseline": arm(sc_disagree=2)},
                               {"hhr": SWAPS_OK, "baseline": SWAPS_OK})["passes"]
    assert low_effort_rule({"hhr": arm(), "baseline": arm(sc_disagree=1)},
                           {"hhr": SWAPS_OK, "baseline": SWAPS_OK})["passes"]  # 11 of 12


def test_rule_3_swaps_11_of_12_overall_and_5_of_6_per_arm() -> None:
    one_miss = {"swap-lexical": (2, 3), "swap-topical": (3, 3)}
    two_miss = {"swap-lexical": (2, 3), "swap-topical": (2, 3)}
    assert low_effort_rule({"hhr": arm(), "baseline": arm()}, {"hhr": one_miss, "baseline": SWAPS_OK})["passes"]
    assert not low_effort_rule({"hhr": arm(), "baseline": arm()}, {"hhr": two_miss, "baseline": SWAPS_OK})["passes"]
    assert not low_effort_rule({"hhr": arm(), "baseline": arm()}, {"hhr": one_miss, "baseline": one_miss})["passes"]
    short = {"swap-lexical": (3, 3), "swap-topical": (2, 2)}  # only 5 qualified: the check can't be made
    assert not low_effort_rule({"hhr": arm(), "baseline": arm()}, {"hhr": short, "baseline": SWAPS_OK})["passes"]


def test_an_unjudged_verdict_never_counts_as_agreement() -> None:
    a = arm()
    a["sc"] = ([1] * 12, [None] * 2 + [1] * 10)
    assert not low_effort_rule({"hhr": a, "baseline": arm()}, {"hhr": SWAPS_OK, "baseline": SWAPS_OK})["passes"]


def test_flip_rate_unit_costs_and_the_integrity_stop():
    assert flip_rate([1, 0, 1, 1], [1, 1, 1, 0]) == 0.5
    costs = unit_costs({"F-medium": [{"output_tokens": 1000}, {"output_tokens": 2000}, {"output_tokens": 3000}],
                        "C-low": []})
    assert costs == {"F-medium": {"calls": 3, "mean_output_tokens": "2000.0"}}  # the mean; the rule adds the factor
    assert integrity_stop([answer("x"), answer("y", failures=["cited_text doesn't match"])]) == \
        ["answer 1: cited_text doesn't match"]


def test_rule_2_counts_failing_answers_not_failure_lines() -> None:
    long_uncited = answer(" ".join(["word"] * 301), cited=False)  # cites nothing and runs over 300 words
    out = prompt_fix_needed({"hhr-4.8k-off": FINE + [long_uncited], "hhr-4.8k-med": FINE, "base-4.8k-med": FINE})
    assert len(out["format_failures"]) == 2 and out["failing_answers"] == 1
    assert not any("format check" in r for r in out["reasons"])  # one answer is within the one allowed


def test_rule_4_reads_the_frozen_power_grid_never_a_new_simulation() -> None:
    import json
    from pathlib import Path

    from raglab.pilot_report import POWER_GRID, power_range

    rows = power_range(0.65)
    assert len(rows) == 8 and {r["runs"] for r in rows} == {1, 2}
    grid = json.loads(Path(POWER_GRID).read_text())["link1"]
    plan_row = next(r for r in grid if (r["m1"], r["e"], r["gamma"], r["R"]) == (0.65, 0.03, 0.5, 1))
    assert next(r for r in rows if (r["e"], r["gamma"], r["runs"]) == (0.03, 0.5, 1))["power"] == plan_row["power"]
    assert round(plan_row["power"], 2) == 0.72  # the row PLAN.md quotes


def test_a_format_failure_names_its_answer_by_condition_question_and_run() -> None:
    bad = answer("Use a middleware.", cited=False)
    out = prompt_fix_needed({"hhr-4.8k-off": [bad], "hhr-4.8k-med": FINE[:1], "base-4.8k-med": FINE[:1]},
                            labels={"hhr-4.8k-off": ["p04 run 1"], "hhr-4.8k-med": ["p01 run 1"],
                                    "base-4.8k-med": ["p01 run 1"]})
    assert out["format_failures"] == ["hhr-4.8k-off p04 run 1: answers without citing anything"]
