"""Stage 4's confirmatory tests on synthetic scores with known answers."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from raglab.evaluate4 import ArmScores, confirmatory, instrument_check, kappa_or_none, stated_verdict

N = 98


def questions():
    return [SimpleNamespace(id=f"q{i:03d}", target=f"page{i // 2}#t") for i in range(N)]


def arm(sc, runs=1, unjudged=None, lost=None, rate=None, relaxed=None) -> ArmScores:
    sc = np.asarray(sc, float).reshape(N, -1) if np.ndim(sc) == 2 else np.repeat(np.asarray(sc, float)[:, None], runs, 1)
    z = np.zeros_like(sc, dtype=bool)
    return ArmScores(sc=sc, correct=np.minimum(1, sc + 0.1), support_rate=sc if rate is None else rate,
                     penalized=2 * sc - 1, sc_quote_relaxed=sc if relaxed is None else relaxed,
                     unjudged_f=z if unjudged is None else unjudged, unjudged_c=z.copy(),
                     lost_citation=z.copy() if lost is None else lost)


def arms(base, hhr, bm25=None):
    out = {"base-4.8k-med": base, "hhr-4.8k-med": hhr}
    if bm25 is not None:
        out["bm25-4.8k-med"] = bm25
    return out


def test_a_clear_gain_is_supported_and_link_2_is_then_tested():
    rng = np.random.default_rng(1)
    base = (rng.random(N) < 0.3).astype(float)
    hhr = np.maximum(base, (rng.random(N) < 0.4).astype(float))
    out = confirmatory(questions(), arms(arm(base), arm(hhr), bm25=arm(hhr)))
    link1, link2 = out["chain"]
    assert link1["verdict"] == "supported" and link2["tested"] and link2["verdict"] == "inconclusive"
    assert [t["metric"] for t in out["supporting"]] == ["correct", "claim_support_rate", "penalized"]


def test_no_difference_is_inconclusive_and_nothing_downstream_is_claimed():
    base = np.tile([0.0, 1.0], N // 2)
    out = confirmatory(questions(), arms(arm(base), arm(base.copy())))
    assert out["chain"][0]["verdict"] == "inconclusive" and out["chain"][1]["verdict"].startswith("not run")
    assert all(t["verdict"].startswith("exploratory") for t in out["supporting"])


def test_a_win_that_unjudged_or_citation_losing_baseline_answers_could_erase_is_marked_sensitive():
    base, hhr = np.zeros(N), np.zeros(N)
    hhr[:14] = 1.0
    lost = np.zeros((N, 1), bool)
    lost[:14] = True  # the baseline's answers that lost a citation sit exactly where hhr won
    out = confirmatory(questions(), arms(arm(base, lost=lost), arm(hhr)))
    assert out["chain"][0]["verdict"].startswith("supported, sensitive to 14 unjudged answers")


def test_a_contradiction_that_unjudged_hhr_answers_could_erase_is_marked_sensitive_too():
    base, hhr = np.zeros(N), np.zeros(N)
    base[:14] = 1.0
    unj = np.zeros((N, 1), bool)
    unj[:14] = True  # hhr's unjudged answers sit exactly where the baseline won
    out = confirmatory(questions(), arms(arm(base), arm(hhr, unjudged=unj)))
    assert out["chain"][0]["verdict"] == "contradicted, sensitive to 14 unjudged answers"


def test_a_win_that_rests_on_quote_failures_is_marked_sensitive():
    rng = np.random.default_rng(3)
    base = (rng.random(N) < 0.3).astype(float)
    hhr = np.maximum(base, (rng.random(N) < 0.4).astype(float))
    relaxed = np.ones((N, 1))  # with failed quotes counted, both arms would score 1 everywhere: no difference
    out = confirmatory(questions(), arms(arm(base, relaxed=relaxed), arm(hhr, relaxed=relaxed)))
    assert out["chain"][0]["verdict"] == "supported, sensitive to quote failures"


def test_two_judged_runs_are_averaged_and_must_match_across_arms():
    import pytest

    rng = np.random.default_rng(4)
    two = (rng.random((N, 2)) < 0.5).astype(float)
    out = confirmatory(questions(), arms(arm(two), arm(two.copy())))
    assert out["chain"][0]["judged_runs"] == 2
    with pytest.raises(ValueError, match="same number of judged runs"):
        confirmatory(questions(), arms(arm(two), arm(two[:, :1])))


def test_s2_is_eligible_run_by_run_and_reports_who_dropped_out():
    rng = np.random.default_rng(2)
    a = (rng.random((N, 2)) < 0.3).astype(float)
    b = (rng.random((N, 2)) < 0.6).astype(float)
    rate_a, rate_b = a.copy(), b.copy()
    rate_a[:10, :] = np.nan  # the baseline made no claim in either run on 10 questions
    rate_b[10:15, 0] = np.nan  # hhr made no claim in run 1 on 5 questions: those keep run 2
    out = confirmatory(questions(), arms(arm(a, rate=rate_a), arm(b, rate=rate_b)))
    s2 = next(t for t in out["supporting"] if t["metric"] == "claim_support_rate")
    assert s2["questions"] == N - 10 and s2["dropped"]["questions"] == 10
    assert s2["dropped"]["runs with no claim in the baseline only"] == 20  # both runs of the 10


def test_kappa_is_undefined_when_a_rater_uses_one_value():
    assert kappa_or_none([0, 0, 0], [0, 1, 0]) is None and kappa_or_none([0, 1], [0, 1]) == 1.0


def test_the_instrument_check_is_balanced_across_arms_and_the_readme_wording():
    swaps = {"hhr-4.8k-med": (10, 10), "base-4.8k-med": (9, 10)}
    plants = {"plant-cross": (10, 10), "plant-shape": (9, 10)}
    sc = [1, 0, 1, 1, 0, 0] * 4
    ok = instrument_check(swaps, plants, sc, sc, (-0.05, 0.08))
    assert ok["passes"] and ok["kappa"] == 1.0
    assert stated_verdict("supported", ok) == "supported (no gross judge error found)"
    lenient_on_baseline = instrument_check({"hhr-4.8k-med": (10, 10), "base-4.8k-med": (7, 10)}, plants, sc, sc,
                                           (-0.05, 0.08))
    assert not lenient_on_baseline["passes"]  # 17 of 20 overall fails, and 7 of 10 in one arm fails too
    strict_on_fragments = instrument_check(swaps, {"plant-cross": (10, 10), "plant-shape": (6, 10)}, sc, sc,
                                           (-0.05, 0.08))
    assert strict_on_fragments["parts"]["true_support_plants"] is False
    one_value = instrument_check(swaps, plants, [0] * 24, [0] * 24, (-0.05, 0.08))
    worded = stated_verdict("supported", one_value)
    assert "not confirmed" in worded and "kappa not computable" in worded
    assert stated_verdict("inconclusive", one_value) == "inconclusive"


def _clear_win():
    rng = np.random.default_rng(1)
    base = (rng.random(N) < 0.3).astype(float)
    return base, np.maximum(base, (rng.random(N) < 0.4).astype(float))


def test_the_reference_check_can_mark_a_win_sensitive_and_not_run_is_reported():
    import dataclasses

    base, hhr = _clear_win()
    a, b = arm(base), arm(hhr)
    out = confirmatory(questions(), arms(a, b), reference_check="not run")
    assert out["chain"][0]["verdict"] == "supported" and out["chain"][0]["reference_check"]["state"] == "not run"
    same = confirmatory(questions(), arms(dataclasses.replace(a, sc_rekeyed=a.sc),
                                          dataclasses.replace(b, sc_rekeyed=b.sc)), reference_check="run")
    assert same["chain"][0]["verdict"] == "supported"
    gone = confirmatory(questions(), arms(dataclasses.replace(a, sc_rekeyed=b.sc),
                                          dataclasses.replace(b, sc_rekeyed=b.sc)), reference_check="run")
    assert gone["chain"][0]["verdict"] == "supported, sensitive to the reference top-up"


def test_citation_losing_answers_enter_every_supporting_worst_case():
    base, hhr = _clear_win()
    lost = np.zeros((N, 1), bool)
    lost[(hhr == 1) & (base == 0)] = True  # every hhr win lost a citation
    clean = confirmatory(questions(), arms(arm(base), arm(hhr)))
    hit = confirmatory(questions(), arms(arm(base), arm(hhr, lost=lost)))
    for c, h in zip(clean["supporting"], hit["supporting"]):
        assert h["worst_p"] >= c["worst_p"]
    assert any(h["worst_p"] > c["worst_p"] for c, h in zip(clean["supporting"], hit["supporting"]))


def test_s2_dropouts_add_up_even_when_the_arms_lose_different_runs():
    base_rate = np.array([[0.8, 0.9], [np.nan, 0.5]] + [[0.5, 0.5]] * (N - 2))
    hhr_rate = np.array([[1.0, 1.0], [0.7, np.nan]] + [[0.6, 0.6]] * (N - 2))
    sc = np.zeros((N, 2))
    out = confirmatory(questions(), arms(arm(sc, rate=base_rate), arm(sc, rate=hhr_rate)))
    s2 = next(t for t in out["supporting"] if t["metric"] == "claim_support_rate")
    assert s2["questions"] == N - 1
    assert s2["dropped"] == {"questions": 1, "runs with no claim in the baseline only": 1,
                             "runs with no claim in hhr only": 1, "runs with no claim in either": 0}


def test_a_planted_gate_with_a_shortfall_keeps_the_same_shares_and_needs_6_per_group():
    from raglab.evaluate4 import gate_passes

    assert gate_passes({"hhr": (9, 10), "base": (9, 10)}, 20, 18, 8)
    assert not gate_passes({"hhr": (10, 10), "base": (7, 10)}, 20, 18, 8)
    assert gate_passes({"hhr": (10, 10), "base": (7, 8)}, 20, 18, 8)  # only 18 qualified: 17 of 18, 7 of 8
    assert not gate_passes({"hhr": (9, 10), "base": (7, 8)}, 20, 18, 8)  # 16 of 18 is under 90%
    assert not gate_passes({"hhr": (10, 10), "base": (6, 8)}, 20, 18, 8)  # 6 of 8 is under 80%
    assert not gate_passes({"hhr": (5, 5), "base": (10, 10)}, 20, 18, 8)  # fewer than 6 judged: not computable


def test_the_golden_citation_stop_and_the_leniency_flag():
    from raglab.evaluate4 import citation_failure_stop, lenient

    assert citation_failure_stop({"hhr": (1, 1000), "base": (6, 1000)}) == ["base"]
    assert lenient([0] * 40)["lenient"] is False
    assert lenient([1] * 6 + [0] * 34)["lenient"] is True  # 15%, lower bound about 7%
    assert lenient([])["n"] == 0


def test_a_gate_never_passes_on_one_arm_alone_and_only_plain_supported_gets_the_check_wording():
    from raglab.evaluate4 import gate_passes

    assert not gate_passes({"hhr-4.8k-med": (9, 10)}, 20, 18, 8, ("hhr-4.8k-med", "base-4.8k-med"))
    sc = [1, 0, 1, 1, 0, 0] * 4
    ok = instrument_check({"hhr-4.8k-med": (10, 10), "base-4.8k-med": (9, 10)},
                          {"plant-cross": (10, 10), "plant-shape": (9, 10)}, sc, sc, (-0.05, 0.08))
    missing_arm = instrument_check({"hhr-4.8k-med": (10, 10)}, {"plant-cross": (10, 10), "plant-shape": (9, 10)},
                                   sc, sc, (-0.05, 0.08))
    assert ok["passes"] and not missing_arm["parts"]["near_miss_swaps"]
    qualified = "supported, sensitive to 3 unjudged answers"
    assert stated_verdict(qualified, ok) == qualified  # the table's Supported rows are for plain "supported" only


def test_link_1_names_the_primary_metric():
    base, hhr = _clear_win()
    out = confirmatory(questions(), arms(arm(base), arm(hhr)))
    assert out["chain"][0]["metric"] == "supported_correct"
