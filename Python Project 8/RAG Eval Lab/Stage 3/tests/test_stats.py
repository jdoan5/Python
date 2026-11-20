from __future__ import annotations

import numpy as np
import pytest

from raglab.stats import bootstrap_indices, holm, mean_ci, paired_diff_ci, sign_flip_pvalue


def test_bootstrap_is_reproducible_and_seed_dependent():
    assert np.array_equal(bootstrap_indices(30, 500, seed=1), bootstrap_indices(30, 500, seed=1))
    assert not np.array_equal(bootstrap_indices(30, 500, seed=1), bootstrap_indices(30, 500, seed=2))


def test_interval_brackets_the_mean_and_tightens_with_more_questions():
    rng = np.random.default_rng(0)
    small, large = rng.random(20), rng.random(400)
    for x in (small, large):
        mean, lo, hi = mean_ci(x, bootstrap_indices(len(x), 4000))
        assert lo < mean < hi

    def width(x):
        _, lo, hi = mean_ci(x, bootstrap_indices(len(x), 4000))
        return hi - lo

    assert width(large) < width(small) / 2


def test_constant_scores_have_a_zero_width_interval():
    assert mean_ci([1.0] * 10, bootstrap_indices(10, 1000)) == (1.0, 1.0, 1.0)


def test_paired_difference_sees_through_question_difficulty():
    # Question difficulty varies wildly, but b beats a by exactly 0.1 on every one.
    rng = np.random.default_rng(1)
    a = rng.random(50) * 0.8
    b = a + 0.1
    delta, lo, hi = paired_diff_ci(a, b, bootstrap_indices(50, 2000))
    assert delta == pytest.approx(0.1) and lo == pytest.approx(0.1) and hi == pytest.approx(0.1)


def test_mismatched_bootstrap_table_is_rejected():
    with pytest.raises(ValueError, match="bootstrap table"):
        mean_ci([0.5] * 5, bootstrap_indices(6, 100))


def test_no_difference_gives_p_of_one():
    assert sign_flip_pvalue([0.0] * 20) == 1.0


def test_a_consistent_difference_is_significant_and_p_is_never_zero():
    p = sign_flip_pvalue([1.0] * 30, resamples=5000)
    assert p == pytest.approx(1 / 5001)


def test_the_test_is_calibrated_under_the_null():
    # Symmetric noise: about 5% of datasets should come out "significant" at 0.05.
    rng = np.random.default_rng(7)
    rejections = sum(sign_flip_pvalue(rng.normal(size=40), resamples=2000, seed=i) < 0.05 for i in range(400))
    assert 0.02 <= rejections / 400 <= 0.09


def test_the_test_has_power_against_a_real_effect():
    rng = np.random.default_rng(8)
    rejections = sum(sign_flip_pvalue(rng.normal(0.6, 1.0, size=40), resamples=2000, seed=i) < 0.05 for i in range(100))
    assert rejections >= 85


def test_holm_matches_a_worked_example():
    # sorted: .005*4=.02, .01*3=.03, .03*2=.06, .04*1=.04 -> monotone: .02 .03 .06 .06
    assert holm([0.01, 0.04, 0.03, 0.005]) == pytest.approx([0.03, 0.06, 0.06, 0.02])


def test_holm_never_lowers_a_p_value_or_exceeds_one():
    ps = [0.2, 0.9, 0.04, 0.5]
    for raw, adjusted in zip(ps, holm(ps)):
        assert raw <= adjusted <= 1.0


def test_kappa_is_one_for_perfect_agreement_and_near_zero_for_chance():
    from raglab.stats import cohen_kappa

    grades = [0, 1, 2, 2, 0, 1, 0]
    assert cohen_kappa(grades, grades) == pytest.approx(1.0)
    rng = np.random.default_rng(3)
    a, b = rng.integers(0, 3, 3000).tolist(), rng.integers(0, 3, 3000).tolist()
    assert abs(cohen_kappa(a, b)) < 0.05


def test_kappa_matches_a_worked_example():
    from raglab.stats import cohen_kappa

    # 2x2 textbook case: 20 yes/yes, 5 yes/no, 10 no/yes, 15 no/no -> po = .7, pe = .5, kappa = .4
    a = [1] * 25 + [0] * 25
    b = [1] * 20 + [0] * 5 + [1] * 10 + [0] * 15
    assert cohen_kappa(a, b, categories=(0, 1)) == pytest.approx(0.4)


def test_weighted_kappa_forgives_near_misses_more_than_far_ones():
    from raglab.stats import cohen_kappa

    truth = [0, 1, 2] * 30
    near = [min(g + 1, 2) if i % 3 == 0 else g for i, g in enumerate(truth)]     # some 0->1 slips
    far = [2 if g == 0 and i % 3 == 0 else g for i, g in enumerate(truth)]       # some 0->2 jumps
    assert cohen_kappa(truth, near, weighted=True) > cohen_kappa(truth, far, weighted=True)


def test_the_interval_is_a_95_percent_percentile_interval():
    x = np.random.default_rng(5).random(60)
    idx = bootstrap_indices(60, 4000, seed=9)
    _, lo, hi = mean_ci(x, idx)
    boot = x[idx].mean(axis=1)
    assert (lo, hi) == pytest.approx(tuple(np.percentile(boot, [2.5, 97.5])))


def test_weighted_kappa_is_linear_like_scikit_learn():
    from sklearn.metrics import cohen_kappa_score

    from raglab.stats import cohen_kappa

    rng = np.random.default_rng(11)
    a = rng.integers(0, 3, 400)
    b = np.where(rng.random(400) < 0.7, a, rng.integers(0, 3, 400))
    assert cohen_kappa(a.tolist(), b.tolist(), weighted=True) == pytest.approx(cohen_kappa_score(a, b, weights="linear"))
    assert cohen_kappa(a.tolist(), b.tolist()) == pytest.approx(cohen_kappa_score(a, b))


def test_wilson_interval_stays_honest_at_the_edges():
    from raglab.stats import wilson_ci

    mean, lo, hi = wilson_ci([1.0] * 33)
    assert mean == 1.0 and hi == 1.0 and lo == pytest.approx(0.896, abs=0.002)  # not the bootstrap's [1, 1]
    mean, lo, hi = wilson_ci([1.0] * 20 + [0.0] * 20)
    assert mean == 0.5 and lo < 0.5 < hi
    with pytest.raises(ValueError, match="0/1"):
        wilson_ci([0.5, 1.0])


# --- Stage 3: groups of different questions, and target-page clusters ---------------------------


def test_two_groups_of_different_questions_get_a_stratified_interval_and_a_permutation_test():
    from raglab.stats import permutation_pvalue, two_group_diff_ci

    a, b = np.array([1.0, 0.8, 0.9, 1.0, 0.7] * 6), np.array([0.1, 0.0, 0.2, 0.1, 0.0] * 6)
    diff, lo, hi = two_group_diff_ci(a, b, resamples=2000, seed=1)
    assert diff == pytest.approx(a.mean() - b.mean()) and lo <= diff <= hi and lo > 0
    assert permutation_pvalue(a, b, resamples=2000, seed=1) == pytest.approx(1 / 2001)  # never exactly 0
    assert permutation_pvalue(a, a.copy(), resamples=2000, seed=1) == 1.0
    noise = np.random.default_rng(3)
    assert permutation_pvalue(noise.random(30), noise.random(30), resamples=2000, seed=1) > 0.05
    with pytest.raises(ValueError):
        two_group_diff_ci([], [1.0])


def test_page_clusters_flip_and_resample_together():
    from raglab.stats import cluster_bootstrap_ci, cluster_sign_flip_pvalue, sign_flip_pvalue

    # 20 pages, two questions each; the two questions on a page always move together.
    pages = [f"p{i // 2}" for i in range(40)]
    d = np.repeat(np.random.default_rng(5).normal(0.15, 0.3, 20), 2)
    per_question = sign_flip_pvalue(d, 5000, 7)
    per_page = cluster_sign_flip_pvalue(d, pages, 5000, 7)
    assert per_page > per_question  # pairs that agree look twice as convincing when counted twice
    mean, lo, hi = cluster_bootstrap_ci(d, pages, 5000, 7)
    assert mean == pytest.approx(d.mean()) and lo <= mean <= hi
    # with every question on its own page, the cluster test is the plain test
    singletons = [f"p{i}" for i in range(40)]
    assert cluster_sign_flip_pvalue(d, singletons, 5000, 7) == pytest.approx(per_question, abs=0.02)
    assert cluster_sign_flip_pvalue(np.zeros(4), ["a", "a", "b", "b"]) == 1.0
    with pytest.raises(ValueError, match="one cluster label"):
        cluster_sign_flip_pvalue(d, pages[:-1])
