"""Uncertainty for a small evaluation set.

With a hundred questions, a five-point difference between two configurations
can be noise, so every number Stage 2 reports carries an interval:

- each configuration's mean gets a 95% interval: a percentile bootstrap over
  questions for scores, a Wilson score interval for yes/no rates (a bootstrap
  of 33 successes out of 33 could only say [1, 1]);
- each comparison is *paired*: both configurations answer the same resampled
  questions, so the interval is on the per-question difference, which is far
  tighter than comparing two independent intervals;
- each comparison against the baseline also gets a p-value from a paired
  sign-flip randomisation test, Holm-adjusted across the configurations
  compared, so "eleven tries, one lucky" doesn't read as a finding.

Everything is seeded: the same inputs give the same intervals, bit for bit.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

RESAMPLES = 10_000
SEED = 20261120  # the day Stage 2 began; any fixed value would do
CONFIDENCE = 0.95


def bootstrap_indices(n: int, resamples: int = RESAMPLES, seed: int = SEED) -> np.ndarray:
    """One shared (resamples x n) table of resampled question indices, so comparisons are paired."""
    if n < 1:
        raise ValueError("need at least one question to resample")
    return np.random.default_rng(seed).integers(0, n, size=(resamples, n))


def _percentiles(samples: np.ndarray, confidence: float) -> tuple[float, float]:
    tail = (1 - confidence) / 2 * 100
    lo, hi = np.percentile(samples, [tail, 100 - tail])
    return float(lo), float(hi)


def mean_ci(values: Sequence[float], indices: np.ndarray, confidence: float = CONFIDENCE) -> tuple[float, float, float]:
    """(mean, lo, hi) for one configuration's per-question scores."""
    x = np.asarray(values, dtype=float)
    _check(x, indices)
    return (float(x.mean()), *_percentiles(x[indices].mean(axis=1), confidence))


def wilson_ci(values: Sequence[float], confidence: float = CONFIDENCE) -> tuple[float, float, float]:
    """(mean, lo, hi) for a yes/no score: the Wilson score interval.

    A percentile bootstrap of 33 successes out of 33 can only return [1, 1];
    the Wilson interval stays honest at the edges (here about 0.90-1.00).
    """
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or x.size == 0 or not np.isin(x, (0.0, 1.0)).all():
        raise ValueError("need a non-empty 1-D array of 0/1 scores")
    n, p = x.size, float(x.mean())
    z = float(_normal_quantile(0.5 + confidence / 2))
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    spread = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    lo = 0.0 if p == 0.0 else max(0.0, float(centre - spread))  # the edges are exact, not float noise
    hi = 1.0 if p == 1.0 else min(1.0, float(centre + spread))
    return p, lo, hi


def _normal_quantile(q: float) -> float:
    from statistics import NormalDist

    return NormalDist().inv_cdf(q)


def paired_diff_ci(
    a: Sequence[float], b: Sequence[float], indices: np.ndarray, confidence: float = CONFIDENCE
) -> tuple[float, float, float]:
    """(mean of b - a, lo, hi): how much better b is than a on the same questions."""
    d = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    _check(d, indices)
    return (float(d.mean()), *_percentiles(d[indices].mean(axis=1), confidence))


def sign_flip_pvalue(diffs: Sequence[float], resamples: int = RESAMPLES, seed: int = SEED) -> float:
    """Two-sided paired randomisation test of "no difference".

    Under the null hypothesis each question's difference is as likely to be
    positive as negative, so its sign is flipped at random. The p-value is the
    share of flips whose mean is at least as extreme as the observed one,
    counting the observed arrangement itself, so it is never exactly zero.
    """
    d = np.asarray(diffs, dtype=float)
    if d.ndim != 1 or d.size == 0:
        raise ValueError("need a non-empty 1-D array of paired differences")
    observed = abs(d.mean())
    if observed == 0.0:
        return 1.0
    signs = np.random.default_rng(seed).choice((-1.0, 1.0), size=(resamples, d.size))
    flipped = np.abs(signs @ d) / d.size
    extreme = int(np.count_nonzero(flipped >= observed * (1 - 1e-12)))
    return (extreme + 1) / (resamples + 1)


def two_group_diff_ci(
    a: Sequence[float], b: Sequence[float], resamples: int = RESAMPLES, seed: int = SEED,
    confidence: float = CONFIDENCE,
) -> tuple[float, float, float]:
    """(mean of a - mean of b, lo, hi) for two groups of *different* questions.

    The groups can't be paired, so each is resampled on its own (a bootstrap
    stratified by group). Used to compare question styles.
    """
    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or x.size == 0 or y.size == 0:
        raise ValueError("need two non-empty 1-D arrays")
    rng = np.random.default_rng(seed)
    boots = x[rng.integers(0, x.size, size=(resamples, x.size))].mean(axis=1) \
        - y[rng.integers(0, y.size, size=(resamples, y.size))].mean(axis=1)
    return (float(x.mean() - y.mean()), *_percentiles(boots, confidence))


def permutation_pvalue(a: Sequence[float], b: Sequence[float], resamples: int = RESAMPLES, seed: int = SEED) -> float:
    """Two-sided test that two groups of different questions share a mean.

    Under the null hypothesis the group labels are exchangeable, so they are
    shuffled at random. The p-value is the share of shuffles whose difference
    in means is at least as extreme as the observed one, counting the observed
    arrangement itself.
    """
    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or x.size == 0 or y.size == 0:
        raise ValueError("need two non-empty 1-D arrays")
    observed = abs(x.mean() - y.mean())
    if observed == 0.0:
        return 1.0
    pooled = np.concatenate([x, y])
    shuffled = np.random.default_rng(seed).permuted(np.tile(pooled, (resamples, 1)), axis=1)
    diffs = np.abs(shuffled[:, : x.size].mean(axis=1) - shuffled[:, x.size:].mean(axis=1))
    extreme = int(np.count_nonzero(diffs >= observed * (1 - 1e-12)))
    return (extreme + 1) / (resamples + 1)


def _clusters(diffs: Sequence[float], clusters: Sequence[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = np.asarray(diffs, dtype=float)
    if d.ndim != 1 or d.size == 0 or len(clusters) != d.size:
        raise ValueError("need one cluster label per paired difference")
    _, inverse = np.unique(np.asarray(clusters), return_inverse=True)
    return d, np.bincount(inverse, weights=d), np.bincount(inverse).astype(float)


def cluster_sign_flip_pvalue(
    diffs: Sequence[float], clusters: Sequence[str], resamples: int = RESAMPLES, seed: int = SEED,
) -> float:
    """The sign-flip test with one sign per cluster, so questions that share a target page flip together.

    Questions written from the same page share its answers, so they aren't
    independent; flipping them together gives an honest p-value when they agree.
    """
    d, sums, _ = _clusters(diffs, clusters)
    observed = abs(d.mean())
    if observed == 0.0:
        return 1.0
    signs = np.random.default_rng(seed).choice((-1.0, 1.0), size=(resamples, sums.size))
    flipped = np.abs(signs @ sums) / d.size
    extreme = int(np.count_nonzero(flipped >= observed * (1 - 1e-12)))
    return (extreme + 1) / (resamples + 1)


def cluster_bootstrap_ci(
    diffs: Sequence[float], clusters: Sequence[str], resamples: int = RESAMPLES, seed: int = SEED,
    confidence: float = CONFIDENCE,
) -> tuple[float, float, float]:
    """(mean, lo, hi) with a percentile bootstrap that resamples whole clusters (target pages)."""
    d, sums, counts = _clusters(diffs, clusters)
    picks = np.random.default_rng(seed).integers(0, sums.size, size=(resamples, sums.size))
    boots = sums[picks].sum(axis=1) / counts[picks].sum(axis=1)
    return (float(d.mean()), *_percentiles(boots, confidence))


def holm(pvalues: Sequence[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, in the input order."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


def _check(x: np.ndarray, indices: np.ndarray) -> None:
    if x.ndim != 1 or x.size == 0:
        raise ValueError("need a non-empty 1-D array of per-question scores")
    if indices.ndim != 2 or indices.shape[1] != x.size:
        raise ValueError(f"bootstrap table is for {indices.shape[-1]} questions, scores are for {x.size}")


def cohen_kappa(
    a: Sequence[int],
    b: Sequence[int],
    categories: Sequence[int] = (0, 1, 2),
    weighted: bool = False,
    sample_weight: Sequence[float] | None = None,
) -> float:
    """Agreement between two judges beyond what chance would give: 1 = perfect, 0 = chance.

    `weighted=True` gives linear weights, for ordered grades: calling a 2 a 1
    is a smaller disagreement than calling it a 0. `sample_weight` lets a
    stratified sample stand for the population it was drawn from.
    """
    if len(a) != len(b) or not a:
        raise ValueError("need two equal-length, non-empty lists of grades")
    w = [1.0] * len(a) if sample_weight is None else list(sample_weight)
    if len(w) != len(a):
        raise ValueError("need one sample weight per pair of grades")
    k = len(categories)
    pos = {c: i for i, c in enumerate(categories)}
    observed = np.zeros((k, k))
    for x, y, weight in zip(a, b, w):
        observed[pos[x], pos[y]] += weight
    observed /= observed.sum()
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0))
    i, j = np.indices((k, k))
    disagreement = np.abs(i - j) / (k - 1) if weighted else (i != j).astype(float)
    denominator = (disagreement * expected).sum()
    if denominator == 0:
        return 1.0
    return float(1 - (disagreement * observed).sum() / denominator)
