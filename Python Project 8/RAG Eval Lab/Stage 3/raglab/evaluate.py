"""Score every configuration against the golden set, with uncertainty, and record how to reproduce it.

A run:

1. builds each configuration's retriever over the verified corpus;
2. ranks every chunk with a positive score for every question, so even the
   largest budget is filled whenever the retriever has enough to fill it, and
   scores the ranking (metrics.py);
3. summarises them. That means a bootstrap interval on every mean, paired
   differences against the baseline with Holm-adjusted p-values, the average
   effect of each switch in the factorial, and the analysis plan's
   confirmatory family: one primary test, then a Holm-adjusted secondary
   family that counts only if the primary test passes;
4. writes the run to a directory:
   - manifest.json: what produced the numbers (code, corpus, golden set,
     every parameter, library versions);
   - summary.json: the numbers;
   - questions.jsonl: every question's scores and top-10 section ranking
     under every configuration, one line each, so two runs can be diffed.

Paths in the manifest are relative to the project, so a committed run never
carries the machine it ran on.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from dataclasses import dataclass, field
from statistics import NormalDist
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import sklearn
import yaml

from raglab import __version__
from raglab.corpus import CorpusLock, Document, project_root
from raglab.experiments import BASELINE, CHAIN, CHALLENGER, EXPLORATORY_PAIRS, INCUMBENT, RetrievalConfig, build_retrievers
from raglab.golden import STYLES, GoldenSet, Question
from raglab.metrics import (
    BINARY_METRICS, BUDGETS, GAIN, HEADLINE_BUDGET, HEADLINE_METRICS, K_VALUES, METRICS, RANK_DEPTH, Span, pack, score,
)
from raglab.retrieval import Retriever
from raglab.sections import MIN_OVERLAP_CHARS, SectionIndex
from raglab.stats import (
    CONFIDENCE, RESAMPLES, SEED, bootstrap_indices, cluster_bootstrap_ci, cluster_sign_flip_pvalue, holm, mean_ci,
    paired_diff_ci, permutation_pvalue, sign_flip_pvalue, two_group_diff_ci, wilson_ci,
)

SCHEMA = 4  # 4: neural retrievers, the confirmatory family, models, numerics and the plan in the manifest
# Switch effects cover every metric the README's tables show.
EFFECT_METRICS = (
    "success@5", "recall@5", "mrr@10", "ndcg@10", "ndcg_answers@10",
    *(f"{m}@{HEADLINE_BUDGET}" for m in ("budget_success", "budget_recall", "coverage", "best_coverage")),
)

# The factorial's switches: position in RetrievalConfig.cell, and the comparisons reported.
# Retriever switches are matched over the six chunker x breadcrumb cells. Chunker and breadcrumb
# effects are computed within each retriever family, never pooled across families that work differently.
FACTORS: tuple[tuple[str, int, tuple[tuple[object, object], ...]], ...] = (
    ("chunker", 0, (("paragraph", "heading"), ("paragraph", "fixed"), ("fixed", "heading"))),
    ("retriever", 1, (("tfidf", "bm25"), ("bm25", "dense"), ("bm25", "hybrid"), ("dense", "hybrid"),
                      ("hybrid", "hybrid+rerank"),
                      # the control exists only at heading chunks with the breadcrumb: one matched pair each
                      ("bm25", "bm25+rerank"), ("bm25+rerank", "hybrid+rerank"))),
    ("breadcrumb", 2, ((False, True),)),
)
RETRIEVER_FAMILIES: dict[str, tuple[str, ...]] = {
    "lexical": ("bm25", "tfidf"), "dense": ("dense",), "hybrid": ("hybrid",),
    "bm25+rerank": ("bm25+rerank",), "hybrid+rerank": ("hybrid+rerank",),
}

# The analysis plan's confirmatory tests (PLAN.md). The CHAIN is tested in order on PRIMARY_METRIC, each
# link at ALPHA and only if every earlier link was significant; its first link is the headline. The
# headline pair's SUPPORTING_METRICS are Holm-adjusted together and claimed only if the headline holds.
PRIMARY_METRIC = "mrr@10"
ALPHA = 0.05
SUPPORTING_METRICS = ("success@5", "ndcg_answers@10", "ndcg@10", "recall@5", f"coverage@{HEADLINE_BUDGET}",
                      f"budget_recall@{HEADLINE_BUDGET}")
# The style interaction (descriptive): dense - BM25 on paraphrase questions minus the same on keyword questions.
INTERACTION_PAIR = ("heading+bm25", "heading+dense")
INTERACTION_STYLES = ("paraphrase", "keyword")
INTERACTION_METRICS = ("ndcg_answers@10", "recall@5")  # the first is the pre-specified one
_Z80 = NormalDist().inv_cdf(1 - ALPHA / 2) + NormalDist().inv_cdf(0.80)  # two-sided alpha, 80% power


@dataclass(frozen=True)
class QuestionResult:
    values: Mapping[str, float]
    ranking: tuple[str | None, ...]  # top sections; None = a chunk that retrieved no section
    top5_chars: int                  # how much text the first five chunks are
    in_budget: frozenset[str] = frozenset()  # sections retrieved inside the headline budget
    in_budgets: Mapping[int, frozenset[str]] = field(default_factory=dict)  # ...and inside every budget
    rerank_depth: int | None = None  # how many first-stage chunks a reranker re-scored
    window_chunks: Mapping[int, int] = field(default_factory=dict)  # chunks packed into each budget
    tie_in_top5: bool = False        # an exact score tie set the order of two of the first five chunks
    near_tie_in_top10: bool = False  # two adjacent scores among the first ten chunks differ by under 1e-6


RunResults = dict[str, list[QuestionResult]]  # config name -> one result per question, in golden order


def ranked_spans(retriever: Retriever, query: str) -> list[Span]:
    """Every chunk with a positive score, best first, as spans.

    No fixed depth: a cut-off of a few hundred chunks can hold less text than
    the largest budget when the chunks are tiny paragraphs, which would score
    those configurations on a context they never got to fill.
    """
    return [(h.chunk.doc_id, h.chunk.start, h.chunk.end) for h in retriever.search(query, len(retriever.chunks))]


def run_configs(
    docs: list[Document],
    questions: Sequence[Question],
    configs: Sequence[RetrievalConfig],
    index: SectionIndex,
) -> RunResults:
    names = [c.name for c in configs]
    if len(set(names)) != len(names):
        raise ValueError("configuration names must be unique")
    retrievers = build_retrievers(docs, tuple(configs))
    results: RunResults = {}
    for config in configs:
        retriever = retrievers[config.name]
        per_question = []
        for q in questions:
            hits = retriever.search(q.text, len(retriever.chunks))
            spans = [(h.chunk.doc_id, h.chunk.start, h.chunk.end) for h in hits]
            values, ranking = score(spans, q.primary, q.judgments, index)
            windows = {
                budget: frozenset(ref.location for doc_id, start, end in pack(spans, budget)
                                  for ref, _ in index.retrieved_by(doc_id, start, end))
                for budget in BUDGETS
            }
            depth = None
            if hasattr(retriever, "depth_for"):
                depth = retriever.depth_for(retriever.first_stage.search(q.text, len(retriever.chunks)))
            top = [h.score for h in hits[:RANK_DEPTH]]
            per_question.append(QuestionResult(
                values, tuple(ranking), sum(end - start for _, start, end in spans[:5]), windows[HEADLINE_BUDGET],
                windows, depth, {budget: len(pack(spans, budget)) for budget in BUDGETS},
                tie_in_top5=any(top[i] == top[i + 1] for i in range(min(5, len(top)) - 1)),
                near_tie_in_top10=any(0 < abs(top[i] - top[i + 1]) < 1e-6 for i in range(len(top) - 1)),
            ))
        results[config.name] = per_question
    return results


# --- pooling ---------------------------------------------------------------------------------


def unjudged(
    questions: Sequence[Question], results: RunResults, depth: int
) -> dict[str, dict[str, list[str]]]:
    """For each question: sections in some configuration's top `depth` that nobody has judged yet.

    Maps question id -> location -> the configurations that surfaced it.
    """
    pool: dict[str, dict[str, list[str]]] = {}
    for i, q in enumerate(questions):
        for name, per_question in results.items():
            for location in per_question[i].ranking[:depth]:
                if location is not None and location not in q.judgments:
                    pool.setdefault(q.id, {}).setdefault(location, []).append(name)
    return pool


def unjudged_in_windows(
    questions: Sequence[Question], results: RunResults, budget: int = HEADLINE_BUDGET,
    configs: Sequence[str] | None = None,
) -> dict[str, dict[str, list[str]]]:
    """For each question: sections inside some configuration's `budget` window that nobody has judged yet."""
    pool: dict[str, dict[str, list[str]]] = {}
    for i, q in enumerate(questions):
        for name, per_question in results.items():
            if configs is not None and name not in configs:
                continue
            for location in sorted(per_question[i].in_budgets.get(budget, per_question[i].in_budget)):
                if location not in q.judgments:
                    pool.setdefault(q.id, {}).setdefault(location, []).append(name)
    return pool


# --- summarising -------------------------------------------------------------------------------


def _matrix(results: RunResults) -> dict[str, dict[str, np.ndarray]]:
    return {
        name: {m: np.array([r.values[m] for r in per_question], dtype=float) for m in METRICS}
        for name, per_question in results.items()
    }


def summarize(
    questions: Sequence[Question],
    configs: Sequence[RetrievalConfig],
    results: RunResults,
    baseline: RetrievalConfig = BASELINE,
    resamples: int = RESAMPLES,
    seed: int = SEED,
) -> dict:
    names = [c.name for c in configs]
    if baseline.name not in results:
        raise ValueError(f"the baseline {baseline.name!r} must be one of the configurations")
    idx = bootstrap_indices(len(questions), resamples, seed)
    matrix = _matrix(results)

    def interval(values: np.ndarray, metric: str, indices: np.ndarray) -> dict:
        # Yes/no scores get Wilson intervals: a bootstrap of 33 out of 33 would claim [1, 1].
        ci = wilson_ci(values) if metric in BINARY_METRICS else mean_ci(values, indices)
        return dict(zip(("mean", "lo", "hi"), ci))

    summary = {name: {m: interval(matrix[name][m], m, idx) for m in METRICS} for name in names}

    vs_baseline: dict[str, dict] = {name: {} for name in names if name != baseline.name}
    for m in METRICS:
        rows = []
        for name in vs_baseline:
            delta, lo, hi = paired_diff_ci(matrix[baseline.name][m], matrix[name][m], idx)
            p = sign_flip_pvalue(matrix[name][m] - matrix[baseline.name][m], resamples, seed + 1)
            rows.append((name, {"delta": delta, "lo": lo, "hi": hi, "p": p}))
        for (name, row), p_holm in zip(rows, holm([row["p"] for _, row in rows])):
            vs_baseline[name][m] = {**row, "p_holm": p_holm}

    by_style = {}
    style_metrics = EFFECT_METRICS
    for style in STYLES:
        mask = np.array([q.style == style for q in questions])
        if mask.any():
            style_idx = bootstrap_indices(int(mask.sum()), resamples, seed)
            by_style[style] = {
                "n": int(mask.sum()),
                "configs": {
                    name: {m: interval(matrix[name][m][mask], m, style_idx) for m in style_metrics}
                    for name in names
                },
            }

    diagnostics = {}
    for name in names:
        judged, empty = [], []
        judged_budget: dict[int, list[float]] = {b: [] for b in BUDGETS}
        for q, r in zip(questions, results[name]):
            found = [loc for loc in r.ranking if loc is not None]
            judged.append(sum(loc in q.judgments for loc in found) / len(found) if found else 1.0)
            for b in BUDGETS:
                window = r.in_budgets.get(b, r.in_budget if b == HEADLINE_BUDGET else frozenset())
                judged_budget[b].append(sum(loc in q.judgments for loc in window) / len(window) if window else 1.0)
            empty.append(sum(loc is None for loc in r.ranking) / RANK_DEPTH)
        depths = [r.rerank_depth for r in results[name] if r.rerank_depth is not None]
        related = [sum(q.judgments.get(loc) == 1 for loc in r.ranking if loc is not None) / RANK_DEPTH
                   for q, r in zip(questions, results[name])]
        unreranked = {
            str(b): float(np.mean([max(0, r.window_chunks[b] - r.rerank_depth) / r.window_chunks[b]
                                   if r.window_chunks.get(b) else 0.0 for r in results[name]]))
            for b in BUDGETS
        } if depths else None
        diagnostics[name] = {
            "judged@10": float(np.mean(judged)),
            f"judged_in_budget@{HEADLINE_BUDGET}": float(np.mean(judged_budget[HEADLINE_BUDGET])),
            f"sections_in_budget@{HEADLINE_BUDGET}": float(np.mean([len(r.in_budget) for r in results[name]])),
            "empty_slots@10": float(np.mean(empty)),
            "top5_chars": float(np.mean([r.top5_chars for r in results[name]])),
            "judged_in_budget": {str(b): float(np.mean(v)) for b, v in judged_budget.items()},
            "rerank_depth": {"mean": float(np.mean(depths)), "max": int(max(depths))} if depths else None,
            "unreranked_share_of_window": unreranked,  # chunks in each budget window that lie below the rerank depth
            "related_share@10": float(np.mean(related)),  # top-10 slots holding a section judged merely related
            "tie_in_top5": float(np.mean([r.tie_in_top5 for r in results[name]])),
            "near_tie_in_top10": float(np.mean([r.near_tie_in_top10 for r in results[name]])),
        }

    return {
        "questions": len(questions),
        "baseline": baseline.name,
        "configs": names,
        "metrics": list(METRICS),
        "headline": list(HEADLINE_METRICS),
        "summary": summary,
        "vs_baseline": vs_baseline,
        "effects": factor_effects(configs, matrix, idx, resamples, seed),
        "by_style": by_style,
        "diagnostics": diagnostics,
        "confirmatory": confirmatory(questions, matrix, idx, resamples, seed),
        "by_style_paired": by_style_paired(questions, matrix, resamples, seed),
        "interaction": interaction(questions, matrix, resamples, seed),
    }


def _verdict(delta: float, hi: float, p: float, predicted: int) -> str:
    """How a test reads against its prediction: supported, contradicted, or inconclusive."""
    if p < ALPHA:
        if predicted == 0:
            return "difference: higher" if delta > 0 else "difference: lower"
        return "supported" if delta * predicted > 0 else "contradicted"
    return "contradicted" if predicted > 0 and hi < 0 else "inconclusive"


def confirmatory(
    questions: Sequence[Question], matrix: Mapping[str, Mapping[str, np.ndarray]], idx: np.ndarray,
    resamples: int = RESAMPLES, seed: int = SEED,
) -> dict | None:
    """The analysis plan's confirmatory tests, or None when their configurations weren't run.

    The CHAIN is a fixed sequence on PRIMARY_METRIC: each link is tested at
    ALPHA (two-sided paired sign-flip) only if every earlier link was
    significant, which keeps the chance of any false claim at ALPHA. The
    headline pair's SUPPORTING_METRICS are Holm-adjusted together and claimed
    only if the headline (the first link) is significant. Every test is also
    run at the level of target pages (questions from the same page flip and
    resample together); a result that holds only per question is fragile.
    """
    needed = {c.name for a, b, _ in CHAIN for c in (a, b)}
    if any(name not in matrix for name in needed):
        return None
    n = len(questions)
    pages = [q.target.split("#", 1)[0] for q in questions]

    def test(a: str, b: str, metric: str, predicted: int) -> dict:
        d = matrix[b][metric] - matrix[a][metric]
        delta, lo, hi = paired_diff_ci(matrix[a][metric], matrix[b][metric], idx)
        _, page_lo, page_hi = cluster_bootstrap_ci(d, pages, resamples, seed)
        p = sign_flip_pvalue(d, resamples, seed + 1)
        page_p = cluster_sign_flip_pvalue(d, pages, resamples, seed + 1)
        return {"from": a, "to": b, "metric": metric, "predicted": predicted, "delta": delta, "lo": lo, "hi": hi,
                "p": p, "page_level": {"lo": page_lo, "hi": page_hi, "p": page_p},
                "fragile": bool(p < ALPHA <= page_p),
                "resolvable": float(_Z80 * d.std(ddof=1) / np.sqrt(n)) if n > 1 else None}

    chain, reached = [], True
    for a, b, predicted in CHAIN:
        t = test(a.name, b.name, PRIMARY_METRIC, predicted)
        t["tested"] = reached
        if reached:
            t["significant"] = t["p"] < ALPHA
            t["verdict"] = _verdict(t["delta"], t["hi"], t["p"], predicted)
            reached = t["significant"]
        else:
            t["significant"] = None
            t["verdict"] = "not reached: an earlier link was not significant, so this is exploratory"
        chain.append(t)
    headline = chain[0]
    supporting = [test(INCUMBENT.name, CHALLENGER.name, m, 1) for m in SUPPORTING_METRICS]
    for t, p_holm in zip(supporting, holm([t["p"] for t in supporting])):
        t["p_holm"] = p_holm
        t["verdict"] = (_verdict(t["delta"], t["hi"], p_holm, 1) if headline["significant"]
                        else "exploratory: the headline was not significant")
    graded, answers = (next(t for t in supporting if t["metric"] == m) for m in ("ndcg@10", "ndcg_answers@10"))
    incumbent, challenger = matrix[INCUMBENT.name]["success@5"], matrix[CHALLENGER.name]["success@5"]
    return {
        "alpha": ALPHA,
        "procedure": "chain tested in order on the primary metric, each link only if all earlier links were "
                     "significant; supporting metrics Holm-adjusted, claimed only if the headline is significant",
        "chain": chain,
        "supporting": supporting,
        "ndcg_versions_agree": bool(np.sign(graded["delta"]) == np.sign(answers["delta"])),
        "exploratory": [test(a.name, b.name, PRIMARY_METRIC, 0) for a, b in EXPLORATORY_PAIRS
                        if a.name in matrix and b.name in matrix],
        "success@5_changes": {
            "fixed": int(np.count_nonzero((incumbent == 0) & (challenger == 1))),
            "broken": int(np.count_nonzero((incumbent == 1) & (challenger == 0))),
            "both_found": int(np.count_nonzero((incumbent == 1) & (challenger == 1))),
            "both_missed": int(np.count_nonzero((incumbent == 0) & (challenger == 0))),
        },
    }


def by_style_paired(
    questions: Sequence[Question], matrix: Mapping[str, Mapping[str, np.ndarray]],
    resamples: int = RESAMPLES, seed: int = SEED,
) -> dict:
    """Paired differences within each question style: the headline pair on every metric the tables
    show, and every other chain link and exploratory pair on the primary metric. Descriptive: no tests."""
    pairs = [(INCUMBENT.name, CHALLENGER.name, EFFECT_METRICS)] + [
        (a.name, b.name, (PRIMARY_METRIC,)) for a, b, _ in CHAIN[1:]
    ] + [(a.name, b.name, (PRIMARY_METRIC,)) for a, b in EXPLORATORY_PAIRS]
    out: dict = {}
    for style in STYLES:
        mask = np.array([q.style == style for q in questions])
        if not mask.any():
            continue
        idx = bootstrap_indices(int(mask.sum()), resamples, seed)
        rows = {}
        for a, b, metrics in pairs:
            if a in matrix and b in matrix:
                rows[f"{a} -> {b}"] = {
                    m: dict(zip(("delta", "lo", "hi"), paired_diff_ci(matrix[a][m][mask], matrix[b][m][mask], idx)))
                    for m in metrics
                }
        out[style] = {"n": int(mask.sum()), "pairs": rows}
    return out


def interaction(
    questions: Sequence[Question], matrix: Mapping[str, Mapping[str, np.ndarray]],
    resamples: int = RESAMPLES, seed: int = SEED,
) -> dict | None:
    """Descriptive: is dense retrieval's gain over BM25 larger on paraphrase than on keyword questions?

    The two styles hold different questions, so the interval resamples within
    each style and the p-value shuffles style labels between them.
    """
    a, b = INTERACTION_PAIR
    if a not in matrix or b not in matrix:
        return None
    first, second = INTERACTION_STYLES
    in_first = np.array([q.style == first for q in questions])
    in_second = np.array([q.style == second for q in questions])
    if not in_first.any() or not in_second.any():
        return None
    out = {"pair": f"{a} -> {b}", "styles": [first, second], "pre_specified_metric": INTERACTION_METRICS[0],
           "metrics": {}}
    for m in INTERACTION_METRICS:
        d = matrix[b][m] - matrix[a][m]
        diff, lo, hi = two_group_diff_ci(d[in_first], d[in_second], resamples, seed)
        out["metrics"][m] = {
            f"{first}_mean": float(d[in_first].mean()), f"{second}_mean": float(d[in_second].mean()),
            "difference": diff, "lo": lo, "hi": hi,
            "p": permutation_pvalue(d[in_first], d[in_second], resamples, seed + 2),
        }
    return out


def factor_effects(
    configs: Sequence[RetrievalConfig],
    matrix: Mapping[str, Mapping[str, np.ndarray]],
    idx: np.ndarray,
    resamples: int = RESAMPLES,
    seed: int = SEED,
) -> list[dict]:
    """What flipping one switch is worth, averaged over the matched settings of the others.

    For a switch going a -> b, each pair of configurations identical except for
    that switch gives a per-question difference. The effect is the average of
    those differences, with a bootstrap interval over questions. Each pair is
    kept too, with its own paired test (Holm-adjusted within the switch),
    because an average can hide a switch that helps one strategy and does
    nothing for another.
    """
    by_cell = {c.cell: c.name for c in configs}
    effects = []
    for factor, position, switches in FACTORS:
        # Retriever switches span every chunker x breadcrumb cell; the other switches are read within
        # each retriever family, so a breadcrumb effect on embeddings is never averaged with one on BM25.
        families = [(None, None)] if factor == "retriever" else list(RETRIEVER_FAMILIES.items())
        for family, members in families:
            for a, b in switches:
                matched = []
                for cell, name in by_cell.items():
                    if cell[position] == a and (members is None or cell[1] in members):
                        other = cell[:position] + (b,) + cell[position + 1:]
                        if other in by_cell:
                            matched.append((name, by_cell[other]))
                if not matched:
                    continue
                for m in EFFECT_METRICS:
                    per_pair = [matrix[nb][m] - matrix[na][m] for na, nb in matched]
                    delta, lo, hi = mean_ci(np.mean(per_pair, axis=0), idx)
                    pvalues = [sign_flip_pvalue(d, resamples, seed + 1) for d in per_pair]
                    pairs = {}
                    for (na, nb), d, pv, ph in zip(matched, per_pair, pvalues, holm(pvalues)):
                        d_mean, d_lo, d_hi = paired_diff_ci(np.zeros_like(d), d, idx)
                        pairs[f"{na} -> {nb}"] = {"delta": d_mean, "lo": d_lo, "hi": d_hi, "p": pv, "p_holm": ph}
                    effects.append({
                        "factor": factor, "within": family, "from": a, "to": b, "metric": m,
                        "delta": delta, "lo": lo, "hi": hi, "pairs": pairs,
                    })
    return effects


# --- provenance ------------------------------------------------------------------------------


def code_version() -> dict:
    """What code ran: a hash of the package itself, plus git's view when there is one.

    The package hash is the authority, since it works without git and can't be
    fooled by uncommitted edits. The commit and dirty flag are for humans, and
    are null when git can't answer.
    """
    package = Path(__file__).resolve().parent
    h = hashlib.sha256()
    for path in sorted(package.glob("*.py")):
        h.update(path.name.encode("utf-8") + b"\0" + path.read_bytes() + b"\0")
    return {"raglab_version": __version__, "package_sha256": h.hexdigest(), **_git_state(package.parent)}


def _git_state(stage_dir: Path) -> dict:
    def git(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(["git", *args], cwd=stage_dir, capture_output=True, text=True, timeout=10)

    try:
        commit = git("rev-parse", "--verify", "--quiet", "HEAD")
        if commit.returncode != 0:
            return {"git_commit": None, "git_dirty": None}
        status = git("status", "--porcelain", "--", "raglab", "pyproject.toml", "requirements-lock.txt", "PLAN.md",
                     "../models.lock.json", "../corpus.lock.json")
        dirty = bool(status.stdout.strip()) if status.returncode == 0 else None
        return {"git_commit": commit.stdout.strip(), "git_dirty": dirty}
    except (OSError, subprocess.SubprocessError):
        return {"git_commit": None, "git_dirty": None}


def _sha256_of(paths: Sequence[Path]) -> str | None:
    """One hash over files' names and bytes, or None when there are none."""
    existing = [p for p in sorted(paths) if p.is_file()]
    if not existing:
        return None
    h = hashlib.sha256()
    for p in existing:
        h.update(p.name.encode("utf-8") + b"\0" + p.read_bytes() + b"\0")
    return h.hexdigest()


def _chip() -> str | None:
    """The processor's marketing name (it decides the last bits of float arithmetic), or None."""
    if platform.system() == "Darwin":
        try:
            out = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, timeout=5)
            return out.stdout.strip() or None
        except (OSError, subprocess.SubprocessError):
            return None
    return platform.processor() or None


def plan_path() -> Path:
    return Path(__file__).resolve().parents[1] / "PLAN.md"


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(project_root()).as_posix()
    except ValueError:
        return path.name  # outside the project: never record an absolute (personal) path


def build_manifest(
    golden: GoldenSet,
    lock: CorpusLock,
    configs: Sequence[RetrievalConfig],
    resamples: int = RESAMPLES,
    seed: int = SEED,
    now: datetime | None = None,
) -> dict:
    from raglab.models import default_models_lock_path
    from raglab.neural import THREADS, library_versions, model_lock, model_settings

    now = now or datetime.now(timezone.utc)
    code = code_version()
    models = model_lock()
    folder = golden.path.parent
    return {
        "schema": SCHEMA,
        "run_id": f"{now:%Y%m%d-%H%M%S}-{code['package_sha256'][:8]}",
        "created_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "code": code,
        "corpus": {
            "repo": lock.repo, "ref": lock.ref, "commit": lock.commit,
            "fingerprint": lock.fingerprint, "files": lock.file_count,
        },
        "golden": {
            "path": _relative(golden.path), "version": golden.version,
            "sha256": golden.sha256, "questions": len(golden.questions),
        },
        # What the README's robustness claims are computed from, so they are pinned too: every
        # study file beside the golden set (v1's and v2's), the judging prompts, and the analysis scripts.
        "robustness_inputs": {
            **{p.name: _sha256_of([p]) for p in sorted(folder.glob("v[0-9]*-*.json"))},
            "prompts_sha256": _sha256_of(list((folder / "prompts").glob("*.md"))),
            "analysis_sha256": _sha256_of(list((Path(__file__).resolve().parents[1] / "analysis").glob("*.py"))),
        },
        "plan": {"path": _relative(plan_path()), "sha256": _sha256_of([plan_path()])},
        "models": {
            "lock_sha256": _sha256_of([default_models_lock_path()]),
            "embedder": model_settings(models["embedder"], "embedder"),
            "reranker": model_settings(models["reranker"], "reranker"),
        },
        "numerics": {"threads": THREADS, "platform": f"{platform.system()} {platform.machine()}", "chip": _chip(),
                     **library_versions()},
        "confirmatory_plan": {
            "primary_metric": PRIMARY_METRIC, "alpha": ALPHA,
            "chain": [[a.name, b.name, predicted] for a, b, predicted in CHAIN],
            "supporting_metrics": list(SUPPORTING_METRICS),
            "exploratory_pairs": [[a.name, b.name] for a, b in EXPLORATORY_PAIRS],
            "procedure": "chain tested in order on the primary metric, each link only if all earlier links were "
                         "significant; supporting metrics Holm-adjusted, claimed only if the headline is significant",
        },
        "configs": [c.manifest() for c in configs],
        "baseline": BASELINE.name,
        "scoring": {
            "unit": "section (doc_id#anchor); a chunk retrieves a section when it holds at least "
                    "min(min_overlap_chars, the section's body, all the section-body text the chunk holds) "
                    "characters of that section's body",
            "min_overlap_chars": MIN_OVERLAP_CHARS,
            "chunks_ranked": "lexical retrievers and bm25+rerank: every chunk with a positive score; dense, hybrid "
                             "and hybrid+rerank: every chunk",
            "rank_depth_sections": RANK_DEPTH,
            "k": list(K_VALUES),
            "budgets_chars": list(BUDGETS),
            "headline_budget_chars": HEADLINE_BUDGET,
            "gain": {str(g): v for g, v in GAIN.items()},
        },
        "statistics": {
            "bootstrap_resamples": resamples, "confidence": CONFIDENCE, "seed": seed,
            "interval": "percentile bootstrap over questions; Wilson score interval for yes/no metrics",
            "test": "paired sign-flip randomisation, two-sided; Holm-adjusted across configurations, per metric",
        },
        "environment": {
            "python": platform.python_version(), "numpy": np.__version__,
            "scikit-learn": sklearn.__version__, "pyyaml": yaml.__version__,
            "os": f"{platform.system()} {platform.machine()}",
        },
    }


# --- saving and loading runs --------------------------------------------------------------


def _rounded(x: float) -> float:
    return round(float(x), 6)


def _round_tree(obj):
    if isinstance(obj, float):
        return _rounded(obj)
    if isinstance(obj, dict):
        return {k: _round_tree(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round_tree(v) for v in obj]
    return obj


def _write(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    tmp.replace(path)


def save_run(
    run_dir: Path,
    manifest: Mapping,
    summary: Mapping,
    questions: Sequence[Question],
    results: RunResults,
    overwrite: bool = False,
) -> None:
    if (run_dir / "manifest.json").exists() and not overwrite:
        raise FileExistsError(f"{run_dir} already holds a run; choose another --out or pass --overwrite")
    run_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    for i, q in enumerate(questions):
        for name, per_question in results.items():
            r = per_question[i]
            line = {
                "question": q.id, "config": name,
                "values": [_rounded(r.values[m]) for m in METRICS],
                "ranking": list(r.ranking), "top5_chars": r.top5_chars,
            }
            if r.rerank_depth is not None:
                line["rerank_depth"] = r.rerank_depth
            lines.append(json.dumps(line, ensure_ascii=False))
    _write(run_dir / "questions.jsonl", "\n".join(lines) + "\n")
    _write(run_dir / "summary.json", json.dumps(_round_tree(dict(summary)), indent=1, ensure_ascii=False) + "\n")
    _write(run_dir / "manifest.json", json.dumps(dict(manifest), indent=1, ensure_ascii=False) + "\n")


_MANIFEST_KEYS = ("schema", "run_id", "code", "corpus", "golden", "statistics", "models", "plan", "confirmatory_plan")
_SUMMARY_KEYS = ("questions", "baseline", "configs", "headline", "summary", "vs_baseline", "effects", "by_style",
                 "diagnostics", "confirmatory", "by_style_paired", "interaction")


def load_run(run_dir: Path) -> tuple[dict, dict]:
    """(manifest, summary) of a saved run, with its format and top-level shape checked.

    Deeper damage (a missing metric, a wrong type inside) is caught where the
    reports render, and reported as a damaged run rather than a traceback.
    """
    try:
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, RecursionError) as e:  # ValueError covers JSONDecodeError and bad UTF-8
        raise ValueError(f"{run_dir} is not a readable run: {type(e).__name__}: {e}") from e
    if not isinstance(manifest, dict) or not isinstance(summary, dict):
        raise ValueError(f"{run_dir} is not a run: manifest.json and summary.json must be JSON objects")
    if manifest.get("schema") != SCHEMA:
        raise ValueError(f"{run_dir} was written by a different run format (schema {manifest.get('schema')!r}, "
                         f"this code reads {SCHEMA}); re-run `eval run` to regenerate it")
    missing = [f"manifest.{k}" for k in _MANIFEST_KEYS if k not in manifest] + \
              [f"summary.{k}" for k in _SUMMARY_KEYS if k not in summary]
    if missing:
        raise ValueError(f"{run_dir} is an incomplete run: missing {', '.join(missing)}")
    return manifest, summary
