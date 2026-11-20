"""Score every configuration against the golden set, with uncertainty, and record how to reproduce it.

A run:

1. builds each configuration's retriever over the verified corpus;
2. ranks every chunk with a positive score for every question, so even the
   largest budget is filled whenever the retriever has enough to fill it, and
   scores the ranking (metrics.py);
3. summarises them. That means a bootstrap interval on every mean, paired
   differences against the baseline with Holm-adjusted p-values, and the
   average effect of each switch in the factorial;
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
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import sklearn
import yaml

from raglab import __version__
from raglab.corpus import CorpusLock, Document, project_root
from raglab.experiments import BASELINE, RetrievalConfig, build_retrievers
from raglab.golden import STYLES, GoldenSet, Question
from raglab.metrics import (
    BINARY_METRICS, BUDGETS, GAIN, HEADLINE_BUDGET, HEADLINE_METRICS, K_VALUES, METRICS, RANK_DEPTH, Span, pack, score,
)
from raglab.retrieval import Retriever
from raglab.sections import MIN_OVERLAP_CHARS, SectionIndex
from raglab.stats import (
    CONFIDENCE, RESAMPLES, SEED, bootstrap_indices, holm, mean_ci, paired_diff_ci, sign_flip_pvalue, wilson_ci,
)

SCHEMA = 3  # 3: best coverage, Wilson intervals for yes/no rates, robustness-file hashes in the manifest
# Switch effects cover every metric the README's tables show.
EFFECT_METRICS = (
    "success@5", "recall@5", "mrr@10", "ndcg@10",
    *(f"{m}@{HEADLINE_BUDGET}" for m in ("budget_success", "budget_recall", "coverage", "best_coverage")),
)

# The factorial's switches: position in RetrievalConfig.cell, and the comparisons reported.
FACTORS: tuple[tuple[str, int, tuple[tuple[object, object], ...]], ...] = (
    ("chunker", 0, (("paragraph", "heading"), ("paragraph", "fixed"), ("fixed", "heading"))),
    ("retriever", 1, (("tfidf", "bm25"),)),
    ("breadcrumb", 2, ((False, True),)),
)


@dataclass(frozen=True)
class QuestionResult:
    values: Mapping[str, float]
    ranking: tuple[str | None, ...]  # top sections; None = a chunk that retrieved no section
    top5_chars: int                  # how much text the first five chunks are
    in_budget: frozenset[str] = frozenset()  # sections retrieved inside the headline budget


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
            spans = ranked_spans(retriever, q.text)
            values, ranking = score(spans, q.primary, q.judgments, index)
            in_budget = frozenset(
                ref.location for doc_id, start, end in pack(spans, HEADLINE_BUDGET)
                for ref, _ in index.retrieved_by(doc_id, start, end)
            )
            per_question.append(QuestionResult(
                values, tuple(ranking), sum(end - start for _, start, end in spans[:5]), in_budget,
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
        judged, judged_budget, sections_budget, empty = [], [], [], []
        for q, r in zip(questions, results[name]):
            found = [loc for loc in r.ranking if loc is not None]
            judged.append(sum(loc in q.judgments for loc in found) / len(found) if found else 1.0)
            judged_budget.append(sum(loc in q.judgments for loc in r.in_budget) / len(r.in_budget) if r.in_budget else 1.0)
            sections_budget.append(len(r.in_budget))
            empty.append(sum(loc is None for loc in r.ranking) / RANK_DEPTH)
        diagnostics[name] = {
            "judged@10": float(np.mean(judged)),
            f"judged_in_budget@{HEADLINE_BUDGET}": float(np.mean(judged_budget)),
            f"sections_in_budget@{HEADLINE_BUDGET}": float(np.mean(sections_budget)),
            "empty_slots@10": float(np.mean(empty)),
            "top5_chars": float(np.mean([r.top5_chars for r in results[name]])),
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
    }


def factor_effects(
    configs: Sequence[RetrievalConfig],
    matrix: Mapping[str, Mapping[str, np.ndarray]],
    idx: np.ndarray,
    resamples: int = RESAMPLES,
    seed: int = SEED,
) -> list[dict]:
    """What flipping one switch is worth, averaged over every setting of the other two.

    For a switch going a -> b, each pair of configurations identical except for
    that switch gives a per-question difference. The effect is the average of
    those differences, with a bootstrap interval over questions. Each pair is
    kept too, with its own paired test (Holm-adjusted within the switch),
    because an average can hide a switch that helps one strategy and does
    nothing for another.
    """
    by_cell = {c.cell: c.name for c in configs}
    effects = []
    for factor, position, pairs in FACTORS:
        for a, b in pairs:
            matched = []
            for cell, name in by_cell.items():
                if cell[position] == a:
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
                    "factor": factor, "from": a, "to": b, "metric": m,
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
        status = git("status", "--porcelain", "--", "raglab", "pyproject.toml")
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
    now = now or datetime.now(timezone.utc)
    code = code_version()
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
        # What the README's robustness claims are computed from, so they are pinned too.
        "robustness_inputs": {
            "agreement_sha256": _sha256_of([golden.path.with_name(f"{golden.path.stem}-agreement.json")]),
            "unjudged_sample_sha256": _sha256_of([golden.path.with_name(f"{golden.path.stem}-unjudged-sample.json")]),
            "regrade_log_sha256": _sha256_of([golden.path.with_name(f"{golden.path.stem}-regrade.json")]),
            "analysis_sha256": _sha256_of(list((Path(__file__).resolve().parents[1] / "analysis").glob("*.py"))),
        },
        "configs": [c.manifest() for c in configs],
        "baseline": BASELINE.name,
        "scoring": {
            "unit": "section (doc_id#anchor); a chunk retrieves a section when it holds at least "
                    "min(min_overlap_chars, the section's body, all the section-body text the chunk holds) "
                    "characters of that section's body",
            "min_overlap_chars": MIN_OVERLAP_CHARS,
            "chunks_ranked": "every chunk with a positive score",
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
            lines.append(json.dumps({
                "question": q.id, "config": name,
                "values": [_rounded(r.values[m]) for m in METRICS],
                "ranking": list(r.ranking), "top5_chars": r.top5_chars,
            }, ensure_ascii=False))
    _write(run_dir / "questions.jsonl", "\n".join(lines) + "\n")
    _write(run_dir / "summary.json", json.dumps(_round_tree(dict(summary)), indent=1, ensure_ascii=False) + "\n")
    _write(run_dir / "manifest.json", json.dumps(dict(manifest), indent=1, ensure_ascii=False) + "\n")


_MANIFEST_KEYS = ("schema", "run_id", "code", "corpus", "golden", "statistics")
_SUMMARY_KEYS = ("questions", "baseline", "configs", "headline", "summary", "vs_baseline", "effects", "by_style", "diagnostics")


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
