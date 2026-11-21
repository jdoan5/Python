"""How far last-bit float differences travel (PLAN.md, "Perturbation check"). Run after freeze point 2.

Other hardware can give the models' outputs different last bits. To see what
that could change, every dense similarity and every cross-encoder score is
moved by one float32 unit in the last place, up or down by a coin flip seeded
with 20261120, in 20 runs (seed + run). BM25 and TF-IDF are left alone: they
are float64 sums, not model outputs. Reported: for each chain configuration,
how often its top-10 sections change, and the largest change in link 1's
MRR@10 difference.

    cd "Stage 3" && ../.venv/bin/python analysis/perturbation.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from raglab import neural  # noqa: E402
from raglab.corpus import CorpusLock, default_corpus_dir, default_lock_path, load_corpus  # noqa: E402
from raglab.evaluate import run_configs  # noqa: E402
from raglab.experiments import CHAIN, CHALLENGER, INCUMBENT  # noqa: E402
from raglab.golden import default_golden_path, load_golden  # noqa: E402
from raglab.sections import SectionIndex  # noqa: E402

RUNS = 20
SEED = 20261120


def nudge(values, rng: np.random.Generator) -> np.ndarray:
    """Each value moved by one float32 unit in the last place, up or down at random."""
    v = np.asarray(values, dtype=np.float64)
    ulp = np.spacing(np.abs(v).astype(np.float32)).astype(np.float64)
    return v + rng.choice((-1.0, 1.0), size=v.shape) * ulp


def chain_configs():
    seen, out = set(), []
    for a, b, _ in CHAIN:
        for c in (a, b):
            if c.name not in seen:
                seen.add(c.name)
                out.append(c)
    return tuple(out)


def link1(results) -> float:
    mrr = {n: np.mean([r.values["mrr@10"] for r in results[n]]) for n in (INCUMBENT.name, CHALLENGER.name)}
    return float(mrr[CHALLENGER.name] - mrr[INCUMBENT.name])


def main() -> None:
    golden_dir = default_golden_path(2).parent
    if not (golden_dir / "v2-blind.json").is_file():
        raise SystemExit("v2's labels aren't final (no v2-blind.json); this check runs after freeze point 2")
    lock = CorpusLock.load(default_lock_path())
    docs = list(load_corpus(default_corpus_dir(lock.ref), lock=lock).documents)
    index = SectionIndex(docs)
    v2 = load_golden(default_golden_path(2))
    configs = chain_configs()
    base = run_configs(docs, v2.questions, configs, index)
    base_diff = link1(base)
    dense_scores, rerank_scores = neural.DenseRetriever.scores, neural.CachedScorer.score
    changed = {c.name: 0 for c in configs}
    diffs = []
    try:
        for run in range(RUNS):
            rng = np.random.default_rng(SEED + run)
            neural.DenseRetriever.scores = lambda self, q, f=dense_scores: nudge(f(self, q), rng)
            neural.CachedScorer.score = lambda self, q, t, f=rerank_scores: nudge(f(self, q, t), rng)
            perturbed = run_configs(docs, v2.questions, configs, index)
            for c in configs:
                changed[c.name] += sum(a.ranking != b.ranking for a, b in zip(base[c.name], perturbed[c.name]))
            diffs.append(link1(perturbed))
    finally:
        neural.DenseRetriever.scores, neural.CachedScorer.score = dense_scores, rerank_scores
    result = {
        "runs": RUNS, "questions": len(v2.questions),
        "top10_changed_share": {n: k / (RUNS * len(v2.questions)) for n, k in changed.items()},
        "link1_mrr@10_difference": {"unperturbed": base_diff,
                                    "largest_change": float(max(abs(d - base_diff) for d in diffs))},
    }
    path = Path(__file__).resolve().parents[1] / "results" / "perturbation.json"
    path.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
