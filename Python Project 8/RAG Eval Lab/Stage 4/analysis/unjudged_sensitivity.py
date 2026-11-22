"""Could the sections nobody judged change Stage 2's conclusions?

Labels were pooled from every configuration's top 10 and from each target's
own page. A 4,800-character context reaches further, and anything unjudged
scores as "not relevant". golden/v1-unjudged-sample.json judges those sections in two
strata (see analysis/samples.py):

- focused:   every unjudged section inside a heading or fixed configuration's
             context, all of them judged, so their answers are known exactly;
- scattered: unjudged sections that only paragraph-family contexts reach; a
             random 200 are judged, giving an answer rate p.

This script completes the labels in simulation. Known answers from both strata
are added. Every unsampled scattered pair becomes an answer with probability p,
for p = the sampled rate and p = the upper end of its 95% Wilson interval. Each
completion is scored. Every lead and ranking listed in `conclusions` is
reported as a range over the simulations, growing margins as well as shrinking
ones. Success@5 and MRR@10 can't move, by construction: every top-10 section is
already judged. Recall@k can, because a new answer anywhere enlarges its
denominator; nDCG@10's ideal changes the same way and is not simulated here.
Only unjudged sections inside some 4,800-character context are simulated, not
ones outside every context.

    cd "Stage 2" && ../.venv/bin/python analysis/unjudged_sensitivity.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from types import MappingProxyType

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from raglab.corpus import CorpusLock, default_corpus_dir, default_lock_path, load_corpus  # noqa: E402
from raglab.evaluate import ranked_spans  # noqa: E402
from raglab.experiments import BASELINE, FACTORIAL, build_retrievers  # noqa: E402
from raglab.golden import default_golden_path, load_golden  # noqa: E402
from raglab.metrics import HEADLINE_BUDGET, score  # noqa: E402
from raglab.sections import SectionIndex  # noqa: E402

DRAWS = 100
SEED = 20261120
BEST = "heading+bm25"
SMALL = ("paragraph+bm25", "paragraph+tfidf", BASELINE.name)
B = HEADLINE_BUDGET


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    centre, spread = p + z * z / (2 * n), z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre - spread) / (1 + z * z / n), (centre + spread) / (1 + z * z / n)


def main() -> None:
    lock = CorpusLock.load(default_lock_path())
    docs = list(load_corpus(default_corpus_dir(lock.ref), lock=lock).documents)
    index = SectionIndex(docs)
    golden = load_golden(default_golden_path())
    sample = json.loads((default_golden_path().parent / "v1-unjudged-sample.json").read_text(encoding="utf-8"))
    population = sample["population"]
    known_answers = [(i["question"], i["location"]) for i in sample["items"] if i["grade"] == 2]
    scattered = [i for i in sample["items"] if i["stratum"] == "scattered"]
    k = sum(i["grade"] == 2 for i in scattered)
    rate, (_, upper) = k / len(scattered), wilson(k, len(scattered))
    focused_answers = sum(1 for i in sample["items"] if i["stratum"] == "focused" and i["grade"] == 2)

    retrievers = build_retrievers(docs, FACTORIAL)
    spans = {
        (c.name, q.id): ranked_spans(retrievers[c.name], q.text)
        for c in FACTORIAL for q in golden.questions
    }
    from samples import unjudged_population  # the same population definition as the draw
    pop = unjudged_population(docs, index, golden)
    sampled = {(i["question"], i["location"]) for i in sample["items"]}
    unsampled_scattered = sorted(p for p, s in pop.items() if s == "scattered" and p not in sampled)

    def evaluate(extra: list[tuple[str, str]]) -> dict[str, dict[str, float]]:
        by_q: dict[str, list[str]] = {}
        for qid, loc in extra:
            by_q.setdefault(qid, []).append(loc)
        totals = {c.name: {} for c in FACTORIAL}
        for q in golden.questions:
            judgments = dict(q.judgments)
            judgments.update({loc: 2 for loc in by_q.get(q.id, [])})
            primary = frozenset(loc for loc, g in judgments.items() if g == 2)
            for c in FACTORIAL:
                values, _ = score(spans[(c.name, q.id)], primary, MappingProxyType(judgments), index)
                for m, v in values.items():
                    totals[c.name][m] = totals[c.name].get(m, 0.0) + v / len(golden.questions)
        return totals

    def conclusions(t: dict[str, dict[str, float]]) -> dict[str, float]:
        out = {}
        for metric in ("success@5", "recall@5", f"coverage@{B}", f"best_coverage@{B}"):
            out[f"{metric} lead, heading+bm25 over baseline"] = t[BEST][metric] - t[BASELINE.name][metric]
        for small in SMALL:
            # Finding 4 in both directions: small chunks ahead on arriving at all, behind on how much arrives.
            out[f"budget_success@{B} lead, {small} over heading+bm25"] = t[small][f"budget_success@{B}"] - t[BEST][f"budget_success@{B}"]
            out[f"budget_recall@{B} lead, {small} over heading+bm25"] = t[small][f"budget_recall@{B}"] - t[BEST][f"budget_recall@{B}"]
            out[f"coverage@{B} lead, heading+bm25 over {small}"] = t[BEST][f"coverage@{B}"] - t[small][f"coverage@{B}"]
            out[f"best_coverage@{B} lead, heading+bm25 over {small}"] = t[BEST][f"best_coverage@{B}"] - t[small][f"best_coverage@{B}"]
        for metric, label in (("success@5", "Success@5"), ("recall@5", "Recall@5"), (f"coverage@{B}", "mean coverage")):
            top = max(t, key=lambda n: t[n][metric])
            out[f"top configuration by {label} is heading+bm25"] = float(top == BEST)
            out[f"top configuration by {label} is a heading+bm25 variant"] = float(top.startswith(BEST))
        return out

    print(f"unjudged pairs inside some {B:,}-char context: focused {population['focused']:,} (all judged: "
          f"{focused_answers} answers), scattered {population['scattered']:,} (sampled {len(scattered)}: {k} answers "
          f"-> p = {rate:.1%}, 95% Wilson upper bound {upper:.1%})")
    for label, extra in (("as labelled", []), ("plus the sample's known answers", known_answers)):
        c = conclusions(evaluate(extra))
        print(f"\n{label}:")
        for key, v in c.items():
            print(f"  {key}: " + (("yes" if v else "no") if key.startswith("top configuration") else f"{v:+.3f}"))
    rng = np.random.default_rng(SEED)
    for p in (rate, upper):
        runs = [conclusions(evaluate(known_answers + [x for x in unsampled_scattered if rng.random() < p]))
                for _ in range(DRAWS)]
        print(f"\ncompleting the scattered stratum at p = {p:.1%} ({DRAWS} simulations, ~{p * len(unsampled_scattered):.0f} new answers each):")
        for key in runs[0]:
            vals = np.array([r[key] for r in runs])
            if key.startswith("top configuration"):
                print(f"  {key}: in {vals.mean():.0%} of simulations")
            else:
                print(f"  {key}: median {np.median(vals):+.3f}, range {vals.min():+.3f} to {vals.max():+.3f}, "
                      f"positive in {np.mean(vals > 0):.0%}")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
