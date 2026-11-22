"""Would the second judge's grades change the results?

golden/v1-agreement.json holds a blind second judging of a stratified sample
of v1's labels. This script re-scores every configuration under two variants
of the labels:

  swap  every disputed grade in the sample is replaced by the second judge's,
        targets included; a question left with no answering section can't be
        scored, so it is dropped, and the script names it;
  drop  every question whose target the second judge did not call an answer
        is removed.

For each variant it reports the largest absolute change over all 12
configurations and every metric the README's tables show, both in a mean and
in a difference from the baseline. It also prints the signed lead of
heading+bm25 over the baseline, so the direction is visible and not only the
size. Because the sample is stratified, it also estimates how many of v1's
labels a full re-judging would dispute.

    cd "Stage 2" && ../.venv/bin/python analysis/label_sensitivity.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import MappingProxyType

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from raglab.corpus import CorpusLock, default_corpus_dir, default_lock_path, load_corpus  # noqa: E402
from raglab.evaluate import EFFECT_METRICS, run_configs  # noqa: E402
from raglab.experiments import BASELINE, FACTORIAL  # noqa: E402
from raglab.golden import Question, default_golden_path, load_golden  # noqa: E402
from raglab.metrics import LABELS  # noqa: E402
from raglab.sections import SectionIndex  # noqa: E402

BEST = "heading+bm25"


def means(docs, index, questions) -> dict[str, dict[str, float]]:
    results = run_configs(docs, questions, FACTORIAL, index)
    return {n: {m: float(np.mean([r.values[m] for r in per])) for m in EFFECT_METRICS} for n, per in results.items()}


def largest_moves(base, variant) -> tuple[float, str, float, str]:
    mean_moves = [(abs(variant[n][m] - base[n][m]), f"{n}, {LABELS[m]}") for n in base for m in EFFECT_METRICS]
    delta_moves = [
        (abs((variant[n][m] - variant[BASELINE.name][m]) - (base[n][m] - base[BASELINE.name][m])), f"{n} vs baseline, {LABELS[m]}")
        for n in base if n != BASELINE.name for m in EFFECT_METRICS
    ]
    (m1, w1), (m2, w2) = max(mean_moves), max(delta_moves)
    return m1, w1, m2, w2


def main() -> None:
    lock = CorpusLock.load(default_lock_path())
    docs = list(load_corpus(default_corpus_dir(lock.ref), lock=lock).documents)
    index = SectionIndex(docs)
    golden = load_golden(default_golden_path())
    study = json.loads((default_golden_path().parent / "v1-agreement.json").read_text(encoding="utf-8"))
    items, population = study["items"], study["population"]
    disputed = [i for i in items if i["first"] != i["second"]]
    doubted = sorted({i["question"] for i in items if i["is_target"] and i["second"] != 2})

    def stratum(i) -> str:
        return "targets" if i["is_target"] else {2: "answers", 1: "related", 0: "rejected"}[i["first"]]

    estimate = answer_line = 0.0
    for s, size in population.items():
        in_stratum = [i for i in items if stratum(i) == s]
        if in_stratum:
            estimate += size * sum(i["first"] != i["second"] for i in in_stratum) / len(in_stratum)
            answer_line += size * sum((i["first"] == 2) != (i["second"] == 2) for i in in_stratum) / len(in_stratum)
    print(f"{len(items)} sampled labels, {len(disputed)} disputed (touching {len({i['question'] for i in disputed})} questions); "
          f"targets the second judge doubted: {doubted or 'none'}")
    print(f"reweighted to all {sum(population.values()):,} labels: about {estimate:.0f} would be disputed if all were "
          f"re-judged, about {answer_line:.0f} of them on the answer / not-an-answer line")

    swapped, unscorable = [], []
    for q in golden.questions:
        judgments = dict(q.judgments)
        for i in disputed:
            if i["question"] == q.id:
                judgments[i["location"]] = i["second"]
        if any(g == 2 for g in judgments.values()):
            swapped.append(Question(q.id, q.text, q.style, q.target, MappingProxyType(judgments)))
        else:
            unscorable.append(q.id)
    dropped = [q for q in golden.questions if q.id not in doubted]

    base = means(docs, index, golden.questions)
    variants = {
        "swap": (swapped, f"{len(disputed)} grades swapped; dropped as unscorable: {unscorable or 'none'}"),
        "drop": (dropped, f"the {len(golden.questions) - len(dropped)} questions whose target was doubted dropped"),
    }
    scored = {name: means(docs, index, qs) for name, (qs, _) in variants.items()}
    for name, (_, note) in variants.items():
        m1, w1, m2, w2 = largest_moves(base, scored[name])
        print(f"\n{name}: {note}")
        print(f"  largest change in a mean: {m1:.4f} ({w1})")
        print(f"  largest change in a difference from the baseline: {m2:.4f} ({w2})")
    print(f"\n{BEST}'s lead over the baseline (as labelled -> swap / drop):")
    for m in EFFECT_METRICS:
        lead = lambda t: t[BEST][m] - t[BASELINE.name][m]  # noqa: E731
        print(f"  {LABELS[m]:28} {lead(base):+.3f} -> {lead(scored['swap']):+.3f} / {lead(scored['drop']):+.3f}")


if __name__ == "__main__":
    main()
