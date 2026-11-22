"""What if the breadcrumb were sent to the model, not just indexed?

Budget metrics count chunk text only. The breadcrumb ("Handling Errors > Use
HTTPException") is an indexing feature: how a prompt labels its chunks is a
Stage 4 decision, and it applies to every configuration alike. But if a
breadcrumb configuration also sent each chunk's breadcrumb, its prompt would
fit less. This script charges that cost (breadcrumb plus a newline per chunk)
to every breadcrumb configuration, and reports the 4,800-char budget metrics
both ways, so the effect of that choice is visible rather than assumed.

    cd "Stage 2" && ../.venv/bin/python analysis/breadcrumb_budget.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from raglab.corpus import CorpusLock, default_corpus_dir, default_lock_path, load_corpus  # noqa: E402
from raglab.experiments import BASELINE, FACTORIAL, build_retrievers  # noqa: E402
from raglab.golden import default_golden_path, load_golden  # noqa: E402
from raglab.metrics import HEADLINE_BUDGET, coverage, delivered  # noqa: E402
from raglab.sections import SectionIndex  # noqa: E402


def packed(hits, budget: int, charge_breadcrumb: bool) -> list[tuple[str, int, int]]:
    """Longest prefix that fits, where each chunk may also pay for its breadcrumb."""
    out, used = [], 0
    for h in hits:
        cost = h.chunk.end - h.chunk.start
        if charge_breadcrumb and h.chunk.heading_path:
            cost += len(" > ".join(h.chunk.heading_path)) + 1
        if used + cost > budget:
            break
        out.append((h.chunk.doc_id, h.chunk.start, h.chunk.end))
        used += cost
    return out


def main() -> None:
    lock = CorpusLock.load(default_lock_path())
    docs = list(load_corpus(default_corpus_dir(lock.ref), lock=lock).documents)
    index = SectionIndex(docs)
    golden = load_golden(default_golden_path())
    retrievers = build_retrievers(docs, FACTORIAL)
    print(f"{'configuration':32} {'recall in 4.8k':>22} {'success in 4.8k':>22} {'coverage in 4.8k':>22}")
    print(f"{'':32} {'text only -> +crumb':>22} {'text only -> +crumb':>22} {'text only -> +crumb':>22}")
    for config in FACTORIAL:
        rows = {False: [], True: []}
        for q in golden.questions:
            hits = retrievers[config.name].search(q.text, len(retrievers[config.name].chunks))
            refs = [index.get(loc) for loc in sorted(q.primary)]
            for charge in (False, True):
                ctx = packed(hits, HEADLINE_BUDGET, charge and config.with_context)
                arrived = q.primary & delivered(ctx, index)
                rows[charge].append((len(arrived) / len(q.primary), float(bool(arrived)), coverage(ctx, refs, index)))
        a, b = np.mean(rows[False], axis=0), np.mean(rows[True], axis=0)
        name = config.name + ("" if config.with_context else "  (no breadcrumb: unchanged)")
        print(f"{name:32} " + " ".join(f"{x:>13.3f} -> {y:.3f}" for x, y in zip(a, b)))
    print(f"\nbaseline = {BASELINE.name}; budget = {HEADLINE_BUDGET:,} characters")


if __name__ == "__main__":
    main()
