"""The extras' paired differences (PLAN.md, "The extras and two more pairs"). Exploratory: no claims.

Added after freeze point 2. The plan lists the instruction's effect and the
rerank-depth curve among the exploratory results; the frozen evaluation
reports the extras' means only. This script gives each one a paired
difference, with Stage 2's machinery and seeds (bootstrap interval, sign-flip
p, both per question and per target page):

- the query instruction: heading+dense against heading+dense [no instruction];
- the rerank-depth curve: reranking the hybrid's top 10, 20, 50 and 100
  (the factorial's own cell), each against the hybrid it reranks.

    cd "Stage 3" && ../.venv/bin/python analysis/extras.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from raglab.corpus import CorpusLock, default_corpus_dir, default_lock_path, load_corpus  # noqa: E402
from raglab.evaluate import _matrix, paired_test, run_configs  # noqa: E402
from raglab.experiments import EVALUATED  # noqa: E402
from raglab.golden import default_golden_path, load_golden  # noqa: E402
from raglab.metrics import LABELS  # noqa: E402
from raglab.sections import SectionIndex  # noqa: E402
from raglab.stats import bootstrap_indices  # noqa: E402

METRICS = ("mrr@10", "success@5", "recall@5", "ndcg_answers@10")
# (a, b): how much b differs from a.
PAIRS = (
    ("heading+dense [no instruction]", "heading+dense"),
    *(("heading+hybrid", f"heading+hybrid+rerank{d}")
      for d in (" [depth 10]", " [depth 20]", " [depth 50]", "")),
)


def main() -> None:
    golden_dir = default_golden_path(2).parent
    if not (golden_dir / "v2-blind.json").is_file():
        raise SystemExit("v2's labels aren't final (no v2-blind.json); this runs after freeze point 2")
    lock = CorpusLock.load(default_lock_path())
    docs = list(load_corpus(default_corpus_dir(lock.ref), lock=lock).documents)
    v2 = load_golden(default_golden_path(2))
    names = {n for pair in PAIRS for n in pair}
    configs = tuple(c for c in EVALUATED if c.name in names)
    values = _matrix(run_configs(docs, v2.questions, configs, SectionIndex(docs)))
    idx = bootstrap_indices(len(v2.questions))
    out = [{"from": a, "to": b, **{m: paired_test(v2.questions, values[a][m], values[b][m], 0, idx)
                                   for m in METRICS}} for a, b in PAIRS]
    path = Path(__file__).resolve().parents[1] / "results" / "extras.json"
    path.write_text(json.dumps({"metrics": list(METRICS), "pairs": out}, indent=1) + "\n", encoding="utf-8")
    for row in out:
        print(f"{row['to']} vs {row['from']}")
        for m in METRICS:
            t = row[m]
            print(f"  {LABELS[m]:24} {t['delta']:+.3f} ({t['lo']:+.3f} to {t['hi']:+.3f})  p {t['p']:.4f}  "
                  f"per page {t['page_level']['p']:.4f}")
    print(f"wrote {path.relative_to(path.parents[1])}")


if __name__ == "__main__":
    main()
