"""Do the three question styles really differ? Word overlap with the target section.

For each question: the share of its distinct content words (scikit-learn's
English analyzer: lowercase word tokens of 2+ characters, English stop words
removed, which is the analyzer the BM25 retriever uses) that also appear in its
target section's heading path or body. Reported as the mean per style, with a
95% bootstrap interval. A paraphrase that reused the section's own words would
show up here.

    cd "Stage 2" && ../.venv/bin/python analysis/question_style_overlap.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from raglab.corpus import CorpusLock, default_corpus_dir, default_lock_path, load_corpus  # noqa: E402
from raglab.golden import STYLES, default_golden_path, load_golden  # noqa: E402
from raglab.sections import SectionIndex  # noqa: E402
from raglab.stats import bootstrap_indices, mean_ci  # noqa: E402


def overlaps() -> dict[str, list[float]]:
    lock = CorpusLock.load(default_lock_path())
    docs = {d.doc_id: d for d in load_corpus(default_corpus_dir(lock.ref), lock=lock).documents}
    index = SectionIndex(docs.values())
    analyze = CountVectorizer(stop_words="english").build_analyzer()
    by_style: dict[str, list[float]] = {s: [] for s in STYLES}
    for q in load_golden(default_golden_path()).questions:
        ref = index.get(q.target)
        section_words = set(analyze(" ".join(ref.heading_path) + "\n" + docs[ref.doc_id].text[ref.start:ref.end]))
        question_words = set(analyze(q.text))
        by_style[q.style].append(len(question_words & section_words) / len(question_words))
    return by_style


def main() -> None:
    for style, values in overlaps().items():
        mean, lo, hi = mean_ci(values, bootstrap_indices(len(values)))
        print(f"{style:10} n={len(values):2}  share of question words found in the target section: "
              f"{mean:.2f} (95% CI {lo:.2f}-{hi:.2f})")


if __name__ == "__main__":
    main()
