# RAG Eval Lab

Question answering over the FastAPI documentation — built so that every choice
in the pipeline is **measured** rather than assumed.

Most retrieval-augmented generation demos show that a pipeline answers a few
questions. This project asks a harder one: *which* chunking strategy, retriever,
and model actually work, by how much, and at what cost — answered with a
labelled question set, standard retrieval metrics, and a CI gate that keeps the
answer from quietly regressing.

It grew out of a gap my own [Job Search Agent](../../Python%20Project%203/Job%20Search%20Agent/)
found: across job postings, the requirements it most often couldn't match to
evidence were *practical RAG with chunking, ranking and citation grounding*
and *an evaluation harness with regression suites and reproducible artifacts*.
This is the project that closes both.

## Stages

| Stage | What it adds | Status |
|---|---|---|
| [1](Stage%201/) | Pinned corpus, three chunking strategies, BM25 and TF-IDF retrievers | ✅ done |
| [2](Stage%202/) | Golden set of 98 questions labelled by section; recall@k, MRR, nDCG, answer coverage, each with 95% intervals; paired tests; versioned run manifests | ✅ done |
| [3](Stage%203/) | Dense embeddings, hybrid fusion, reranking — each measured against Stage 2 under a [pre-registered plan](Stage%203/PLAN.md). Supported: the reranked hybrid beats BM25 on MRR@10 (+0.09; its blind-round confirmation is fragile per page). Exploratory: the hybrid alone scored 0.72 to the reranked hybrid's 0.71 (reranking it changed MRR@10 by −0.014, 95% CI −0.073 to +0.048), at a median 6 ms a question instead of 758 ms on one Apple M5 Max CPU, one text at a time; whether the hybrid alone is enough needs new questions | ✅ done |
| 4 | Answers with enforced citations; LLM-as-judge faithfulness; model cost/quality trade-offs | |
| 5 | GitHub Actions regression gate: retrieval evals on every PR, LLM-judged evals nightly | |
| 6 | Live demo: ask the docs, with clickable citations and an experiments leaderboard | |

## Layout

```
RAG Eval Lab/
  corpus.lock.json   the pinned corpus: FastAPI tag, commit, SHA-256 per file (kept in git)
  models.lock.json   the pinned models: Hugging Face repo, revision, SHA-256 and size per file (kept in git)
  golden/            the questions every stage is scored against, with graded section labels
  data/              the fetched docs, models and caches (git-ignored; rebuilt from the locks)
  .venv/             one environment shared by every stage (git-ignored)
  Stage N/           a self-contained snapshot of the project at that stage
```

Each stage folder carries its own copy of the `raglab` package and runs from
its own directory (`python -m raglab ...`), so earlier stages stay runnable
exactly as they were while one shared environment holds the heavy
dependencies.

## The corpus

The FastAPI documentation, English edition, pinned at release `0.165.0`.
FastAPI and its documentation are © Sebastián Ramírez, used under the MIT
License. The docs are fetched at build time and never committed to this
repository as such; the annotations (the golden set's and Stage 4's) and
Stage 4's answers quote short passages from them, under that license and with
this attribution.
