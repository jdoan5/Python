# RAG Eval Lab — Stage 3: dense embeddings, hybrid fusion and reranking

**In progress.** Stage 3 measures three additions against Stage 2's lexical
retrievers:

- bge-small-en-v1.5 embeddings;
- reciprocal rank fusion of those embeddings with BM25;
- the ms-marco-MiniLM-L6-v2 cross-encoder re-scoring the top 100.

It runs on the same 98 questions, with a wider pool of judged sections.

How it is measured was fixed before any Stage 3 result was read. The
[analysis plan](PLAN.md) names:

- the one primary test, and the comparisons that follow it;
- the decision rules;
- exactly what had already run when the plan was written.

Results will appear here once the new labels (`golden/v2.yaml`) are judged and
committed.

## What's new in the code

- `raglab/models.py` pins both models in `../models.lock.json` by full
  revision and the SHA-256 and size of every file. `raglab models fetch`
  downloads exactly those files without credentials. Every load checks them
  first and reads them from a verified folder. Only safetensors weights are
  used, never remote code.
- `raglab/neural.py` holds the retrievers: dense, hybrid (reciprocal rank
  fusion) and reranking. Every text is embedded or scored on its own, so a
  number never depends on its batch, and results are cached per text. A model
  loads only when something isn't cached.
- `analysis/v2_labels.py` builds the new labels by rules that `verify` can
  check:
  - the pool;
  - hidden calibration items;
  - worksheets;
  - the blind re-judging round.

## Run it

From this folder, with the project's shared virtual environment:

```bash
../.venv/bin/pip install --only-binary :all: -r requirements-lock.txt
```

```bash
../.venv/bin/python -m raglab corpus fetch
```

```bash
../.venv/bin/python -m raglab models fetch
```

```bash
../.venv/bin/python -m pytest -q
```

Tests that need the real corpus or the real models skip when those aren't
downloaded. Set `RAGLAB_REQUIRE_CORPUS=1` and `RAGLAB_REQUIRE_MODELS=1` to make
their absence a failure instead.
