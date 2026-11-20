# Stage 3 analysis plan

Written, reviewed and committed before any Stage 3 metric was read. The next
section says exactly what had already run when it was written. Git shows the
order of commits, not that nothing ran before them, so that list is the record.

## The question

On 98 questions an LLM wrote from sampled FastAPI doc sections (a third of
them phrased to avoid the section's own wording), do these three rank the
sections that answer a question higher than BM25 does, and at what cost?

- bge-small-en-v1.5 embeddings (33M parameters);
- reciprocal rank fusion of those embeddings with BM25;
- the ms-marco-MiniLM-L6-v2 cross-encoder (22M parameters) re-scoring the top 100.

This is a test of these two small models on this question set. It is not a
verdict on dense retrieval or reranking in general.

## What ran before this plan

All on 2026-11-20, while the code was being built and timed:

- **The code was already in git.** Commit 4bbe4caa, pushed at 16:47 EST,
  holds an early version of the code. It already fixed:
  - the query instruction, k = 60, rerank depth 100 and 4 threads;
  - the 30-cell grid.

  The plan keeps all of these. It adds what the review below asked for:
  - the bm25+rerank control and four variants;
  - competition-ranked fusion inputs;
  - a seeded tie rule;
  - the answers-only nDCG.
- **The retrievers ran on all 98 golden questions.** Every question was
  embedded, and each question's top 100 was re-scored by the cross-encoder in
  every chunker and breadcrumb setting. The caches were filled between 16:30
  and 17:25.

  Rankings were compared with the labels only to count which sections had no
  grade:
  - about 2,100 in some configuration's top 10;
  - about 50 more inside heading and fixed 4.8k windows;
  - about 4,900 more inside paragraph windows.

  The inherited Stage 2 tests computed metric values in memory once, at about
  16:35. Their output file was deleted unread. No Stage 3 metric value was
  printed or read.
- **Measurements on chunk texts and made-up questions:** speed, determinism
  and truncation.
- **The review.** Four adversarial reviewers and a critic read this plan's
  draft, Stage 2's committed results and v1's labels. The reviewers covered
  method, validity, engineering and portfolio value. None of them computed a
  Stage 3 metric.

The author of this plan had seen the questions, v1's labels and Stage 2's
per-question results. The predictions build on them.

## Configurations: 35, all judged to the same depth

**The factorial: 30 cells.** 3 chunkers × 5 retrievers × breadcrumb on or off.
The retrievers are tfidf, bm25, dense, hybrid and hybrid+rerank. Stage 2's 12
configurations are the tfidf and bm25 cells.

**Five extras**, scored and pooled with the factorial, all on heading chunks
with the breadcrumb:

| extra | why |
|---|---|
| bm25+rerank | the control: the same cross-encoder over BM25's top 100. Was the vector index needed at all? |
| dense [no instruction] | the query instruction is a documented option; this shows its effect, whatever it is |
| hybrid+rerank [depth 10], [depth 20], [depth 50] | the rerank-depth curve; depth 100 is the factorial's own cell |

**Fixed settings:**

| | setting |
|---|---|
| dense | bge-small-en-v1.5 at revision `5c38ec7c…`, cosine similarity. The question gets the model card's instruction `Represent this sentence for searching relevant passages: `; chunks get none. Every chunk is ranked. |
| hybrid | score = Σ 1 / (60 + rank) over BM25's and dense's rankings. Tied input scores share the best rank (competition ranking). BM25 lists only chunks it scores above 0, so a chunk it scores 0 gets no BM25 term. Both lists are fused in full. |
| rerank | the first stage's top 100 are re-scored on (question, chunk text as indexed). Ties keep first-stage order. Ranks 101 onwards keep first-stage order. |
| ties | exact ties in dense or fused scores are ordered by sha256 of `20261120:<question>:<chunk id>`. This is arbitrary but fixed. Corpus order would favour some docs areas, because pages sort alphabetically. Stage 2's lexical retrievers keep their own corpus-order ties, unchanged. |
| numerics | CPU, float32, 4 threads, batch of one, inputs truncated at 512 tokens (the passage's end, never the question). |
| models | pinned in `models.lock.json` by full revision and per-file SHA-256. Safetensors weights only, no remote code, loaded offline from a verified folder. |

None of these values was tuned on the golden questions. Their sources:
- the instruction comes from the model card;
- k = 60 comes from Cormack, Clarke & Büttcher (2009);
- depth 100 is the common deployment choice.

No alternative value of any of them was tried on the golden questions.

Scoring, budgets, metrics, intervals and tests are Stage 2's, with one metric
added. **nDCG@10 (answers only)** credits only sections graded 2. Graded
nDCG@10 also credits sections judged merely related, and that is where judges
agree least (κ 0.69 against 0.92 on answer versus not).

## Labels: golden/v2.yaml

v1 stays frozen, so Stage 2's results stay reproducible from it. v2 is v1 plus
**one judging round**, built by `analysis/v2_labels.py`, which `verify` checks.

**The pool:**
- every section any of the 35 configurations ranks in its top 10 that v1 left
  unjudged;
- every unjudged section inside the 4.8k window of any heading or fixed
  configuration. Those windows hold about 6–8 sections, so judging them is
  cheap. It also gives the depth-10 gate a margin against last-bit float
  differences on other machines.

Paragraph windows are not judged in full. Their budget metrics are
exploratory and are shown with their judged share.

**The judges** use `golden/prompts/v2-pool-judge.md`:
- It is v1's prompt, word for word, plus one sentence: "Open no file other
  than the input file."
- Model: claude-opus-5-5, the model v1's judges ran as.
- References: each question's v1 answers in full. Frozen: nothing judged in
  this round is shown to another judge.
- Batches hold whole questions in id order, closing at 60 or more items.
- Each worksheet sits alone in its own folder outside the project. Judges
  never see which configuration found a section, how many did, or any ranking,
  result or plan.

**200 hidden calibration items** are v1 labels mixed into the same batches,
indistinguishable from new candidates. They are drawn by a committed sha256
rule in two strata:
- 100 graded 0 or 1, which a lenient round could promote;
- 100 non-target answers, which a strict round could demote. Each is left out
  of its own question's answer list.

Each stratum is half from heading+bm25's top 10 (the side of the headline
whose labels all come from v1) and half not, off-target-page sections first.
Their new grades never enter v2.yaml. They measure drift between the rounds,
reported as promotions against demotions on the answer line.

**A second, blind round** uses the same model and
`golden/prompts/v2-blind-judge.md` (v1's blind prompt plus the same
sentence). The judge sees the question and the section only. It covers:
- every section either headline configuration ranks in its top 10, about
  1,500. These are all the labels the primary metric reads, judged in one run
  under one condition. The headline is reported on these labels too.
- an agreement sample of v2's other new labels: all 2s up to 100, 60 1s and
  60 0s. At least 40% of each stratum comes from paraphrase questions.
- 40 bridge items from v1's agreement study, to show whether the blind judge
  itself has moved.

**A decorrelated judge** (`golden/prompts/v2-quote-judge.md`) sees the
agreement sample and the bridge items. It must quote the sentence or line that
answers, and a quote counts only if code finds it in the section. That
catches a judge answering from its own knowledge of FastAPI.

**Rules for labels:**
- No label changes because of the calibration, blind or quote studies.
- Only a whole batch that fails validation (a missing, duplicate or invalid
  grade) is judged again, and every such re-judge is logged.
- The v1-unjudged-sample pairs that enter the pool are judged again, not
  reused, so all of v2's new labels come from one condition.

## The confirmatory tests

The primary metric is **MRR@10**: two-sided paired sign-flip test, 10,000
resamples, seed 20261120, α = 0.05.

MRR@10 reads only judged top-10 answer labels, the line judges agree on most.
Its denominator doesn't grow with the pool. It also registers a reranker's
reordering, which a yes/no metric can't.

**The chain** runs on heading chunks with the breadcrumb, tested in this
order. Each link counts only if every earlier link was significant, which
keeps the chance of any false claim at 5%.

| link | comparison (b against a) | prediction |
|---|---|---|
| 1 (the headline) | hybrid+rerank against bm25 | higher |
| 2 | hybrid against bm25 | higher |
| 3 | dense against bm25 | none: two-sided, read as "higher" or "lower" |
| 4 | hybrid+rerank against hybrid | higher (the reranker adds) |
| 5 | hybrid+rerank against bm25+rerank | higher (dense candidates add, even once reranked) |

**Supporting metrics** apply to the headline pair. They are Holm-adjusted
together, and claimed only if link 1 is significant:
1. Success@5, with the questions fixed and broken beside it;
2. nDCG@10 (answers only);
3. graded nDCG@10;
4. Recall@5;
5. mean coverage in 4.8k;
6. Recall in 4.8k.

An nDCG gain counts as robust only if both nDCG versions agree in sign.

**Decision rules:**
- **Supported:** significant, in the predicted direction.
- **Contradicted:** significant in the other direction, or a 95% interval
  entirely below zero.
- **Inconclusive:** anything else. It is reported beside the smallest
  difference this design can resolve.

**Resolution.** At 80% power with 98 questions, computed from Stage 2's paired
standard deviations:

| metric | smallest detectable difference |
|---|---|
| MRR@10 | about 0.08, or 0.08–0.10 allowing for questions that share a page |
| Success@5 | about 0.10 |
| nDCG@10 (answers only) | about 0.06 |
| Recall@5 and the budget metrics | about 0.08 |

**Page-level check.** 27 pages hold two of the 98 targets. So every
confirmatory test is also run with signs flipped, and the bootstrap
resampled, per target page. A result that holds per question but not per page
is reported as fragile.

**Label check.** The chain is also reported on the blind round's labels. If
the blind judge's answer-versus-not agreement with the new labels falls below
κ 0.70, the README leads with the headline's range across the two label sets,
not a single number.

## Exploratory, reported with intervals and no claims

- **By style.**
  - The headline per question style, as three paired differences, scenario
    first.
  - The style interaction on nDCG@10 (answers only): dense minus BM25 on
    paraphrase questions, minus the same on keyword questions. Its interval
    comes from a bootstrap within each style, and its p-value from shuffling
    style labels. Smallest detectable effect about 0.12.

  The paraphrases were written to avoid their section's wording, so this
  interaction is a manipulation check, expected by design.
- **Settings and switches.**
  - Every other chunker and breadcrumb setting.
  - Each switch's average effect: retriever switches over all six settings,
    chunker and breadcrumb switches within each retriever family.
  - All 29 comparisons with Stage 1's baseline. Holm over 29 shifts Stage 2's
    adjusted p-values, so those marks differ from Stage 2's README.
- **The extras.**
  - The rerank-depth curve beside its latency.
  - The instruction's effect.
  - Whether the hybrid is within 0.05 MRR@10 of each input (descriptive).
- **The pool.**
  - The new configurations scored on v1 (unjudged counts as not relevant)
    against v2: the penalty a stale pool puts on a new retriever.
  - Stage 2's 12 configurations on v1 against v2. Their Success@5 and MRR@10
    can't change, by construction. Recall, nDCG and coverage can, as answer
    sets grow.
  - Leaving one retriever family's judgments out, with the change in ranking
    order (Kendall's τ).
- **Fixed denominators for Stage 2's second target.** Of the 60 known answers
  no lexical configuration ranks in its top 10, how many each new
  configuration reaches in its top 5 and top 10. All 60 are on target pages,
  and 16 are targets. The same count is reported for the 8 questions with no
  answer in any lexical top 10.
- **Bias checks.**
  - Recall@10 over target-page answers other than the target (116 answers on
    50 questions). Every one was judged, so the pool can't bias it.
  - A source-left-out check: on questions with an answer besides the target,
    the target is removed from the answers and every ranking.
- **Diagnostics.**
  - The judged share at every budget.
  - The share of top-10 slots graded 1.
  - The share of each reranked window below rank 100.
  - Exact ties and near-ties (under 1e-6).
  - A perturbation check of how far last-bit float differences travel.

## Cost, measured outside the reproducible results

`raglab bench` writes `results/cost.json`. Nothing in the evaluation reads it.
- **Machine:** the chip, core count, RAM, thread count (4) and power source
  are recorded.
- **Caches off.**
- **Cold start:** import, load and first question, timed in fresh processes.
- **Warm timing:**
  - per question, after a warm-up, with retrievers interleaved in seeded
    order over several passes;
  - p50 and p95 with their n;
  - each step timed on its own: question embedding, dense scoring, BM25,
    fusion, reranking.
- **Reranking** timed both at batch 1, as evaluated, and batched, as a
  deployment would run it.
- **Sizes:** CPU-seconds per 1,000 questions, exact array sizes, model file
  sizes.

Made-up questions are timed until this plan's commit is pushed, then the 98.

## Rules

- **Fixed settings.** The settings, metrics, comparisons and decision rules
  above don't change after any result is seen. Anything added later is
  labelled as added later. A deviations log in the README lists every change
  after results, bug fixes included, with its reason and its effect on the
  primary metric.
- **Frozen questions.** Every confirmatory analysis uses exactly the 98
  fingerprinted questions (`c4673ddd…`). None is added, removed or reweighted
  after a Stage 3 metric is read. Re-scores that exclude questions are
  sensitivity analyses only. Any new questions form a separate set and never
  enter the confirmatory tests.
- **A loss is a result.** If dense retrieval loses to BM25, that is the
  finding.
- **Stage 4's retriever.** Stage 4 inherits heading+hybrid+rerank (breadcrumb
  on) if link 1 holds, and heading+bm25 otherwise. No other configuration is
  carried forward on the strength of these tables. The tables are ordered by
  design, not by score.

## Freeze points, each committed and pushed by the developer

1. **This plan.** The plan, the code that implements it, the prompts and both
   locks, before the judging round is drawn.
2. **The labels.** `v2.yaml`, every judge output (`v2-judgments.json`,
   `v2-calibration.json`, `v2-blind.json`, `v2-quote.json`) and
   `v2-provenance.json`, before any Stage 3 metric is computed for reading.
3. **The results.**

## Reproducibility

Byte-identical re-runs are promised for the recorded environment only. That
means:
- `requirements-lock.txt`;
- 4 threads;
- the chip and OS in the manifest.

Every manifest records this plan's SHA-256, both locks' hashes and the
prompts' hash. The reproduction test runs once from an empty cache, so it
checks the models, not just the cache. On other hardware the last float bits
can differ; the perturbation check measures how much that matters.

## Not in this plan

- A larger or newer embedding model. The two small models were chosen for
  this stage; a larger one is a later stage.
- A human check of the AI judges' labels.
- Questions written by the developer without the docs open.

Either of the last two can still be added before this plan is committed. If
so, it is listed here with its procedure.

## Parameters (machine-readable)

A test checks that the code uses exactly these values.

```json
{
  "questions_fingerprint": "c4673dddd9bdce5b73ae05f94672ad9e603c591d80b9469133ea442b821fcb4a",
  "embedder": {"repo": "BAAI/bge-small-en-v1.5", "revision": "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"},
  "reranker": {"repo": "cross-encoder/ms-marco-MiniLM-L6-v2", "revision": "233902d25c440f23af6f7d6e94d2946bac0bee0a"},
  "query_instruction": "Represent this sentence for searching relevant passages: ",
  "rrf_k": 60,
  "rerank_depth": 100,
  "tie_seed": "20261120",
  "threads": 4,
  "max_tokens": 512,
  "configurations": 35,
  "extras": ["heading+bm25+rerank", "heading+dense [no instruction]", "heading+hybrid+rerank [depth 10]",
             "heading+hybrid+rerank [depth 20]", "heading+hybrid+rerank [depth 50]"],
  "primary_metric": "mrr@10",
  "alpha": 0.05,
  "chain": [["heading+bm25", "heading+hybrid+rerank", 1], ["heading+bm25", "heading+hybrid", 1],
            ["heading+bm25", "heading+dense", 0], ["heading+hybrid", "heading+hybrid+rerank", 1],
            ["heading+bm25+rerank", "heading+hybrid+rerank", 1]],
  "supporting_metrics": ["success@5", "ndcg_answers@10", "ndcg@10", "recall@5", "coverage@4800", "budget_recall@4800"],
  "interaction_metric": "ndcg_answers@10",
  "budgets": [1200, 2400, 4800, 9600],
  "headline_budget": 4800,
  "resamples": 10000,
  "seed": 20261120,
  "pool": {"depth": 10, "window_budget": 4800, "window_chunkers": ["heading", "fixed"]},
  "calibration": {"promotable": 100, "answers": 100},
  "agreement": {"2": 100, "1": 60, "0": 60},
  "paraphrase_share": 0.4,
  "bridge": 40,
  "batch_items": 60,
  "judge_model": "claude-opus-5-5"
}
```
