# Stage 3 analysis plan

This plan was written, reviewed and committed before any Stage 3 metric was
read. It is frozen at **freeze point 1**: the commit the developer pushes and
tags `stage3-plan`, before the judging round is drawn. When the round is
drawn, `golden/v2-provenance.json` records this file's SHA-256. A test fails
if the file changes after that, and every run manifest records the hash too.
A draft was committed earlier (4ed27aff, below) and revised after a final
review.

Git shows the order of commits, not that nothing ran before them. So the next
section is the record of what had already run.

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

All on 2026-11-20, between about 16:15 and 17:25 EST, while the code was
built and timed:

- **The code was already in git.** Commit 4bbe4caa, made at 16:47 EST and
  pushed straight after, holds an early version of the code. It already fixed:
  - the query instruction, k = 60, rerank depth 100 and 4 threads;
  - the 30-cell grid.

  The plan keeps all of these. Since then the code has gained:
  - the bm25+rerank control and four variants;
  - tied scores sharing a rank before fusion;
  - dense ranking every chunk, and both rankings fused in full (4bbe4caa ranked
    and fused only chunks with a positive score);
  - a seeded tie rule;
  - the answers-only nDCG.
- **A draft of this plan was already in git.** Commit 4ed27aff, made and
  pushed at 17:51 EST, holds the reviewed draft with nearly all of today's
  code. `git diff 4ed27aff` at freeze point 1 shows every change after it.
- **The retrievers ran on all 98 golden questions.**
  - Every question was embedded both with and without the query instruction.
  - The cross-encoder re-scored each question's BM25 top 100 and hybrid top
    100 in every chunker and breadcrumb setting, so bm25+rerank could have been
    computed in all six settings, not only the control's.
  - A rerank depth set in characters (at least 100 chunks, and enough to fill
    9.6k) was implemented and run deeper into the paragraph rankings, then
    replaced by a fixed 100 on the reviewers' advice.

  Rankings were compared with the labels only to count which sections had no
  grade:
  - about 2,100 in some configuration's top 10;
  - about 50 more inside heading and fixed 4.8k windows;
  - about 4,900 more inside paragraph windows.
- **Metrics were computed in memory but never read.**
  - At about 16:35 the inherited Stage 2 tests scored all 30 cells against
    v1's labels, on the real models; the test suite's stand-in models didn't
    exist yet. The byte-for-byte test wrote two run folders into pytest's
    temporary directory.
  - pytest's report was deleted unread. The folders were never opened, and
    pytest has since removed them (it keeps only its three most recent).
  - Every count after that went through `run_configs`, which until this plan
    also computed every metric in memory.

  No Stage 3 metric value was printed or read. Since this plan, labels are
  drawn from runs that rank without reading a grade (`scored=False`).
- **Measurements on chunk texts and made-up questions:** speed, determinism
  and truncation.
- **Review.** Four adversarial reviewers and a critic reviewed the first draft;
  their lenses were method, validity, engineering and portfolio value. Two
  checkers reviewed the final draft. They read Stage 2's committed results and
  v1's labels. None computed a Stage 3 metric.

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
| hybrid | score = Σ 1 / (60 + rank) over BM25's and dense's rankings. Tied scores share the best rank before fusion. BM25 lists only chunks it scores above 0, so a chunk it scores 0 gets no BM25 term. Both lists are fused in full. |
| rerank | the first stage's top 100 are re-scored on (question, chunk text as indexed). Ties keep first-stage order. Ranks 101 onwards keep first-stage order. |
| ties | exact ties in dense or fused scores are ordered by sha256 of `20261120:<question>:<chunk id>`. This is arbitrary but fixed. Corpus order would favour some docs areas, because pages sort alphabetically. Stage 2's lexical retrievers keep their own corpus-order ties, unchanged. |
| numerics | CPU, float32, 4 threads, batch of one, inputs truncated at 512 tokens. A long input loses the end of its passage; no question (32 words at most) is long enough to be cut. |
| models | pinned in `models.lock.json` by full revision and per-file SHA-256 and size. Safetensors weights only, no remote code, loaded offline from a verified folder. |

None of these values was tuned on the golden questions. Their sources:
- the instruction comes from the model card;
- k = 60 comes from Cormack, Clarke & Büttcher (2009);
- depth 100 is the common deployment choice.

No alternative value of any of them was ever scored against the golden labels
for reading.

Scoring, budgets, metrics and intervals are Stage 2's, with one metric added.
**nDCG@10 (answers only)** credits only sections graded 2. Graded nDCG@10 also
credits sections judged merely related, and that is where judges agree least:
reweighted to the whole label set, κ is 0.69 on related-or-answer versus not,
against 0.92 on answer versus not. The paired tests are Stage 2's too, plus
per-page versions and a style-shuffle test, described below.

## Labels: golden/v2.yaml

v1 stays frozen, so Stage 2's results stay reproducible from it. v2 is v1 plus
**one judging round**, built by `analysis/v2_labels.py`. Its `verify`
recomputes every draw (the pool, the calibration items, the blind union, the
agreement sample, the bridge, the quote judge's items) and checks the
committed files against them. Every ranking it uses is computed without
reading a grade.

**The pool:**
- every section any of the 35 configurations ranks in its top 10 that v1 left
  unjudged;
- every unjudged section inside the 4.8k window of any heading or fixed
  configuration. Those windows hold about 6–8 sections, mostly inside the top
  10 already, so they add only about 50 sections.

The windows give no margin against float differences on other hardware: a
near-tie at rank 10 can still bring in an unjudged section, and the
perturbation check below measures how often that happens. Paragraph windows
are not judged in full. Their budget metrics are exploratory and are shown
with their judged share.

**The judges** use `golden/prompts/v2-pool-judge.md`:
- It is v1's prompt, word for word, plus one sentence: "Open no file other
  than the input file."
- Model: claude-opus-5-5. v1's files name only "Claude", so the match with
  v1's judges rests on the judging sessions' records, not on the repository;
  `v2-provenance.json` records the model for v2.
- References: the full text of each question's v1 answers, except answers
  that are hidden calibration items, which are withheld from their own
  question. No grade from this round is shown to any judge.
- Batches hold whole questions in id order, closing at 60 or more items.
- Each worksheet sits alone in its own folder outside the project. Judges
  never see which configuration found a section, how many did, or any ranking,
  result or plan.

**200 hidden calibration items** are v1 labels mixed into the same batches
and presented exactly like new candidates. Three strata, drawn by a committed
sha256 rule:
- 50 graded 1 and 50 graded 0, which a lenient round could promote. They are
  split by grade because promotions to an answer come mostly from grade 1.
- 100 non-target answers, which a strict round could demote.

Each stratum is split evenly. Half are sections heading+bm25 ranks in its top
10 (every label on that side of the headline comes from v1), and half are not.
Within each half, sections off the question's target page come first, because
that is where new candidates come from, then the rest in sha256 order. A half
that runs short is filled from the other.

One tell remains. v1 judged every section on each target page, so no new
candidate is on its question's target page, and about 45 of the 100 answers
are (only 55 non-target answers lie elsewhere).

Their new grades never enter v2.yaml. Reported, each with a Wilson interval,
overall and for each half:
- promotions: how many of the 100 graded 0 or 1 this round grades 2;
- demotions: how many of the 100 answers it grades 0 or 1, also for the 55
  off-page answers alone.

Drift never changes a label. The blind round is what puts both sides of the
headline under one condition.

**A second, blind round** uses the same model and
`golden/prompts/v2-blind-judge.md` (v1's blind prompt plus the same
sentence). The judge sees the question and the section only. It covers:
- every section either headline configuration ranks in its top 10, about
  1,500 (the union). These are all the labels link 1 reads, judged in one
  round under one condition.
- an agreement sample of v2's other new labels: all 2s up to 100, then 60 1s
  and 60 0s. Each stratum takes paraphrase-question items first, up to 40% of
  its quota (40, 24 and 24) where that many exist, then fills up in sha256
  order.
- 40 bridge items from v1's agreement study, drawn from those outside the
  union, to show whether the blind judge itself has moved.

Blind batches hold 60 items each in sha256 order, mixing questions. Each
worksheet sits alone in its own folder outside the project.

**A quote-checking judge** uses the same model and
`golden/prompts/v2-quote-judge.md`. It sees the agreement sample, the bridge,
and every section of the union newly graded 2 in this round. It must quote the
sentence or line that answers, and a quote counts only if code finds it in the
section. That catches a judge answering from its own knowledge of FastAPI.
Reported, for each first grade: the share of items it calls answers with a
quote that holds.

**Rules for labels:**
- No label changes because of the calibration, blind or quote studies.
- In any round, only a whole batch that fails validation (a missing,
  duplicate, unexpected or invalid answer) is judged again. The code refuses to
  re-judge a batch that passed, and logs every re-judge with the first run's
  output.
- The v1-unjudged-sample pairs that enter the pool are judged again, not
  reused, so all of v2's new labels come from one condition.

## The confirmatory tests

The primary metric is **MRR@10**. A question scores 1 if its first answering
section is ranked first, 1/2 if second, and so on down to 1/10, and 0 if no
answer is in the top 10.
- It depends only on whether top-10 sections are answers, the grade judges
  agree on most.
- Unlike recall, it doesn't change when more answers turn up elsewhere.
- It rewards moving an answer from rank 3 to rank 1, which a yes/no metric
  such as Success@5 can't see.

**The test:** two-sided paired sign-flip test, 10,000 resamples, α = 0.05.
The sign flips use seed 20261121 and the bootstrap intervals seed 20261120
(`seed + 1` and `seed` in the code). The style-shuffle test uses 20261122.

**The chain** runs on heading chunks with the breadcrumb, tested in this
order. Each link is tested only if every earlier link had p < 0.05 per
question, in either direction. That keeps the chance of any false claim among
the five links at 5%.

| link | comparison (b against a) | prediction |
|---|---|---|
| 1 (the headline) | hybrid+rerank against bm25 | higher |
| 2 | hybrid against bm25 | higher |
| 3 | dense against bm25 | none: two-sided, read as "higher" or "lower" |
| 4 | hybrid+rerank against hybrid | higher (the reranker adds) |
| 5 | hybrid+rerank against bm25+rerank | higher (dense candidates add, even once reranked) |

**Supporting metrics** apply to the headline pair. They are Holm-adjusted
together and claimed only if link 1 is significant. They are a second family
with its own 5%, so across both families the chance of a false claim can reach
about 10%. In order:
1. Success@5, with the questions fixed and broken beside it;
2. nDCG@10 (answers only);
3. graded nDCG@10;
4. Recall@5;
5. mean coverage in 4.8k;
6. Recall in 4.8k.

An nDCG gain is called robust only if at least one nDCG version is Supported
and the other version's estimate is also above zero. If the two estimates
differ in sign, neither nDCG result is described as a gain.

**Decision rules:**
- **Supported:** p < 0.05 (Holm-adjusted for the supporting metrics) in the
  predicted direction.
- **Contradicted**, for comparisons predicted "higher": p < 0.05 in the other
  direction, or a per-question 95% bootstrap interval of the difference
  entirely below zero. That interval is uncorrected, for the supporting
  metrics too.
- **Inconclusive:** anything else. It is reported with its 95% interval and
  with `resolvable`: the smallest difference detectable at 80% power, given
  that comparison's own paired standard deviation.

**What the design can resolve.** Before any data, from the paired standard
deviations of Stage 2's heading+tfidf against heading+bm25 (a retriever swap
on the same chunks), at 80% power with 98 questions:

| metric | smallest detectable difference |
|---|---|
| MRR@10 | about 0.08 (0.07–0.12 across heading+bm25's eleven Stage 2 comparisons) |
| Success@5 | about 0.10 (0.06–0.15) |
| nDCG@10 (answers only) | about 0.06 |
| Recall@5 and the budget metrics | about 0.08 |

Swaps that change rankings more vary more, so for lexical against neural
these figures are probably optimistic.

**Page-level check.** 27 pages hold two of the 98 targets. So every
confirmatory test is also run with signs flipped, and the bootstrap
resampled, per target page. A test with p < 0.05 per question and p ≥ 0.05 per
page is reported as fragile. For the supporting metrics, both p-values are
Holm-adjusted before this comparison. Fragility is stated beside a verdict and
never changes it.

**Label check.** The confirmatory tests are scored on `golden/v2.yaml`. Link 1
is also tested the same way, per question and per page, on the blind labels:
every section in either headline configuration's top 10 takes its blind grade,
targets included. Links 2–5 are not, because the blind round doesn't cover
their configurations' top 10s.

| link 1 on v2's labels | link 1 on the blind labels | verdict |
|---|---|---|
| Supported | Supported | Supported |
| Supported | not Supported | "supported on v2's labels, not confirmed by the blind round" |
| not Supported | either | v2's verdict |

The agreement figure is Cohen's κ on answer versus not (grade 2 against 0 or
1), unweighted, between the blind grade and the v2 grade, over every union
label new in this round. If κ is below 0.70, the README's first statement of
the headline gives link 1's MRR@10 difference and 95% interval on both label
sets side by side, instead of one number.

## Exploratory, reported with intervals and no claims

Bold in an exploratory table marks p < 0.05 after Holm within that table, or
an interval that excludes zero. It describes; it is not a claim.

- **By style.**
  - Link 1's MRR@10 difference within each question style (keyword,
    paraphrase, scenario), each with its interval.
  - The style interaction on nDCG@10 (answers only): heading+dense minus
    heading+bm25 (breadcrumb on) on paraphrase questions, minus the same on
    keyword questions. Its interval comes from a bootstrap within each style,
    and its p-value from shuffling style labels. Smallest detectable effect
    about 0.12. The code also computes it on Recall@5; that version is
    secondary and labelled as such.

  The paraphrases were written to avoid their section's wording, so a larger
  dense gain on them is expected by design. It checks that the question styles
  work as intended; it is not evidence that dense retrieval is better.
- **Settings and switches.**
  - Every other chunker and breadcrumb setting.
  - Each switch's average effect. Retriever switches are averaged over the six
    chunker and breadcrumb settings. The two involving bm25+rerank exist only
    on heading chunks with the breadcrumb, so each is a single pair. Chunker
    and breadcrumb switches are read within each retriever family.
  - The 29 other factorial cells against Stage 1's baseline. The five extras
    are scored but kept out of this Holm family. Holm over 29 comparisons
    instead of Stage 2's 11 makes adjusted p-values larger, so some differences
    bold in Stage 2's README may not be bold here.
- **The extras and two more pairs.**
  - The rerank-depth curve beside its latency.
  - The instruction's effect.
  - MRR@10 for heading+bm25+rerank against heading+bm25 (what the reranker adds
    to BM25), and for heading+hybrid against heading+dense, each with a p-value
    and no claim.
  - Heading+hybrid's MRR@10 difference from each of its inputs, called "within
    0.05" only if its whole 95% interval lies inside ±0.05. Link 2 is the test.
- **The pool.**
  - The new configurations scored on v1 (unjudged counts as not relevant)
    against v2: the penalty a stale pool puts on a new retriever.
  - Stage 2's 12 configurations on v1 against v2. Their Success@5 and MRR@10
    can't change, by construction. Recall, nDCG and coverage can, as answer
    sets grow.
  - **Leaving one family out.** For each new retriever family (dense, hybrid,
    hybrid+rerank, bm25+rerank), drop the new v2 judgments that only that
    family's top 10s or windows surfaced, per `v2-provenance.json`, and
    re-score. Reported: each configuration's change in MRR@10, and Kendall's τ
    between the 35 configurations' MRR@10 orderings with and without those
    judgments.
- **Answers Stage 2 couldn't reach.** Of the 60 known answers no lexical
  configuration ranks in its top 10 (all on target pages; 16 are the
  questions' own targets), how many each new configuration ranks in its top 5
  and top 10. The same count is given for the 8 questions with no answer in
  any lexical top 10.
- **Bias checks.**
  - Recall@10 over target-page answers other than the target (116 answers on
    50 questions). Every one was judged, so the pool can't bias it.
  - **Source left out.** On the 64 questions with an answer besides the
    target, the target is removed from the answers and from every ranking, and
    the sections below it each move up one place. Reported: link 1's MRR@10
    difference on those questions, with its interval.
- **Diagnostics.**
  - The judged share at every budget.
  - The share of top-10 slots graded 1.
  - The share of each budget window's chunks that lie below the rerank depth
    (100, or 10, 20 and 50 for the depth variants) and so were never
    re-scored.
  - Exact ties and near-ties (under 1e-6).
- **Perturbation check.** Every dense similarity and cross-encoder score moves
  by one float32 unit in the last place, up or down by a coin flip seeded with
  20261120, in 20 runs. Reported: how often each chain configuration's top-10
  sections change, and the largest change in link 1's MRR@10 difference.

## Cost, measured outside the reproducible results

`raglab bench`, written after freeze point 1 and reading no labels, writes
`results/cost.json`. Nothing in the evaluation reads it.
- **Machine:** the chip, core count, RAM, thread count (4) and power source
  are recorded.
- **Caches off.**
- **Cold start:** import, load and first question, timed in five fresh
  processes.
- **Warm timing:**
  - per question, after 10 warm-up questions, over 5 passes of the 98
    questions, with the retrievers' order shuffled by seed 20261120;
  - p50 and p95 with their n;
  - each step timed on its own: question embedding, dense scoring, BM25,
    fusion, reranking.
- **Reranking** timed both at batch 1, as evaluated, and with each question's
  100 pairs in one batch, as a deployment would run it.
- **Sizes:** CPU-seconds per 1,000 questions, exact array sizes, model file
  sizes.

Made-up questions are timed until freeze point 1 is pushed, then the 98.

## Rules

- **Fixed settings.** The settings, metrics, comparisons and decision rules
  above don't change after freeze point 1. Anything added later is labelled as
  added later. A deviations log in the README lists every change to this plan
  or to the code after freeze point 1, bug fixes included, with its date, its
  reason, and its effect on link 1's MRR@10 difference and verdict.
- **Frozen questions.** Every confirmatory analysis uses exactly the 98
  fingerprinted questions (`c4673ddd…`). None is added, removed or reweighted
  after a Stage 3 metric is read. Re-scores that exclude questions are
  sensitivity analyses only. Any new questions form a separate set and never
  enter the confirmatory tests.
- **A loss is a result.** If hybrid+rerank, or dense alone, does worse than
  BM25, that is the finding.
- **Stage 4's retriever.** Stage 4 inherits heading+hybrid+rerank (breadcrumb
  on) only if link 1's verdict under the label check is Supported, and
  heading+bm25 otherwise.
  - A fragile Supported still counts, and the README calls it fragile
    wherever it states it.
  - Cost doesn't enter this rule; it is reported beside it.
  - No other configuration is carried forward on the strength of these tables,
    which are ordered by design, never by score.

## Freeze points, each committed and pushed by the developer

1. **This plan.** The plan; the code for the evaluation, both label rounds,
   the confirmatory tests, the label check and the README's confirmatory
   table; the prompts; `models.lock.json` and `requirements-lock.txt`
   (`corpus.lock.json` is unchanged since Stage 1). All are committed and
   pushed before the judging round is drawn.

   The exploratory analyses and `raglab bench` are written after it, from this
   plan's definitions, and committed before freeze point 2.
2. **The labels.** `v2.yaml`, every judge output (`v2-judgments.json`,
   `v2-calibration.json`, `v2-blind.json`, `v2-quote.json`) and
   `v2-provenance.json`, committed and pushed before any Stage 3 configuration
   is scored against v2's labels.

   Until then, the tests that score Stage 3 configurations are skipped (they
   wait for `v2-blind.json`), and `analysis/v2_labels.py` computes rankings
   only, never metric values.
3. **The results.**

## Reproducibility

Byte-identical re-runs are promised for the recorded environment only:
- `requirements-lock.txt`, which pins every wheel by hash;
- 4 threads;
- the chip and OS in the manifest.

**Every manifest records:**
- this plan's SHA-256;
- the SHA-256 of `models.lock.json` and of `requirements-lock.txt`;
- the corpus fingerprint from `corpus.lock.json`;
- every locked library's version;
- a hash of the prompts.

**Tests that check reproduction:**
- A seeded sample of cached vectors and scores is recomputed from the models
  and must be bit-identical, so the cache is checked, not trusted.
- A full run from an empty cache, compared byte for byte with a run on the
  filled cache (about 15 minutes, `RAGLAB_COLD=1`), is run once and recorded
  in the README.

On other hardware the last float bits can differ; the perturbation check
measures how much that matters.

## Not in this plan

- A larger or newer embedding model. The two small models were chosen for
  this stage; a larger one is a later stage.
- A human check of the judges' labels.
- Questions written by the developer without the docs open.

Neither of the last two is part of this plan. Either would be a separate,
later study, reported as such.

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
  "vs_baseline_comparisons": 29,
  "extras": ["heading+bm25+rerank", "heading+dense [no instruction]", "heading+hybrid+rerank [depth 10]",
             "heading+hybrid+rerank [depth 20]", "heading+hybrid+rerank [depth 50]"],
  "primary_metric": "mrr@10",
  "alpha": 0.05,
  "chain": [["heading+bm25", "heading+hybrid+rerank", 1], ["heading+bm25", "heading+hybrid", 1],
            ["heading+bm25", "heading+dense", 0], ["heading+hybrid", "heading+hybrid+rerank", 1],
            ["heading+bm25+rerank", "heading+hybrid+rerank", 1]],
  "supporting_metrics": ["success@5", "ndcg_answers@10", "ndcg@10", "recall@5", "coverage@4800", "budget_recall@4800"],
  "interaction_metric": "ndcg_answers@10",
  "label_check_kappa": 0.70,
  "budgets": [1200, 2400, 4800, 9600],
  "headline_budget": 4800,
  "resamples": 10000,
  "seed": 20261120,
  "sign_flip_seed": 20261121,
  "style_shuffle_seed": 20261122,
  "pool": {"depth": 10, "window_budget": 4800, "window_chunkers": ["heading", "fixed"]},
  "calibration": {"related": 50, "rejected": 50, "answers": 100},
  "agreement": {"2": 100, "1": 60, "0": 60},
  "paraphrase_share": 0.4,
  "bridge": 40,
  "batch_items": 60,
  "judge_model": "claude-opus-5-5"
}
```
