# RAG Eval Lab — Stage 3: dense embeddings, hybrid fusion and reranking

Stage 2 found where lexical retrieval breaks: on paraphrased questions, and on
the 22% of known answers that no configuration ranked in its top 10. Stage 3
adds three things, and measures each against BM25 on the same 98 questions:

- **dense retrieval** with bge-small-en-v1.5 (33M parameters);
- a **hybrid**: reciprocal rank fusion of those embeddings with BM25;
- **reranking**: the ms-marco-MiniLM-L6-v2 cross-encoder (22M parameters)
  re-scoring the first stage's top 100.

How it would be measured was fixed before any Stage 3 result was read. The
[analysis plan](PLAN.md) names one primary test, the order of the tests after
it, the decision rules, and what had already run when it was written. It was
pushed (freeze point 1) before any new label was drawn. The new labels were
pushed (freeze point 2) before any Stage 3 configuration was scored against
them, and the results were pushed at freeze point 3, commit `711f8b40`.

## What it found

The plan tests five comparisons in a fixed order, the **chain**, on MRR@10:
how high the first answering section ranks. A link is tested only if every
earlier link was significant, which holds the chance of any false claim among
the five at 5%. Everything marked exploratory below makes no claim, including
links 4 and 5, which the chain never reached.

1. **Supported: the reranked hybrid ranks answers higher than BM25 (its
   blind-round confirmation is fragile per page).** On
   heading chunks with the breadcrumb, MRR@10 rises from 0.62 to 0.71: +0.090
   (95% CI +0.020 to +0.159, p = 0.013). Treating the questions that share a
   target page as one unit gives p = 0.021. Five of the six supporting metrics
   agree after Holm correction:
   - nDCG@10 +0.07, and +0.09 counting answers only;
   - Recall@5 +0.06;
   - in a 4,800-character prompt, Recall +0.06 and mean coverage +0.05.

   Success@5 is inconclusive: +0.06, Holm p = 0.071. 7 questions gained an
   answer in the top 5 and 1 lost one.

   A blind judge re-graded every section either pipeline ranks in its top 10,
   seeing only the question and the section. On those grades link 1 is +0.063
   (+0.002 to +0.125, p = 0.044), so the verdict stands. That confirmation is
   fragile per page (p = 0.055), and this README says so wherever it states
   the verdict. By the plan's rule, Stage 4 inherits heading+hybrid+rerank.
2. **Supported: fusion alone beats BM25. Dense retrieval alone can't be told
   apart from it.**
   - Link 2, the hybrid against BM25: +0.104 (+0.052 to +0.157,
     p = 0.0004).
   - Link 3, dense against BM25: +0.036 (−0.041 to +0.115, p = 0.37),
     inconclusive. With this comparison's spread, it could reliably detect
     only a difference of about 0.11 or more.

   The chain stops at link 3, so links 4 and 5 are exploratory.
3. **Exploratory: reranking the hybrid added nothing measurable, and took
   14–129 times as long.**
   - The reranked hybrid against the hybrid alone: −0.014 (−0.073 to
     +0.048). Averaged over all six chunker and breadcrumb settings: −0.01
     (−0.04 to +0.03).
   - Reranking only the top 10, 20 or 50 puts the difference at +0.012,
     +0.004 and −0.006 (all p ≥ 0.69). These are paired tests from
     `analysis/extras.py`, added after freeze point 2.
   - Reranking BM25's top 100 gains about what fusion gains: +0.098 over BM25
     (+0.031 to +0.167).

   Here, once 100 candidates were reranked, the first stage barely changed
   the top of the ranking. hybrid+rerank and bm25+rerank put the same section
   first on 92 of the 98 questions and agree on Success@5 for all 98, though
   their top 10s differ on 43. Their MRR@10 differs by −0.008 (−0.028 to
   +0.006). Fusion and reranking each gained about 0.10 over BM25 (+0.104 and
   +0.098), and combining them added nothing measurable. Against the hybrid
   alone, though, the interval (−0.073 to +0.048) is too wide to call the two
   equal.

   The hybrid takes 5.9 ms per question and the reranked pipelines 82–758 ms
   (medians on one Apple M5 Max, 4 threads). The plan's rule ignores cost, and
   no other pipeline is carried forward on exploratory numbers. Whether the
   hybrid alone is enough is a question for new questions, not these.
4. **Exploratory: paraphrased questions are still the weak spot.** On the 32
   paraphrases:
   - Success@5 is 0.47 for BM25, 0.62 for dense and for the hybrid, and 0.53
     for the reranked hybrid;
   - MRR@10 is 0.32, 0.39, 0.45 and 0.42.

   On keyword questions BM25 and every new heading pipeline already have a
   Success@5 of 1.00. Link 1 within each style is +0.048 (−0.030 to +0.134)
   on the 33 keyword questions, +0.101 (−0.019 to +0.220) on the 32
   paraphrases and +0.121 (−0.025 to +0.270) on the 33 scenarios: every
   interval includes zero.

   The pre-specified style interaction is dense's nDCG@10 (answers only) gain
   on paraphrases minus its gain on keyword questions. It is +0.10 (−0.04 to
   +0.25, p = 0.18): in the expected direction, but below the 0.12 this design
   can detect. On Recall@5, the secondary version, it is +0.09 (−0.08 to
   +0.25, p = 0.31).

   Of the 60 known answers no lexical configuration had in its top 10,
   heading+dense ranks 22 in its top 10 (15 in its top 5), and the reranked
   hybrid 18 (9).
5. **Exploratory: with the model card's query instruction, dense retrieval's
   MRR@10 was lower than without it.** The difference is −0.041 (−0.079 to
   −0.006, p = 0.028), from a paired test in `analysis/extras.py`, added after
   freeze point 2. Success@5, Recall@5 and nDCG@10 (answers only) moved by
   +0.020, −0.012 and −0.019 (p = 0.62, 0.54 and 0.15). These p-values are
   uncorrected, from one of many exploratory comparisons. The instruction was
   fixed in advance from the model card and kept; nothing was tuned on the
   golden labels.
6. **Label checks, and exploratory checks on the pool: the new labels were
   consistent across rounds and judges, and the wider pool changed little.**
   - **Drift.** 200 hidden v1 labels went through the new judging round.
     None of the 100 non-answers became an answer, and 2 of the 100 answers
     lost their grade. That round's answer-versus-not agreement with v1 is
     κ = 0.98.
   - **Blind round.** It agreed on answer versus not with κ = 0.88 over the
     headline's new labels, above the plan's 0.70.
   - **Quote judge.** It had to quote the line that answers, and code looked
     for the quote in the section. For each of the 62 answers it saw,
     including all 49 of v2's new ones, it agreed and gave a quote the
     section contains.
   - **Stale pool (exploratory).** On v1's labels, 12–40% of the new
     configurations' top-10 slots held a section v1 never judged (14–41% of
     the sections they ranked). Even so, their MRR@10 there was never higher
     than on v2, and at most 0.013 lower.
   - **Leaving one retriever family out (exploratory).** Dropping the
     judgments only one family surfaced leaves the 35 configurations' MRR@10
     order nearly unchanged: Kendall's τ is 0.99 with dense's judgments
     dropped, and 1 for each other family.
   - **Source left out (exploratory; a sensitivity analysis on 66 of the 98
     questions).** Each target was removed from its own question's answers
     and from every ranking, so the sections below it move up. Link 1 is then
     +0.108 (+0.011 to +0.206, uncorrected p = 0.038).
7. **The results reproduce on the machine that produced them.**
   - **From an empty cache.** Once, on 2026-11-20, all 35 configurations
     were scored with every vector and score recomputed by the models, and
     again from the filled cache. The two runs' `summary.json` and
     `questions.jsonl` were identical byte for byte. That test runs at 2,000
     resamples without the label check, so it compares the two runs with
     each other.
   - **Against the committed files.** Another test re-runs the evaluation as
     `eval run` does, and requires the committed `summary.json` and
     `questions.jsonl` byte for byte. Re-running from the final code, before
     and after it was committed, reproduced both exactly.
   - **Perturbation (exploratory).** Every model score was moved by one
     float32 unit in the last place, in 20 runs. No chain configuration's top
     10 changed on any question. One unit is the smallest possible change;
     other hardware can differ by more.

![MRR@10 and time per question for each heading-chunk pipeline](results/quality_cost.svg)

## Results

`results/` holds the run quoted here, and [TABLES.md](TABLES.md) every table
for all 35 configurations. Metrics, budgets and intervals are Stage 2's (see
[how scoring works](../Stage%202/README.md#how-scoring-works)), with one
metric added. **nDCG@10 (answers only)** credits only sections graded 2. Graded
nDCG@10 also credits sections judged merely related, the line judges agree on
least.

Values are means over the 98 questions with 95% intervals. **Verdicts come
only from the confirmatory tables.** They follow the plan's rules:

- **Supported** means p < 0.05, Holm-adjusted for the supporting metrics, in
  the predicted direction.
- **Inconclusive** comes with `resolvable`: the smallest difference that
  comparison could detect at 80% power.
- **p per page** flips signs per target page, because 27 pages hold two
  targets. A result significant per question but not per page is called
  fragile.

In the other tables, **bold** marks a difference from the baseline that is
significant after Holm correction across the 29 factorial cells, metric by
metric. It describes; it is not a claim.

The intervals aren't adjusted for those 29 comparisons, so an interval can
exclude zero without being bold. heading+bm25's Recall in 4.8k chars against
the baseline is −0.09 (−0.16 to −0.01): p = 0.022 before Holm, 0.37 after.

Intervals on differences come from a paired bootstrap, and a configuration's
own yes/no rates get Wilson intervals. p-values come from a sign-flip test,
so near zero an interval and a p can disagree even before any adjustment.
Success@5's interval in the supporting table excludes zero, yet its p is
0.071: with 7 questions gained and 1 lost, the sign-flip test is the stricter
of the two.

Run `20261121-163240-25ad5ed6` · 98 questions · golden v2 `4e8a0f5c03d0` · FastAPI 0.165.0 `3ec279619717` · code `25ad5ed6ddb6` · 10,000 resamples

### The confirmatory tests

| test | difference (95% CI) | p | p per page | resolvable | verdict |
|---|---:|---:|---:|---:|---|
| link 1: heading+hybrid+rerank vs heading+bm25, MRR@10 | +0.090 (+0.020 to +0.159) | 0.0126 | 0.0210 | ±0.099 | supported |
| link 2: heading+hybrid vs heading+bm25, MRR@10 | +0.104 (+0.052 to +0.157) | 0.0004 | 0.0001 | ±0.076 | supported |
| link 3: heading+dense vs heading+bm25, MRR@10 | +0.036 (-0.041 to +0.115) | 0.3722 | 0.3791 | ±0.114 | inconclusive |
| link 4: heading+hybrid+rerank vs heading+hybrid, MRR@10 | -0.014 (-0.073 to +0.048) | 0.6649 | 0.7069 | ±0.086 | not reached: an earlier link was not significant, so this is exploratory |
| link 5: heading+hybrid+rerank vs heading+bm25+rerank, MRR@10 | -0.008 (-0.028 to +0.006) | 0.4492 | 0.4528 | ±0.025 | not reached: an earlier link was not significant, so this is exploratory |

Supporting metrics, headline pair (Holm-adjusted p):

| test | difference (95% CI) | p | p per page | resolvable | verdict |
|---|---:|---:|---:|---:|---|
| Success@5 | +0.061 (+0.010 to +0.122) | 0.0711 | 0.0692 | ±0.079 | inconclusive |
| nDCG@10 (answers only) | +0.091 (+0.041 to +0.143) | 0.0035 | 0.0075 | ±0.072 | supported |
| nDCG@10 | +0.071 (+0.038 to +0.105) | 0.0006 | 0.0012 | ±0.049 | supported |
| Recall@5 | +0.062 (+0.016 to +0.115) | 0.0330 | 0.0306 | ±0.072 | supported |
| Mean coverage in 4.8k chars | +0.050 (+0.013 to +0.092) | 0.0330 | 0.0306 | ±0.057 | supported |
| Recall in 4.8k chars | +0.064 (+0.022 to +0.112) | 0.0220 | 0.0232 | ±0.066 | supported |

Success@5: 7 questions fixed, 1 broken, 74 found by both, 16 missed by both.

Label check, link 1 on the blind round's grades:

| test | difference (95% CI) | p | p per page | resolvable | verdict |
|---|---:|---:|---:|---:|---|
| MRR@10 | +0.063 (+0.002 to +0.125) | 0.0442 | 0.0553 | ±0.087 | supported (fragile per page) |

Answer-versus-not agreement with the new labels: κ = 0.88 over 241 labels. Verdict: **supported**, its blind-round confirmation fragile per page. Stage 4 inherits heading+hybrid+rerank.

### Ranking quality

| configuration | Success@5 | Recall@5 | MRR@10 | nDCG@10 | nDCG@10 (answers only) |
|---|---:|---:|---:|---:|---:|
| **baseline (RAG Mini Chat)** | 0.45 (0.35–0.55) | 0.26 (0.19–0.33) | 0.28 (0.21–0.35) | 0.24 (0.20–0.29) | 0.26 (0.20–0.31) |
| heading+bm25 | 0.77 (0.67–0.84) | 0.56 (0.47–0.63) | 0.62 (0.53–0.70) | 0.53 (0.48–0.59) | 0.57 (0.49–0.64) |
| heading+tfidf | 0.65 (0.55–0.74) | 0.46 (0.37–0.54) | 0.49 (0.41–0.57) | 0.46 (0.40–0.52) | 0.47 (0.40–0.55) |
| heading+dense | 0.81 (0.72–0.87) | 0.58 (0.51–0.66) | 0.65 (0.57–0.73) | 0.60 (0.55–0.66) | 0.62 (0.55–0.69) |
| heading+hybrid | 0.85 (0.76–0.90) | 0.64 (0.56–0.72) | 0.72 (0.64–0.79) | 0.63 (0.58–0.68) | 0.66 (0.59–0.72) |
| heading+hybrid+rerank | 0.83 (0.74–0.89) | 0.62 (0.54–0.70) | 0.71 (0.63–0.78) | 0.61 (0.55–0.66) | 0.66 (0.59–0.72) |
| heading+bm25+rerank | 0.83 (0.74–0.89) | 0.62 (0.54–0.70) | 0.71 (0.64–0.79) | 0.60 (0.54–0.65) | 0.65 (0.58–0.72) |

### What a 4,800-character prompt receives

| configuration | Success in 4.8k chars | Recall in 4.8k chars | Mean coverage in 4.8k chars | Best coverage in 4.8k chars |
|---|---:|---:|---:|---:|
| **baseline (RAG Mini Chat)** | 0.87 (0.79–0.92) | 0.67 (0.60–0.74) | 0.23 (0.19–0.28) | 0.34 (0.28–0.40) |
| heading+bm25 | 0.80 (0.71–0.86) | 0.58 (0.50–0.66) | 0.53 (0.45–0.61) | 0.73 (0.64–0.81) |
| heading+tfidf | 0.69 (0.60–0.78) | 0.50 (0.42–0.58) | 0.46 (0.37–0.54) | 0.62 (0.53–0.71) |
| heading+dense | 0.83 (0.74–0.89) | 0.63 (0.55–0.70) | 0.55 (0.47–0.63) | 0.76 (0.68–0.84) |
| heading+hybrid | 0.85 (0.76–0.90) | 0.66 (0.58–0.73) | 0.58 (0.50–0.66) | 0.77 (0.69–0.85) |
| heading+hybrid+rerank | 0.86 (0.77–0.91) | 0.65 (0.57–0.72) | 0.58 (0.50–0.66) | 0.79 (0.71–0.86) |
| heading+bm25+rerank | 0.86 (0.77–0.91) | 0.64 (0.56–0.71) | 0.57 (0.49–0.65) | 0.79 (0.71–0.87) |

### Against the baseline: ranking

| vs baseline (RAG Mini Chat) | Success@5 | Recall@5 | MRR@10 | nDCG@10 | nDCG@10 (answers only) |
|---|---:|---:|---:|---:|---:|
| heading+bm25 | **+0.32 (+0.21 to +0.42)** | **+0.30 (+0.22 to +0.38)** | **+0.34 (+0.25 to +0.42)** | **+0.29 (+0.24 to +0.34)** | **+0.31 (+0.24 to +0.37)** |
| heading+tfidf | **+0.20 (+0.09 to +0.31)** | **+0.20 (+0.12 to +0.28)** | **+0.21 (+0.13 to +0.30)** | **+0.22 (+0.17 to +0.27)** | **+0.22 (+0.15 to +0.28)** |
| heading+dense | **+0.36 (+0.24 to +0.47)** | **+0.33 (+0.24 to +0.41)** | **+0.37 (+0.28 to +0.46)** | **+0.36 (+0.30 to +0.41)** | **+0.36 (+0.29 to +0.44)** |
| heading+hybrid | **+0.40 (+0.29 to +0.50)** | **+0.38 (+0.30 to +0.46)** | **+0.44 (+0.35 to +0.53)** | **+0.39 (+0.34 to +0.44)** | **+0.40 (+0.34 to +0.47)** |
| heading+hybrid+rerank | **+0.38 (+0.28 to +0.48)** | **+0.36 (+0.28 to +0.44)** | **+0.43 (+0.33 to +0.51)** | **+0.36 (+0.31 to +0.41)** | **+0.40 (+0.33 to +0.47)** |

### Against the baseline: the prompt

| vs baseline (RAG Mini Chat) | Success in 4.8k chars | Recall in 4.8k chars | Mean coverage in 4.8k chars | Best coverage in 4.8k chars |
|---|---:|---:|---:|---:|
| heading+bm25 | -0.07 (-0.16 to +0.02) | -0.09 (-0.16 to -0.01) | **+0.30 (+0.24 to +0.37)** | **+0.39 (+0.31 to +0.47)** |
| heading+tfidf | **-0.17 (-0.26 to -0.09)** | **-0.17 (-0.24 to -0.10)** | **+0.23 (+0.17 to +0.29)** | **+0.29 (+0.22 to +0.36)** |
| heading+dense | -0.04 (-0.13 to +0.05) | -0.05 (-0.13 to +0.04) | **+0.32 (+0.26 to +0.38)** | **+0.42 (+0.35 to +0.50)** |
| heading+hybrid | -0.02 (-0.10 to +0.06) | -0.02 (-0.09 to +0.06) | **+0.35 (+0.29 to +0.42)** | **+0.44 (+0.36 to +0.51)** |
| heading+hybrid+rerank | -0.01 (-0.09 to +0.07) | -0.02 (-0.09 to +0.05) | **+0.35 (+0.29 to +0.42)** | **+0.46 (+0.38 to +0.53)** |

### By question style

| Success@5 by question style | keyword (n=33) | paraphrase (n=32) | scenario (n=33) |
|---|---:|---:|---:|
| baseline (RAG Mini Chat) | 0.61 (0.44–0.75) | 0.28 (0.16–0.45) | 0.45 (0.30–0.62) |
| heading+bm25 | 1.00 (0.90–1.00) | 0.47 (0.31–0.64) | 0.82 (0.66–0.91) |
| heading+hybrid+rerank | 1.00 (0.90–1.00) | 0.53 (0.36–0.69) | 0.94 (0.80–0.98) |

### Diagnostics

| configuration | top-10 slots that are lone headings | chars in top 5 chunks | sections in 4.8k chars | top-10 sections judged | sections in 4.8k chars judged |
|---|---:|---:|---:|---:|---:|
| baseline (RAG Mini Chat) | 36% | 347 | 45.6 | 100% | 59% |
| heading+bm25 | 0% | 3,407 | 6.4 | 100% | 100% |
| heading+tfidf | 0% | 3,033 | 7.3 | 100% | 100% |
| heading+dense | 0% | 3,334 | 6.1 | 100% | 100% |
| heading+hybrid | 0% | 3,511 | 6.1 | 100% | 100% |
| heading+hybrid+rerank | 0% | 3,599 | 6.0 | 100% | 100% |
| heading+bm25+rerank | 0% | 3,582 | 6.1 | 100% | 100% |

## How v2's labels were built

v1 stays frozen with Stage 2, so Stage 2's results stay reproducible. v2 is v1
plus **one judging round**, built by `analysis/v2_labels.py` to the plan's
rules. Its `verify` command recomputes every draw and checks the committed
files against them.

**The pool** held 2,053 sections that v1 left unjudged:

- 2,012 ranked in some top 10 by one of the 35 configurations;
- 41 more inside a heading or fixed configuration's 4,800-character window.

The judges graded 49 of them answers, 147 related and 1,857 rejected. v2 holds
5,314 judgments: 318 answers (3.2 per question), 666 related and 4,330
rejected.

**The judges** were claude-opus-5-5, using v1's prompt word for word plus one
sentence: "Open no file other than the input file."

For reference they saw each question's v1 answers in full, with one
exception: the 100 v1 answers serving as hidden calibration items, on 49
questions (see below). Each was withheld from its own question's references,
so the references didn't give it away as a known answer.

Each worksheet sat alone in its own folder outside the project, so no judge
could see which configuration found a section, how many did, any result, or
the plan. All 30 pool batches, 28 blind batches and 4 quote batches passed
validation on their first run, so none was judged again.

**200 hidden calibration items** measured drift between the rounds. They were
v1 labels mixed into the same batches and presented exactly like new
candidates. Their new grades never enter v2.

One tell remains, as the plan notes. v1 judged every section on each target
page, so no new candidate lies on its question's target page, while 45 of the
100 calibration answers do. That is why the 55 answers elsewhere are also
reported on their own.

- **Promotions.** Of the 100 graded 0 or 1 in v1, none was promoted to an
  answer (95% CI 0–3.7%). That is 0 of the 50 in heading+bm25's top 10 and 0
  of the 50 outside it (0–7.1% each).
- **Demotions.** Of the 100 v1 answers, 2 were demoted (0.6–7.0%), both in
  heading+bm25's top 10: 2 of 50 there (1.1–13.5%) and 0 of 50 outside it
  (0–7.1%). Of the 55 answers off their question's target page, 1 was
  demoted (0.3–9.6%).
- **Overall,** the two rounds agreed exactly on 94% (κ = 0.90; answer versus
  not, κ = 0.98).

**A blind round** re-graded 1,638 sections with only the question and the
section in view: no earlier grade, no reference answers. It covered three
groups:

- **Every section either headline pipeline ranks in its top 10: 1,438.**
  These are all the labels link 1 reads, judged in one round under one
  condition. On the 241 new in v2, answer-versus-not agreement with v2 is
  κ = 0.88. That figure is the label check's, and it clears the plan's 0.70.
  On the 1,197 from v1, agreement is 91.7% exact (answer κ = 0.92); the blind
  judge promoted 7 and demoted 19.
- **An agreement sample of 160 of v2's other new labels:** all 40 new answers
  outside the union, plus 60 related and 60 rejected. Agreement is 91.9%
  exact (answer κ = 0.97), or 98.5% reweighted to all new labels, which are
  mostly rejections.
- **40 items from v1's agreement study,** to see whether the blind judge
  itself had moved. It agreed with v1's blind judge on 92.5%.

**A quote-checking judge** saw 209 items: the agreement sample, the bridge
and the union's 9 new answers. It had to quote the sentence or line that
answers, and a quote counted only if code found it in the section. That
catches a judge quoting text the section doesn't contain, as one answering
from its own knowledge of FastAPI might. Code can't tell whether a real line
actually answers.

- Every one of the 62 items graded 2 came back as an answer with a quote that
  holds. They are all 49 of v2's new answers and 13 v1 answers from the
  bridge.
- 6 of the 74 related items were called answers too, and none of the 73
  rejected ones.
- No quote failed.

`analysis/label_reliability.py` recomputes all of these from the committed
judge outputs (`results/label_reliability.json`).

## How the new retrievers work

The settings were fixed before any result was read, and none was tuned on the
golden labels. [PLAN.md](PLAN.md#configurations-35-all-judged-to-the-same-depth)
gives each one's source.

| | setting |
|---|---|
| dense | Cosine similarity between the question and each chunk as indexed. The question gets bge's instruction, `Represent this sentence for searching relevant passages: `; chunks get none. Every chunk is ranked. |
| hybrid | score = Σ 1 / (60 + rank) over BM25's and dense's rankings, both fused in full. Tied BM25 or dense scores share the best rank before fusion. A chunk BM25 scores 0 gets no BM25 term. |
| rerank | The first stage's top 100 are re-scored on (question, chunk text as indexed). Ties keep first-stage order, and so do ranks 101 onwards. The cross-encoder's raw logit is the score. |
| ties | Exact ties in dense or fused scores are ordered by sha256 of `20261120:<question>:<chunk id>`. Stage 2's lexical retrievers keep their own corpus-order ties. |
| numerics | CPU, float32, 4 threads, one text at a time, inputs cut at 512 tokens. That cuts 6 of the 1,538 heading chunks for the embedder and 75 of the 9,800 question-and-chunk pairs the reranker scores for heading+hybrid+rerank (`results/truncation.json`). No question is cut. |
| models | Pinned in [`../models.lock.json`](../models.lock.json) by full revision and the SHA-256 and size of every file. Safetensors weights only, never remote code, loaded offline from a folder that is checked first. After loading, the weights in memory are compared with the files. |
| caches | Every vector and score is cached per text, so a model loads only when something isn't cached. A test recomputes a seeded sample from the models and requires the same bits. |

## What each pipeline costs

`raglab bench` timed the heading-chunk pipelines on the 98 questions, with
every cache off.

- **Machine:** an Apple M5 Max (18 cores, 48 GB), on AC power, with 4
  threads and one text at a time, as evaluated.
- **Protocol:** after 10 warm-up questions, 5 passes over the 98, with the
  order of the pipelines shuffled in each pass.
- **Not part of the evaluation:** it reads no label, sits outside the
  reproducible results, and nothing that computes the results reads
  `results/cost.json`.

| configuration | median ms per question | 95th percentile ms | reranking, mean ms | CPU-seconds per 1,000 questions |
|---|---:|---:|---:|---:|
| heading+tfidf | 0.3 | 0.4 | – | 0.3 |
| heading+bm25 | 0.05 | 0.08 | – | 0.05 |
| heading+dense | 5.7 | 5.9 | – | 6.5 |
| heading+hybrid | 5.9 | 6.1 | – | 6.7 |
| heading+hybrid+rerank [depth 10] | 82 | 107 | 77 | 139 |
| heading+hybrid+rerank [depth 20] | 158 | 200 | 152 | 268 |
| heading+hybrid+rerank [depth 50] | 382 | 479 | 380 | 662 |
| heading+hybrid+rerank | 758 | 927 | 759 | 1,316 |
| heading+bm25+rerank | 730 | 891 | 739 | 1,266 |

- **Question embedding** averages 5.3 ms of dense retrieval's 5.4 ms mean;
  the table gives medians. Fusion adds 0.2 ms.
- **Reranking** is nearly all of a reranked pipeline's time, and grows with
  depth: 77 ms on average for 10 candidates, 759 ms for 100.
- **Batching** all 100 pairs into one batch was slower here: reranking alone
  averaged 1,306 ms batched, against 759 ms one pair at a time. One batch
  probably pads every pair to the longest.
- **Startup**, the median of five fresh processes. Importing PyTorch and
  sentence-transformers takes 1.7 s, and loading both models 0.24 s. The
  first model calls, one question embedding and one cross-encoder pair on a
  made-up passage, take 17 ms. No retrieval is timed there.
- **Index build:** embedding all 1,538 heading chunks one at a time for the
  dense index took 20.5 s, timed once in the benchmark process.
- **Size:** the model files are 128 MiB and 88 MiB. The dense index is 1,538
  × 384 float32 vectors (2.25 MiB), held as a 4.5 MiB float64 matrix.
- **Memory:** the benchmark process peaked at 4,699 MiB of resident memory
  over its whole run. That run loaded PyTorch, both models and all nine
  pipelines, embedded every chunk and ran the one-batch test, so 4,699 MiB is
  not one pipeline's footprint.

## Exploratory results

These follow the plan's exploratory list and carry no claims. Paired
comparisons come with their 95% intervals. Target-page recall, the stale-pool
shifts, Kendall's τ, the diagnostics and the counts are point values. The
p-values are uncorrected, except in the against-the-baseline tables, so with
this many some will fall below 0.05 by chance. Numbers come from
`results/summary.json` unless another file is named.

**Settings and switches** ([TABLES.md](TABLES.md)). Averaged over the six
chunker and breadcrumb settings:

| switch | MRR@10 |
|---|---|
| hybrid instead of BM25 | **+0.09 (+0.05 to +0.13)** |
| dense instead of BM25 | +0.05 (−0.01 to +0.10) |
| hybrid instead of dense | **+0.04 (+0.00 to +0.08)** |
| reranking the hybrid | −0.01 (−0.04 to +0.03) |

Bold: the interval excludes zero; hybrid over dense starts at +0.004.

**Chunkers.** Heading chunks beat both other chunkers with every retriever
family, and every one of these intervals lies above zero.

- Against fixed windows, MRR@10 rises by +0.07 (+0.02 to +0.13) with the
  lexical retrievers, +0.10 (+0.04 to +0.17) with dense, +0.10 (+0.04 to
  +0.16) with the hybrid and +0.11 (+0.05 to +0.18) reranked.
- Against paragraphs it rises by +0.12 (+0.07 to +0.18), +0.06 (+0.00 to
  +0.11), +0.09 (+0.03 to +0.14) and +0.08 (+0.03 to +0.13).

No test compares the families, and their intervals overlap widely.

**The breadcrumb** adds +0.04 (+0.01 to +0.06) with the lexical retrievers,
+0.04 (+0.00 to +0.08) with dense, +0.05 (+0.02 to +0.09) with the hybrid and
+0.03 (−0.01 to +0.06) reranked.

**The extras.** `analysis/extras.py`, added after freeze point 2, writes the
paired differences to `results/extras.json`. The depth curve's MRR@10 levels
and the last two items come from `results/summary.json`.

- **The query instruction:** −0.041 MRR@10 (−0.079 to −0.006), as finding 5
  says.
- **The rerank-depth curve:** MRR@10 runs 0.73, 0.72, 0.71 and 0.71 at depths
  10, 20, 50 and 100, against 0.72 for the hybrid itself. Against the hybrid
  that is +0.012, +0.004, −0.006 and −0.014, with every interval spanning
  about ±0.06.
- **The hybrid against each of its inputs:** +0.104 over BM25 (link 2) and
  +0.067 over dense (+0.010 to +0.124, p = 0.024). Neither interval lies
  within ±0.05, so by the plan's definition the hybrid is not "within 0.05"
  of either.
- **Reranking BM25:** +0.098 (+0.031 to +0.167, p = 0.005, and 0.01 per
  page).

**The pool and bias checks** (`analysis/pool_effects.py`,
`results/pool_effects.json`):

- **Stale pool.** The new configurations were scored on v1's labels, where
  unjudged sections count as not relevant. There, 12–40% of their top-10
  slots held a section v1 never judged (14–41% of the sections they ranked).
  Their MRR@10 on v1 was never higher than on v2, and at most 0.013 lower
  (fixed+hybrid+rerank without the breadcrumb). Recall@5 was higher on v1, by
  up to 0.03, because v2's extra answers enlarge the denominator.
- **Stage 2 on the wider pool.** v2 leaves Stage 2's Success@5 and MRR@10
  exactly as they were, by construction. As answer sets grew, Recall@5, both
  nDCGs and mean coverage fell by at most 0.023. heading+bm25's Recall@5 went
  from 0.57 to 0.56.
- **Leaving one family out.** Each new family's single-family judgments were
  dropped in turn and everything re-scored:
  - dense: 755 judgments; τ = 0.99, and the largest change is −0.010
    MRR@10, for fixed+dense;
  - hybrid: 133; nothing changed;
  - hybrid+rerank: 389; nothing changed;
  - bm25+rerank: 18; nothing changed.
- **Answers Stage 2 couldn't reach.** Of the 60 v1 answers in no lexical top
  10, heading+dense ranks 22 in its top 10. Of the 8 questions with no answer
  in any lexical top 10, it answers 4 in its top 10. For heading+hybrid+rerank
  the counts are 18 and 4. Every new configuration's counts are in the file.
- **Target-page recall.** 50 questions have target-page answers other than
  the target: 116 answers, all judged whatever ranked them. Recall@10,
  averaged per question, is 0.53 for BM25, 0.66 for dense, 0.64 for the
  hybrid and 0.64 for the reranked hybrid. Of the 116 answers, their top 10s
  hold 52, 71, 70 and 68. No interval was computed for these.
- **Source left out,** a sensitivity analysis. On the 66 questions with an
  answer besides the target, the target is removed from the answers and from
  every ranking, and the sections below it move up. Link 1 is then +0.108
  (+0.011 to +0.206, p = 0.038; 0.043 per page).

**Diagnostics** (`results/summary.json`, under `diagnostics`; TABLES.md shows
the top-10 and 4.8k judged shares):

- **Judged share.** Every heading and fixed window is fully judged at 1,200,
  2,400 and 4,800 characters. At 9,600, 92.8–98.9% of sections are judged in
  heading windows and 89.5–95.8% in fixed ones, since the pool covered only
  4,800-character windows. Paragraph windows are 95.8–99.7%, 83.1–98.5%,
  63.7–91.9% and 44.7–75.9% judged at the four budgets, and the baseline's
  88.1%, 75.1%, 58.8% and 42.1%.
- **Related share.** A section graded 1 (related) fills 22% of heading+bm25's
  top-10 slots, 26% of heading+dense's, 28% of heading+hybrid's and 24% of
  heading+hybrid+rerank's. Across all 35 configurations it is 10–28%. Graded
  nDCG@10 credits these slots; the answers-only version doesn't.
- **Reranking depth.** For every heading and fixed pipeline reranked to depth
  50 or 100, no chunk in any budget window lies below the rerank depth.
  Reranking only the top 10 leaves 0.5% of the 4,800-character window and 23%
  of the 9,600-character one never re-scored; the top 20 leaves 0.3% of the
  9,600-character one. paragraph+hybrid+rerank's 9,600-character window is 6%
  unreranked. Without the breadcrumb, 0.7%, 3% and 8% of its 2,400-, 4,800-
  and 9,600-character windows are.
- **Ties.** An exact tie set the order of two of the first five chunks for
  13% of questions under heading+hybrid. That is reciprocal rank fusion's own
  ties, settled by the seeded rule. It happened for 3% under heading+bm25,
  and never under heading+dense or a reranked heading pipeline with the
  breadcrumb.
- **Near-ties** under 1e-6 occur in no heading pipeline's top 10.

**Perturbation** (`analysis/perturbation.py`, `results/perturbation.json`).
Every dense similarity and cross-encoder score moved by one float32 unit in
the last place, up or down by a seeded coin flip. Across 20 runs, none of the
five chain configurations' top 10s changed on any of the 98 questions, so
link 1 did not move. One unit is the smallest possible change; other hardware
can differ by more.

## Run it

```
raglab/
  experiments.py  the 35 configurations (the 30-cell factorial and 5 extras) and the five-link chain
  evaluate.py     scores them: the chain, supporting metrics, label check, page-level tests
  models.py       pins the models: revision, SHA-256 and size per file; download and verify
  neural.py       dense, hybrid and reranking retrievers; per-text caches; the seeded tie rule
  bench.py        `raglab bench`: what each pipeline costs, measured outside the results
  report.py       the README's tables and chart, generated from a saved run
  metrics.py      Stage 2's metrics plus nDCG@10 (answers only)
  stats.py        Stage 2's bootstrap and sign-flip tests, plus per-page versions and the style-shuffle test
  ...             cli, golden and retrieval (updated for Stage 3); corpus, markdown, chunking, sections from Stages 1–2
analysis/         v2_labels.py builds and verifies the labels; label_reliability, truncation, pool_effects,
                  extras and perturbation write their files to results/; samples.py checks v1's sample draws;
                  the other four are Stage 2's, carried over unchanged (run them from Stage 2)
results/          the run quoted above, the cost benchmark, and each analysis's output
tests/            421 tests; those needing the corpus or the models skip without them
```

From the project folder (`Python Project 8/RAG Eval Lab`):

```bash
python3.14 -m venv .venv
```

```bash
.venv/bin/pip install --only-binary :all: -r "Stage 3/requirements-lock.txt"
```

For each package, `requirements-lock.txt` pins the one wheel CPython 3.14
installs on macOS arm64, so this install works only there. Anywhere else,
install the minimums with `.venv/bin/pip install -r "Stage 3/requirements.txt"`.
The code should run there too, though only macOS arm64 was tested, and
`raglab bench` needs the POSIX `resource` module, so not on Windows. Library
versions and the last float bits can differ, so the results may not match
`results/` byte for byte.

Then, from `Stage 3`:

```bash
../.venv/bin/python -m raglab corpus fetch
```

```bash
../.venv/bin/python -m raglab models fetch
```

```bash
../.venv/bin/python -m raglab eval run
```

That fetches the FastAPI docs and both models exactly as the locks pin them,
with no credentials, and scores all 35 configurations into `runs/<run id>/`.
The first run computes every vector and score, which takes about 10 minutes on
the benchmark machine.

The other commands:

```
../.venv/bin/python analysis/v2_labels.py verify       # every draw recomputed, the label files checked
../.venv/bin/python -m raglab golden pool              # exits 1 if any top-10 section is unjudged
../.venv/bin/python -m raglab eval report results --markdown --cost results/cost.json
../.venv/bin/python -m raglab eval report results --markdown --full    # TABLES.md
../.venv/bin/python -m raglab eval cost-chart results --out results/quality_cost.svg
../.venv/bin/python -m raglab bench --out runs/cost.json   # timings on this machine; results/cost.json is the committed one
../.venv/bin/python analysis/label_reliability.py
../.venv/bin/python analysis/truncation.py
../.venv/bin/python analysis/pool_effects.py
../.venv/bin/python analysis/extras.py
../.venv/bin/python analysis/perturbation.py
../.venv/bin/python -m pytest -q
RAGLAB_COLD=1 ../.venv/bin/python -m pytest -q -k empty_cache   # the run from an empty cache (10–15 minutes)
```

**Reproducibility.** Byte-identical results are promised for the recorded
environment only: `requirements-lock.txt`, which pins every wheel by hash, 4
threads, and the chip and OS in the manifest. `results/` came from Python
3.14.7 on macOS arm64 with PyTorch 2.15.0.

The manifest records:

- the plan's SHA-256;
- the SHA-256 of `models.lock.json`, `requirements-lock.txt` and the judging
  prompts;
- the corpus fingerprint;
- the installed versions of Python, NumPy, scikit-learn, PyYAML, PyTorch,
  transformers, sentence-transformers, tokenizers and safetensors. The lock's
  other 40 packages appear only through its SHA-256;
- no personal paths.

A test re-runs the evaluation and requires the committed `summary.json` and
`questions.jsonl` byte for byte. It also requires the manifest, apart from
its run id, time, git state and environment.

If the project sits in iCloud Drive, keep `data` out of sync: `raglab`
refuses to store models or caches in a synced folder, and says how to fix it.

## Deviations from the plan

The plan asks for every change to it, or to the code after freeze point 1,
with its date, reason and effect on link 1. Under `raglab/`, only three files
changed after freeze point 1:

- `bench.py`, which is new;
- `cli.py`: the `bench`, `eval report --cost` and `eval cost-chart` commands,
  and the `golden agreement` default;
- `report.py`: the cost table, the chart and the label-check verdict line.

The modules that compute `summary.json` and `questions.jsonl` are
byte-identical to `0f05de5c`. So no change below moved link 1: +0.090,
supported, its blind-round confirmation fragile per page. Every change was
made on 2026-11-20, US Eastern time as in the plan, except the last re-run,
from the freeze point 3 commit on 2026-11-21. Run ids are stamped in UTC, so
all three runs in the table read 20261121.

| change | why | effect on link 1 |
|---|---|---|
| Freeze point 1 is commit `0f05de5c`. The plan says it would be tagged `stage3-plan`; the repository has no such tag. | Oversight. `v2-provenance.json` records the plan's SHA-256 from that commit, and a test checks that the plan still matches it. | none |
| `raglab bench`, the exploratory analyses and their tests were committed with the labels in the freeze point 2 commit (`88431f1a`), not before it. | They were finished while the labels were judged. `pool_effects.py` and `perturbation.py` score Stage 3's configurations; they ran only after that commit, as did `raglab bench`. Before it, only `truncation.py`, which reads no grade, and `label_reliability.py`, which reads grades but computes no metric, had run. | none |
| The manifest records the versions of Python and eight libraries, not every locked library's version as the plan says. | The code at freeze point 1 already worked this way; the plan's sentence overstated it. The lock file's SHA-256, also in the manifest, still pins every version. | none |
| Added after freeze point 2: `analysis/extras.py`. | It gives paired differences for the instruction's effect and the rerank-depth curve, both on the plan's exploratory list; the frozen code reported their means only. | none |
| Added after freeze point 2: `report.cost_table`, `report.quality_cost_svg`, `eval report --cost` and `eval cost-chart`, plus tests of the cost table and the chart, and tests that this README's cost table and TABLES.md are the generated ones. The test that checks the README's other tables dates from freeze point 1. | The README's cost table and chart. | none |
| Changed after freeze point 2: the confirmatory table's label-check line now says when the blind-round confirmation is fragile per page. | The plan calls a fragile Supported fragile wherever it is stated. Display only. | none |
| Fixed after freeze point 2: `raglab golden agreement` looked for `golden/v2-agreement.json`, which doesn't exist. It now reads `v1-agreement.json` by default, as its help text says. | A Stage 2 default that broke when Stage 3's golden set became v2. It reads no Stage 3 result. | none |
| Fixed a test: `test_a_fresh_run_reproduces_the_committed_results` now passes the blind grades to `summarize`, as `eval run` does. | Without them its label check was empty, so it could never match the committed `summary.json`. | none |
| `results/` was first written (run `20261121-023046-35c58b76`) from a working tree with uncommitted edits to `cli.py` and `report.py`, before `analysis/extras.py` existed, so its code hash matched no commit. It was regenerated from the final code (run `20261121-033633-25ad5ed6`), which was committed at freeze point 3 (`711f8b40`), and run once more from that commit with a clean tree (run `20261121-163240-25ad5ed6`). | So the manifest names committed code and a clean tree. Each time, `summary.json` and `questions.jsonl` came out byte-identical. | none |
| "Source left out" covers 66 questions, not the 64 the plan names. | The rule, questions with an answer besides the target, is applied as written. v2 found such answers for two more questions. | none |

## Known limitations

- **One model family wrote the questions and judged every section.** The
  blind round, the hidden calibration items and the quote checks measure
  consistency, not truth. A human check of the labels is outside this plan.
- **The plan's author had seen the questions, v1's labels and Stage 2's
  per-question results.** The plan says so, and records what ran before it
  ([What ran before this plan](PLAN.md#what-ran-before-this-plan)). Git shows
  the order of commits, not that nothing ran before them.
- **98 questions resolve differences of about 0.08–0.11 MRR@10.** The plan
  expected about 0.08 from Stage 2's lexical comparisons; for lexical against
  neural the spread is larger. Dense against BM25 could resolve only about
  0.11, which is why link 3 is inconclusive rather than negative.
- **Two small models, one corpus.** This tests bge-small and a MiniLM
  cross-encoder on FastAPI's docs; it is no verdict on dense retrieval or
  reranking in general. The cross-encoder was trained on web search queries
  (MS MARCO), and much of these docs is code. Whether that mismatch is why
  reranking the hybrid added nothing measurable wasn't tested; reranking
  BM25's top 100 did gain (+0.098, exploratory).
- **Labels reach only as far as the pool.** v1 judged every section on each
  question's target page. Off the target page, an answer that no
  configuration ranks in its top 10, and that no heading or fixed
  4,800-character window holds, stays unjudged. Paragraph windows, and heading and fixed windows at
  9,600 characters, aren't judged in full, so their budget metrics are
  exploratory.
- **Cost was measured on one machine,** on the CPU, one text at a time. A GPU
  or a different batching strategy would change the ratios. Cost enters no
  verdict.
- **Two families of tests, each at 5%.** The chain and the supporting metrics
  each hold their own 5%, so across both, the chance of a false claim can
  reach about 10%. The exploratory p-values are uncorrected, except in the
  against-the-baseline tables, where each metric is Holm-adjusted across the
  29 factorial cells.

## Next: Stage 4

Stage 4 inherits **heading+hybrid+rerank**, the pipeline the plan's rule
selected. Its blind-round confirmation is fragile per page. It adds answers
with enforced citations, LLM-judged faithfulness, and model cost and quality
trade-offs.

The first thing worth testing on new questions is whether the 6 ms hybrid can
replace the 758 ms reranked pipeline. Here reranking the hybrid changed MRR@10
by −0.014 (−0.073 to +0.048), but that was exploratory.
