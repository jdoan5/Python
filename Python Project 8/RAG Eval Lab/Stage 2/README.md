# RAG Eval Lab — Stage 2: a golden set, and numbers with error bars

Stage 1 ended with one question run through seven configurations, and a
warning that one question is an anecdote. Stage 2 makes it a measurement:

- **98 questions** about the FastAPI docs, with 3,261 graded judgments of
  which doc sections answer them. Two pools were judged: every section any
  configuration ranks in its top 10, and every section on each question's
  target page. So every section any configuration ranks in its top 10 is
  judged, and on the target's page the answer sets reach beyond what these
  retrievers find.
- **All 12 configurations**: 3 chunkers × 2 retrievers × breadcrumb on/off,
  the full factorial. The retriever switch changes two things at once: BM25
  replaces TF-IDF's scoring, and TF-IDF keeps RAG Mini Chat's word pairs
  (bigrams) for fidelity.
- **Every metric mean and every difference has a 95% interval.** Scores get
  bootstrap intervals and yes/no rates get Wilson intervals. Comparisons are
  paired by question, and the eleven against the baseline are Holm-corrected
  for each metric.
- **Every run is reproducible** from a manifest that pins the code, the
  corpus, the questions, the robustness inputs and every parameter. In the
  locked environment, a re-run reproduces the committed `summary.json` and
  `questions.jsonl` byte for byte, and a test checks exactly that.

## What it found

1. **Heading chunks with BM25 rank answers far higher, and deliver each one
   far more completely, but fewer of them reach the prompt.** Against RAG Mini
   Chat's paragraphs + TF-IDF, they put an answering section in the top 5 for
   **77%** of questions, against **45%** (+0.32, 95% CI +0.21 to +0.42). In a
   4,800-character prompt, a question's answering sections are on average
   **54%** present against **23%** (mean coverage, +0.31, +0.25 to +0.38), and
   the most complete one **73%** against **34%**. But fewer of the answering
   sections reach that prompt at all. 0.60 of them arrive against 0.69 (−0.09,
   −0.17 to −0.02, not significant after Holm), and some answering section
   arrives for 0.80 of questions against 0.87 (not significant). Finding 4
   explains the trade-off.

   The ranking lead survives the stress tests. Re-scoring with a second
   judge's grades, or without the two questions whose targets it doubted,
   leaves it unchanged or slightly larger. Simulating the unjudged sections
   inside 4,800-character contexts narrows the Recall@5 lead to +0.226 at
   worst. Success@5 can't move in that simulation, because every top-10
   section is already judged.
2. **BM25 beats TF-IDF, and heading chunks beat paragraphs.** BM25 scores
   higher in all six matched pairs, by +0.10 Recall@5 on average; four of the
   six differences are significant on their own, but not the two fixed-window
   pairs. Heading chunks beat paragraphs in all four pairs (+0.12, all
   significant). They beat fixed windows only with BM25 (+0.07, both pairs
   significant); with TF-IDF the difference is within noise.
3. **The breadcrumb helps paragraph chunks, and only them.** It raises their
   mean coverage by +0.12 and +0.15 (both pairs significant). It raises the
   baseline's Recall@5 by +0.12 (significant), and Recall@5 for paragraphs
   scored by BM25 by +0.05 (not significant). Part of that coverage gain comes
   from the breadcrumb travelling free. If each chunk's breadcrumb were charged
   to the prompt, the two gains would be +0.04 (not significant) and +0.08
   (significant). For heading and fixed chunks, no pair moves by more than 0.03
   on either metric, and none is significant. A plausible reason is that those chunks already carry their heading or plenty
   of surrounding text, but that isn't tested directly. Stage 1 kept the
   breadcrumb as a separate switch so an effect like this could be traced to
   its cause.
4. **Small chunks get *something* into the prompt more often. Big chunks
   deliver far more of an answer.** Comparing matched pairs one at a time,
   swapping paragraphs for heading chunks makes it 0.10–0.19 less likely that
   any answering section reaches a 4,800-character prompt (all four pairs
   significant). Dozens of fragments touch 23–46 different sections, so one
   of them usually lands. The same swap raises how completely the best
   answering section arrives by 0.15–0.37 (all four significant). Which of the
   two predicts a correct *generated* answer is Stage 4's question. Coverage
   is measured against whole sections, which favours heading chunks by
   construction; see [Known limitations](#known-limitations).
5. **Lexical retrieval breaks on paraphrased questions.** heading+bm25 puts
   an answer in its top 5 for every keyword question (95% CI 90–100%), 82% of
   scenario questions (66–91%) and 47% of paraphrases (31–64%). Its
   no-breadcrumb twin is indistinguishable, at 50% on paraphrases. **22% of
   the known answers appear in no configuration's top 10** (35% in no top 5).
   Answers off the target's page are only known when some configuration ranks
   them, so at least that share of all answers is beyond these retrievers.
   These are the gaps Stage 3's embeddings are for.
6. **36% of the baseline's top-10 results are lone headings.** Splitting on
   blank lines makes every heading its own chunk. A heading is short and dense
   with exactly the words people search for, so it often ranks above the text
   that answers.

![Success and coverage against context budget](results/budget.svg)

## Results

`results/` holds the run quoted here.

- **Success** is yes/no: did at least one answering section make it.
- **Recall** is a share: of a question's answering sections, how many made it.
- **Mean coverage** averages, over a question's answering sections, the share
  of each one's text that reached the prompt.
- **Best coverage** is that share for the most completely delivered answering
  section. Each answering section answers on its own, so this measures how
  much of *an* answer arrived.

Recall@5, MRR@10, nDCG@10, Recall in 4.8k and mean coverage were fixed
before any results were seen; they are the `headline` list in `summary.json`.
Success@5, Success in 4.8k and best coverage were added after review, to say
exactly what the findings claim.

Values are means over questions with 95% intervals. In the comparison tables,
**bold** marks a difference that is significant after Holm correction across
the eleven comparisons (paired sign-flip test, p < 0.05). The intervals come
from a different procedure (bootstrap, or Wilson for yes/no rates) and aren't
corrected, so near zero the two can disagree: an interval that barely excludes
zero can still miss bold.

### Ranking quality

| configuration | Success@5 | Recall@5 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|---:|
| heading+bm25 | 0.77 (0.67–0.84) | 0.57 (0.49–0.65) | 0.62 (0.53–0.70) | 0.56 (0.50–0.61) |
| heading+bm25 (no breadcrumb) | 0.76 (0.66–0.83) | 0.57 (0.48–0.65) | 0.60 (0.52–0.69) | 0.55 (0.49–0.60) |
| fixed+bm25 (no breadcrumb) | 0.69 (0.60–0.78) | 0.49 (0.41–0.58) | 0.52 (0.44–0.60) | 0.52 (0.46–0.57) |
| fixed+bm25 | 0.68 (0.59–0.77) | 0.50 (0.41–0.58) | 0.54 (0.46–0.62) | 0.53 (0.47–0.58) |
| heading+tfidf | 0.65 (0.55–0.74) | 0.47 (0.39–0.55) | 0.49 (0.41–0.57) | 0.48 (0.42–0.54) |
| paragraph+bm25 | 0.64 (0.54–0.73) | 0.48 (0.39–0.57) | 0.53 (0.44–0.61) | 0.45 (0.40–0.51) |
| heading+tfidf (no breadcrumb) | 0.64 (0.54–0.73) | 0.44 (0.36–0.52) | 0.49 (0.41–0.56) | 0.47 (0.41–0.52) |
| paragraph+bm25 (no breadcrumb) | 0.62 (0.52–0.71) | 0.43 (0.34–0.51) | 0.50 (0.41–0.58) | 0.41 (0.35–0.46) |
| fixed+tfidf (no breadcrumb) | 0.60 (0.50–0.69) | 0.42 (0.34–0.51) | 0.41 (0.33–0.49) | 0.45 (0.39–0.51) |
| fixed+tfidf | 0.58 (0.48–0.67) | 0.43 (0.34–0.51) | 0.43 (0.35–0.51) | 0.46 (0.39–0.52) |
| paragraph+tfidf | 0.57 (0.47–0.66) | 0.39 (0.31–0.47) | 0.41 (0.33–0.49) | 0.35 (0.30–0.41) |
| **baseline (RAG Mini Chat)** | 0.45 (0.35–0.55) | 0.26 (0.20–0.34) | 0.28 (0.21–0.35) | 0.25 (0.21–0.30) |

### What a 4,800-character prompt receives

| configuration | Success in 4.8k chars | Recall in 4.8k chars | Mean coverage in 4.8k chars | Best coverage in 4.8k chars |
|---|---:|---:|---:|---:|
| paragraph+bm25 | 0.91 (0.83–0.95) | 0.77 (0.70–0.83) | 0.40 (0.33–0.46) | 0.51 (0.43–0.58) |
| paragraph+tfidf | 0.89 (0.81–0.94) | 0.73 (0.65–0.80) | 0.35 (0.29–0.42) | 0.48 (0.40–0.55) |
| **baseline (RAG Mini Chat)** | 0.87 (0.79–0.92) | 0.69 (0.62–0.76) | 0.23 (0.19–0.28) | 0.34 (0.28–0.40) |
| paragraph+bm25 (no breadcrumb) | 0.87 (0.79–0.92) | 0.71 (0.63–0.78) | 0.24 (0.20–0.29) | 0.35 (0.29–0.41) |
| heading+bm25 | 0.80 (0.71–0.86) | 0.60 (0.52–0.68) | 0.54 (0.46–0.62) | 0.73 (0.64–0.81) |
| fixed+bm25 | 0.77 (0.67–0.84) | 0.58 (0.50–0.66) | 0.49 (0.41–0.57) | 0.67 (0.59–0.75) |
| fixed+bm25 (no breadcrumb) | 0.77 (0.67–0.84) | 0.58 (0.49–0.66) | 0.49 (0.41–0.56) | 0.67 (0.59–0.75) |
| heading+bm25 (no breadcrumb) | 0.77 (0.67–0.84) | 0.59 (0.50–0.67) | 0.53 (0.45–0.61) | 0.71 (0.62–0.80) |
| heading+tfidf (no breadcrumb) | 0.70 (0.61–0.79) | 0.52 (0.43–0.60) | 0.48 (0.39–0.56) | 0.65 (0.55–0.73) |
| heading+tfidf | 0.69 (0.60–0.78) | 0.52 (0.43–0.60) | 0.47 (0.39–0.55) | 0.62 (0.53–0.71) |
| fixed+tfidf (no breadcrumb) | 0.65 (0.55–0.74) | 0.49 (0.40–0.57) | 0.42 (0.33–0.50) | 0.55 (0.46–0.64) |
| fixed+tfidf | 0.64 (0.54–0.73) | 0.48 (0.40–0.56) | 0.43 (0.34–0.51) | 0.57 (0.48–0.66) |

### Against the baseline

| vs baseline (RAG Mini Chat) | Success@5 | Recall@5 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|---:|
| heading+bm25 | **+0.32 (+0.21 to +0.42)** | **+0.31 (+0.23 to +0.39)** | **+0.34 (+0.25 to +0.42)** | **+0.30 (+0.25 to +0.35)** |
| heading+bm25 (no breadcrumb) | **+0.31 (+0.20 to +0.41)** | **+0.30 (+0.22 to +0.39)** | **+0.33 (+0.24 to +0.41)** | **+0.29 (+0.24 to +0.34)** |
| fixed+bm25 (no breadcrumb) | **+0.24 (+0.13 to +0.36)** | **+0.23 (+0.15 to +0.31)** | **+0.24 (+0.15 to +0.33)** | **+0.27 (+0.21 to +0.32)** |
| fixed+bm25 | **+0.23 (+0.12 to +0.34)** | **+0.23 (+0.16 to +0.32)** | **+0.26 (+0.17 to +0.34)** | **+0.27 (+0.22 to +0.33)** |
| heading+tfidf | **+0.20 (+0.09 to +0.31)** | **+0.21 (+0.12 to +0.29)** | **+0.21 (+0.13 to +0.30)** | **+0.23 (+0.18 to +0.28)** |
| paragraph+bm25 | **+0.19 (+0.09 to +0.30)** | **+0.22 (+0.14 to +0.29)** | **+0.25 (+0.17 to +0.33)** | **+0.20 (+0.16 to +0.25)** |
| heading+tfidf (no breadcrumb) | **+0.19 (+0.08 to +0.31)** | **+0.18 (+0.10 to +0.26)** | **+0.21 (+0.13 to +0.28)** | **+0.22 (+0.17 to +0.26)** |
| paragraph+bm25 (no breadcrumb) | **+0.17 (+0.08 to +0.27)** | **+0.16 (+0.09 to +0.24)** | **+0.22 (+0.15 to +0.28)** | **+0.16 (+0.12 to +0.20)** |
| fixed+tfidf (no breadcrumb) | **+0.15 (+0.04 to +0.27)** | **+0.16 (+0.08 to +0.24)** | **+0.13 (+0.05 to +0.21)** | **+0.20 (+0.14 to +0.25)** |
| fixed+tfidf | **+0.13 (+0.02 to +0.26)** | **+0.16 (+0.08 to +0.25)** | **+0.15 (+0.07 to +0.24)** | **+0.20 (+0.15 to +0.26)** |
| paragraph+tfidf | **+0.12 (+0.03 to +0.21)** | **+0.12 (+0.06 to +0.19)** | **+0.13 (+0.08 to +0.19)** | **+0.10 (+0.07 to +0.13)** |

| vs baseline (RAG Mini Chat) | Success in 4.8k chars | Recall in 4.8k chars | Mean coverage in 4.8k chars | Best coverage in 4.8k chars |
|---|---:|---:|---:|---:|
| paragraph+bm25 | +0.04 (-0.02 to +0.10) | +0.07 (+0.02 to +0.13) | **+0.16 (+0.11 to +0.21)** | **+0.17 (+0.12 to +0.23)** |
| paragraph+tfidf | +0.02 (-0.03 to +0.08) | +0.03 (-0.02 to +0.09) | **+0.12 (+0.08 to +0.16)** | **+0.14 (+0.09 to +0.20)** |
| paragraph+bm25 (no breadcrumb) | +0.00 (-0.03 to +0.03) | +0.02 (-0.02 to +0.06) | +0.01 (-0.02 to +0.03) | +0.01 (-0.01 to +0.04) |
| heading+bm25 | -0.07 (-0.16 to +0.02) | -0.09 (-0.17 to -0.02) | **+0.31 (+0.25 to +0.38)** | **+0.39 (+0.31 to +0.47)** |
| fixed+bm25 | -0.10 (-0.19 to -0.01) | **-0.11 (-0.19 to -0.04)** | **+0.26 (+0.20 to +0.32)** | **+0.34 (+0.26 to +0.41)** |
| fixed+bm25 (no breadcrumb) | -0.10 (-0.19 to -0.01) | **-0.12 (-0.19 to -0.04)** | **+0.25 (+0.19 to +0.31)** | **+0.34 (+0.26 to +0.41)** |
| heading+bm25 (no breadcrumb) | -0.10 (-0.19 to -0.01) | **-0.11 (-0.18 to -0.03)** | **+0.30 (+0.23 to +0.36)** | **+0.38 (+0.30 to +0.45)** |
| heading+tfidf (no breadcrumb) | **-0.16 (-0.24 to -0.08)** | **-0.18 (-0.25 to -0.11)** | **+0.24 (+0.19 to +0.31)** | **+0.31 (+0.24 to +0.38)** |
| heading+tfidf | **-0.17 (-0.26 to -0.09)** | **-0.18 (-0.25 to -0.11)** | **+0.24 (+0.18 to +0.30)** | **+0.29 (+0.22 to +0.36)** |
| fixed+tfidf (no breadcrumb) | **-0.21 (-0.31 to -0.12)** | **-0.21 (-0.28 to -0.13)** | **+0.18 (+0.12 to +0.24)** | **+0.22 (+0.14 to +0.29)** |
| fixed+tfidf | **-0.22 (-0.33 to -0.13)** | **-0.21 (-0.29 to -0.14)** | **+0.19 (+0.14 to +0.25)** | **+0.23 (+0.16 to +0.31)** |

### What each switch is worth

Each row averages the paired difference over every setting of the other two
switches, so it reads "all else equal". The number in brackets is how many
configuration pairs went into the average. Here **bold** means the 95%
interval excludes zero (a bound shown as +0.00 is above zero at full
precision); these intervals aren't corrected for multiplicity.
Each pair also gets its own paired test in `summary.json`, Holm-adjusted
within the switch, and the findings quote those when they say how many pairs
are significant.

| switch (pairs compared) | Success@5 | Recall@5 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|---:|
| heading chunks instead of paragraphs (4) | **+0.13 (+0.07 to +0.20)** | **+0.12 (+0.07 to +0.18)** | **+0.12 (+0.07 to +0.18)** | **+0.15 (+0.12 to +0.18)** |
| fixed windows instead of paragraphs (4) | **+0.07 (+0.00 to +0.14)** | **+0.07 (+0.02 to +0.12)** | +0.05 (-0.02 to +0.11) | **+0.12 (+0.08 to +0.16)** |
| heading chunks instead of fixed windows (4) | **+0.06 (+0.01 to +0.12)** | **+0.05 (+0.01 to +0.09)** | **+0.07 (+0.02 to +0.13)** | +0.03 (-0.01 to +0.06) |
| BM25 instead of TF-IDF (6) | **+0.11 (+0.06 to +0.16)** | **+0.10 (+0.06 to +0.15)** | **+0.13 (+0.09 to +0.17)** | **+0.09 (+0.07 to +0.12)** |
| breadcrumb on (6) | +0.02 (-0.01 to +0.06) | **+0.04 (+0.01 to +0.06)** | **+0.04 (+0.01 to +0.06)** | **+0.03 (+0.02 to +0.05)** |

| switch (pairs compared) | Success in 4.8k chars | Recall in 4.8k chars | Mean coverage in 4.8k chars | Best coverage in 4.8k chars |
|---|---:|---:|---:|---:|
| heading chunks instead of paragraphs (4) | **-0.14 (-0.21 to -0.08)** | **-0.17 (-0.22 to -0.12)** | **+0.20 (+0.16 to +0.25)** | **+0.26 (+0.21 to +0.32)** |
| fixed windows instead of paragraphs (4) | **-0.18 (-0.24 to -0.11)** | **-0.19 (-0.25 to -0.14)** | **+0.15 (+0.11 to +0.19)** | **+0.20 (+0.14 to +0.25)** |
| heading chunks instead of fixed windows (4) | +0.03 (-0.02 to +0.08) | +0.02 (-0.02 to +0.06) | **+0.05 (+0.02 to +0.08)** | **+0.06 (+0.01 to +0.11)** |
| BM25 instead of TF-IDF (6) | **+0.07 (+0.03 to +0.11)** | **+0.07 (+0.03 to +0.10)** | **+0.05 (+0.02 to +0.09)** | **+0.07 (+0.03 to +0.11)** |
| breadcrumb on (6) | +0.01 (-0.02 to +0.04) | +0.02 (-0.01 to +0.04) | **+0.05 (+0.03 to +0.07)** | **+0.05 (+0.03 to +0.08)** |

### By question style

| Success@5 by question style | keyword (n=33) | paraphrase (n=32) | scenario (n=33) |
|---|---:|---:|---:|
| baseline (RAG Mini Chat) | 0.61 (0.44–0.75) | 0.28 (0.16–0.45) | 0.45 (0.30–0.62) |
| heading+bm25 | 1.00 (0.90–1.00) | 0.47 (0.31–0.64) | 0.82 (0.66–0.91) |
| paragraph+bm25 | 0.91 (0.76–0.97) | 0.28 (0.16–0.45) | 0.73 (0.56–0.85) |
| paragraph+tfidf | 0.91 (0.76–0.97) | 0.22 (0.11–0.39) | 0.58 (0.41–0.73) |
| fixed+bm25 | 0.97 (0.85–0.99) | 0.34 (0.20–0.52) | 0.73 (0.56–0.85) |
| fixed+tfidf | 0.88 (0.73–0.95) | 0.22 (0.11–0.39) | 0.64 (0.47–0.78) |
| heading+tfidf | 0.91 (0.76–0.97) | 0.31 (0.18–0.49) | 0.73 (0.56–0.85) |

### Diagnostics

| configuration | top-10 slots that are lone headings | chars in top 5 chunks | sections in 4.8k chars | top-10 sections judged | sections in 4.8k chars judged |
|---|---:|---:|---:|---:|---:|
| baseline (RAG Mini Chat) | 36% | 347 | 45.6 | 100% | 51% |
| paragraph+bm25 | 20% | 570 | 22.7 | 100% | 74% |
| paragraph+bm25 (no breadcrumb) | 20% | 476 | 36.3 | 100% | 54% |
| paragraph+tfidf | 32% | 464 | 23.0 | 100% | 73% |
| fixed+bm25 | 0% | 5,494 | 7.9 | 100% | 100% |
| fixed+bm25 (no breadcrumb) | 0% | 5,514 | 7.9 | 100% | 100% |
| fixed+tfidf | 0% | 5,473 | 7.7 | 100% | 99% |
| fixed+tfidf (no breadcrumb) | 0% | 5,495 | 8.0 | 100% | 99% |
| heading+bm25 | 0% | 3,407 | 6.4 | 100% | 100% |
| heading+bm25 (no breadcrumb) | 0% | 3,379 | 6.6 | 100% | 100% |
| heading+tfidf | 0% | 3,033 | 7.3 | 100% | 100% |
| heading+tfidf (no breadcrumb) | 0% | 3,043 | 7.4 | 100% | 100% |

## How the golden set was built

The labels decide every number above, so they were built to a written
procedure. The procedure, not the author, is what makes them trustworthy. The
questions were written, and the sections judged, by Claude (an LLM) following
it. The checks at the end measure how far that can be trusted.

**Questions come from a seeded sample, not from intuition.** `golden sample`
drew 100 target sections by a seeded hash, stratified by docs area in
proportion to size: tutorial 47, advanced 25, top-level pages 12, deployment
10, how-to 6. It took at most two per page and skipped the API-reference
stubs and the pages about the FastAPI project itself. Two samples had nothing
to ask (a "skip ahead" note and a narrative lead-in); they are recorded with
their reasons. One question was written per remaining section: 98.

**Each question had a style assigned before it was written,** in rotation:
*keyword* (the docs' own terms), *paraphrase* (plain words, avoiding the
section's distinctive terms) or *scenario* (a symptom or situation). The
styles really differ. The share of a question's words that also appear in its
target section is 0.78 (95% CI 0.73–0.84) for keyword questions, 0.47
(0.42–0.52) for scenarios and 0.26 (0.20–0.31) for paraphrases
(`analysis/question_style_overlap.py`).

**Questions were frozen before judging began.** The procedure was to write
and fingerprint all 98 (`c4673ddd…`) before running any configuration on
them. The fingerprint test detects any later edit to a question. The order of
work itself becomes checkable once it's in git history.

**Labels come from two pools.** The first is every section that any of the 12
configurations ranks in its top 10, which is the depth the rank metrics read. On
its own, that pool would only contain what these lexical retrievers can find,
so the second is every section on the target's own page. 60 of the 269 known
answers come from the page pool alone. A test fails if any configuration's top
10 holds an unjudged section, so a new retriever can't be scored against
labels that miss its own top 10 by accident. Budget contexts reach deeper, and
are covered by the sampling described under Known limitations.

**How the judging worked.** Judges worked under the rubric in `v1.yaml`. They
saw the question, its known answers for reference, and each section's text,
but never which configuration had found a section, or how many had. The
first round showed sections cut at 1,800 characters, and an audit found the
cut made judges under-grade long sections. So the 186 longer sections from
that round were re-judged on their full text. Where the two grades differed,
a separate blind tie-break was run, and the final grade is the median of the
three. So a grade changed only when the tie-break agreed with the full-text
re-judgment. 28 did, all upward, and 18 of them became answers; two kept
their first grade. `golden/v1-regrade.json` logs every one, and a
test recomputes these counts. Everything judged later saw full text from the
start. In total there are 3,261 judgments: 269 answers (2.74 per question),
519 related, and 2,473 rejected.

**A second, blind judge checked the labels.** It re-graded a stratified
sample of 240 of them. It saw only the question and the section's full text:
no first-pass grade, no reference answer, and no sign of which item was a
question's target. It agreed 89% of the time (Cohen's κ = 0.83). Reweighted
to the whole label set, which is mostly rejections, that's 86% (κ = 0.69).

Most disagreements fall on the line between *related* and *not relevant*,
which only nDCG uses. On the line everything else rests on, *answer* versus
*not an answer*, κ is **0.92** reweighted. The second judge also confirmed 38
of the 40 targets it saw. `raglab golden agreement` recomputes all of this
from `golden/v1-agreement.json`.

The labels weren't changed after the study. Re-scoring with the second
judge's grades for all 26 disputed labels leaves heading+bm25's lead
unchanged or slightly larger; the largest move in any mean is 0.031. Scaled
up from the sample, a full re-judging would dispute about 460 of the 3,261
labels, but only about 37 on the answer line (`analysis/label_sensitivity.py`).

## How scoring works

**The unit is the section.** Labels name sections and retrievers return
chunks, so every chunk is mapped to the sections it *retrieves*. A chunk
retrieves a section when it holds at least 100 characters of the section's
body, or all of that body, or is made of nothing but that section's text
(headings aside). A chunk holding only a heading retrieves nothing, because a
title isn't an answer.

**Rank metrics use a ranking of sections.** Walking down a configuration's
chunks, each section a chunk retrieves joins the list unless it's already
there. A chunk that retrieves nothing, such as a lone heading, takes a slot as
a miss. Counting in sections puts all three chunkers in the same units.
Success@5 and Recall@5 look at the top 5 sections, MRR@10 at the first
answering section, and nDCG@10 at both positive grades (gain 3 for a 2, gain 1
for a 1).

**Budget metrics give every configuration the same prompt size.** Five slots
can mean very different amounts of text: the first five chunks are about 350
characters for the baseline and 3,000–5,500 for heading and fixed chunks. So
each configuration's ranking is also packed into a fixed budget, the way a
prompt with a hard limit would be: the longest prefix of its chunks that fits.
Every chunk with a positive score is ranked, so even a 9,600-character budget
fills whenever the retriever has that much to give. The budget counts chunk
text only. The breadcrumb is an indexing feature here; how a prompt labels its
chunks is a Stage 4 decision that applies to every configuration alike.
`analysis/breadcrumb_budget.py` shows what charging it would change. The
headline budget is 4,800 characters (about 1,200 tokens, room for four
full-size chunks), and it was fixed before any results were seen.

**Uncertainty comes from resampling questions.** Intervals on scores are
percentile bootstrap intervals over questions, with 10,000 resamples. Yes/no
rates use Wilson intervals, which stay honest at 0% and 100%. Comparisons are
paired: both configurations are scored on the same resampled questions, which
cancels out how hard each question is. p-values come from a paired sign-flip
randomisation test. Everything is seeded.

## Run it

```
raglab/
  sections.py     which sections a chunk retrieves: the hit rule
  metrics.py      success, recall, MRR, nDCG on section rankings; budget packing, success, recall, coverage
  stats.py        bootstrap and Wilson intervals, paired differences, sign-flip test, Holm, Cohen's kappa
  golden.py       the golden set: schema, strict loading, sampling, merging, agreement
  evaluate.py     runs the 12 configurations: summaries, switch effects, diagnostics, manifests
  report.py       the README's tables and chart, generated from a saved run
  ...             corpus, markdown, chunking, retrieval from Stage 1; experiments and cli extended
analysis/         one script per robustness claim above, plus the sample draws (samples.py)
results/          the run quoted above
tests/            291 tests (18 need the fetched corpus and skip without it)
```

```bash
cd "Python Project 8/RAG Eval Lab"
python3.14 -m venv .venv          # the lock needs Python >= 3.12; results/ came from 3.14.7 on macOS arm64
.venv/bin/pip install -r "Stage 2/requirements-lock.txt"   # exact versions; requirements.txt has the minimums
cd "Stage 2"

../.venv/bin/python -m raglab corpus fetch           # the FastAPI docs, exactly as corpus.lock.json pins them
../.venv/bin/python -m raglab golden validate        # the labels match the pinned corpus
../.venv/bin/python -m raglab golden pool            # exits 1 if any top-10 section is unjudged
../.venv/bin/python -m raglab golden agreement       # how consistent the labels are
../.venv/bin/python -m raglab eval run               # all 12 configurations -> runs/<run id>/
../.venv/bin/python -m raglab eval report results --markdown
../.venv/bin/python -m raglab eval chart results --highlight heading+bm25 --highlight paragraph+bm25 \
    --highlight "baseline (RAG Mini Chat)" --out results/budget.svg
../.venv/bin/python analysis/label_sensitivity.py      # what the second judge's grades would change
../.venv/bin/python analysis/unjudged_sensitivity.py   # what the unjudged sections could change (~30 s)
../.venv/bin/python analysis/breadcrumb_budget.py      # what charging the breadcrumb would change
../.venv/bin/python analysis/samples.py verify         # the committed samples are the documented draws
../.venv/bin/python -m pytest -q
```

Each run writes `manifest.json` (what produced the numbers), `summary.json`
(the numbers) and `questions.jsonl` (every question's scores and top-10
sections under every configuration, so two runs can be diffed question by
question). The manifest records no personal paths, only the OS, the
architecture and library versions. requirements.txt's minimums run from Python
3.10. Byte-for-byte reproduction is only promised in the locked environment.

## Known limitations

- **One model family wrote and judged everything.** The second judge was
  independent and blind, but it wasn't human. A human pass over the
  disagreements would be the next step.
- **Answer sets come from 12 lexical configurations plus each target's page.**
  An answer elsewhere that none of them ranks in its top 10 stays unjudged,
  so the 22% of known answers beyond every top 10 is a floor. Stage 3's
  retrievers will add their top 10 to the pool, in a new `v2.yaml`.
- **Budgets reach past the judged pool.** Inside a 4,800-character context,
  99–100% of the sections a heading or fixed configuration retrieves are
  judged, but only 51–74% for paragraph configurations, which scatter
  fragments widely. Unjudged sections score as not relevant.
  `golden/v1-unjudged-sample.json` judges every remaining unjudged section
  inside a heading or fixed context (29 sections, no answers) and a random 200
  of the 3,768 that only paragraph contexts reach (no answers; the 95% upper
  bound is 1.9%). `analysis/unjudged_sensitivity.py` completes the labels in
  simulation at that bound. Every lead it checks keeps its sign in all 100
  simulations:
  - heading+bm25's lead over the baseline narrows to +0.226–0.282 on
    Recall@5 and +0.204–0.263 on mean coverage. nDCG@10 isn't simulated, and
    neither are unjudged sections outside every context;
  - the small chunks' "arrives at all" lead *grows*, from +0.07 to as much as
    +0.17 for the baseline over heading+bm25;
  - heading+bm25 itself tops Recall@5 in 85% of simulations, and a
    heading+bm25 variant tops it in all of them.
- **Coverage is measured against whole sections,** the unit heading chunks
  are cut on. A heading chunk holds a whole section of up to 1,200
  characters, so retrieving it covers that section completely. The 33% of
  answering sections that are longer get split, as with any chunker. With
  labels that marked the answering sentences, a small chunk carrying just
  that sentence would earn fuller credit. Best coverage is the measure that
  favours heading chunks most; success is the one least tied to the section
  unit.
- **The breadcrumb isn't charged to the budget.** If each chunk's breadcrumb
  were sent too, paragraph+TF-IDF's chance of getting an answer into 4,800
  characters would fall from 0.89 to 0.81, and its mean coverage from 0.35 to
  0.27. For paragraph+BM25 they'd fall from 0.91 to 0.87 and from 0.40 to
  0.33. Heading chunks barely change. Finding 4 rests on all four matched
  pairs, including the two without a breadcrumb, whose prompts would carry
  nothing extra. Finding 3's coverage gains shrink the same way, to +0.04
  (not significant) and +0.08.
- **98 questions give mean intervals of about ±0.08, and ±0.11–0.13 within
  a question style.** Paired differences are much tighter, so gaps of
  0.04–0.07 can be resolved, but not every gap is. heading+bm25 and its
  no-breadcrumb twin are listed in order, yet their paired difference is
  +0.005 Recall@5 (−0.03 to +0.04): they're indistinguishable.
- **The questions are written, not logged.** They follow a sampled
  distribution over the docs, not real user traffic.
- **Holm correction is applied within each metric,** across the eleven
  comparisons with the baseline, not across metrics; the switch-effect
  intervals are uncorrected.
- **The timing claims follow the procedure as described.** "Frozen before
  judging" and "fixed before any results were seen" become independently
  checkable once the work is in git history.

## Next: Stage 3

Dense embeddings, hybrid fusion and reranking, each scored against this
baseline. The targets are the paraphrase questions and the answers no
lexical configuration reaches. Stage 3's new judgments go into `v2.yaml`, so
these numbers stay reproducible from `v1.yaml`.
