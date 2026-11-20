# Golden set

The questions every stage of RAG Eval Lab is measured against, with graded
judgments of which FastAPI doc sections answer them.

| file | what it is |
|---|---|
| `v1.yaml` | 98 questions with 3,261 judgments `doc_id#anchor: grade`. Labelled against FastAPI 0.165.0; its fingerprint is in the file. The rubric is the `rubric` key. |
| `v1-regrade.json` | The 186 sections first judged from a truncated excerpt and re-judged on their full text: every grade, and the 28 that changed. |
| `v1-agreement.json` | A blind second judging of a stratified sample of v1's labels, so their reliability can be recomputed (`raglab golden agreement`). |
| `v1-unjudged-sample.json` | Sections that some configuration puts in its prompt but v1 leaves unjudged, judged separately to bound what they could change (`Stage 2/analysis/unjudged_sensitivity.py`). |

**Grades.** 2 = the section answers the question on its own. 1 = related:
useful context, but not an answer. 0 = judged and rejected. A section that
appears nowhere is unjudged and scores as not relevant.

**How it was built** (details in [Stage 2's README](../Stage%202/README.md#how-the-golden-set-was-built)):

1. 100 target sections were drawn by a seeded, stratified sample (`raglab golden sample`),
   each with a pre-assigned style: keyword, paraphrase or scenario.
2. One question was written per section. Two sections had nothing to ask and are listed under
   `skipped`. The questions were then frozen; their fingerprint, `c4673ddd…`, is checked by a test.
3. Two pools of sections were judged against each question:
   - every section any of Stage 2's 12 configurations ranks in its top 10, which is the depth the
     rank metrics read;
   - every section on the question's target page. This is a design choice: the target's own page
     is the one place an answer can be found without relying on these retrievers. 60 of the 269
     answers come from this pool alone. Answers elsewhere that no configuration ranks in its top 10
     stay unjudged.
4. The first judging round showed sections cut at 1,800 characters. The 186 longer sections were
   judged again on their full text. Where the two grades differed, a blind tie-break was run and the
   final grade is the median of the three, so a grade changed only when the tie-break agreed with
   the full-text judgment (`v1-regrade.json`).

Judges saw the question, the question's known answers for reference, and the
section's text: full text throughout, except in that first round. They never
saw which configuration had found a section, or how many had. The questions
were written, and the sections judged, by Claude (an LLM) following that
procedure. `v1-agreement.json` measures how consistently: answer versus
not-an-answer κ = 0.92, reweighted to the whole label set.

**Changing it.** `golden validate` checks the file. `golden merge` adds
judgments; it changes an existing grade only with `--regrade`, and never a
target's. `golden pool` writes full-text worksheets for anything unjudged.
**v1 is frozen with Stage 2.** Stage 3's new retrievers will surface sections
nobody judged, and those judgments go into a copy, `v2.yaml`, so Stage 2's
results stay reproducible from `v1.yaml`.
