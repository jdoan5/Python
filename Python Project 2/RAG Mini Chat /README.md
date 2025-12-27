# RAG Mini Chat (Stages 1–3)

A staged Retrieval-Augmented Generation (RAG) mini-chat project that progresses from a single-document baseline to a multi-document workflow, then adds logging/polish to make answers easier to debug and evaluate.

## TL;DR (What to review)
- **Stage 1:** working baseline (single-doc retrieval → grounded answer)
- **Stage 2:** multi-doc retrieval (better coverage + top-k context selection)
- **Stage 3:** logging & polish (repeatable runs, visibility into retrieval/context)

---

## Stages

### Stage 1 — Single-Doc RAG
**Goal:** Establish a correct baseline end-to-end.  
**Includes:**
- Document chunking + basic retrieval
- Prompting that uses retrieved context
- Simple Q&A loop to confirm grounding

**Review for:** does retrieval return relevant chunks and does the answer stay within that context?

---

### Stage 2 — Multi-Doc RAG
**Goal:** Extend the pipeline to support multiple documents/sources.  
**Includes:**
- Indexing multiple docs
- Top-k retrieval across sources
- More robust “context packing” for prompts

**Review for:** does it retrieve from the right source(s) and improve coverage vs Stage 1?

---

### Stage 3 — Logging & Polish
**Goal:** Make the system easier to evaluate and troubleshoot.  
**Includes:**
- Run logs (query, retrieved chunks, selected context, final answer)
- Clearer output formatting
- More consistent, reproducible run steps

**Review for:** can a reviewer quickly see *why* the model answered the way it did?

---

## How to run (high level)
1. Run Stage 1 to confirm baseline behavior.
2. Run Stage 2 and compare retrieval/context vs Stage 1.
3. Run Stage 3 and inspect logs to validate grounding + debugability.

> Each stage folder contains its own README with exact commands and expected outputs.

---

## Notes
- This is a learning/prototype project focused on clear workflow and evaluation signals.
- Emphasis is on correctness, grounding, and debuggability rather than production deployment.