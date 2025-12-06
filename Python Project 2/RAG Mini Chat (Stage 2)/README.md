# RAG Mini Chat — Stage 2 (Retrieval + LLM)

Stage 2 builds on the Stage 1 “RAG Mini Chat” by adding a **real LLM call** and a
small **retrieval-augmented chat CLI**.

Instead of only returning raw chunks, the app now:

1. Retrieves the most relevant text chunks from `data/source/`.
2. Builds a context block from those chunks.
3. Calls an LLM with that context.
4. Prints a natural-language answer (plus you still control exactly what text
   the model sees).

---

## 1. Project Structure

```text
RAG Mini Chat (Stage 2)/
  data/
    source/
      internal_notes.txt
      product_overview.txt
      support_faq.txt
  index/
    chunks.parquet           # built by build_index.py
  rag/
    __init__.py
    build_index.py           # Stage 2 index builder
    retrieval.py             # search() + answer_question()
    llm_client.py            # thin wrapper around the LLM API (ask_llm)
    chat_cli_stage1.py       # optional: Stage 1 style CLI (no LLM)
    chat_cli_stage2.py       # Stage 2 RAG chat CLI
  requirements.txt
  README.md
  .env                       # NOT committed – holds API key