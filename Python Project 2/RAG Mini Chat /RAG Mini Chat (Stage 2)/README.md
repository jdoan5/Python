# RAG Mini Chat — Stage 2 (Retrieval + LLM)

Small Retrieval-Augmented Generation (RAG) demo.

Stage 2 builds on Stage 1 by adding:

- A simple text **index** built from `.txt` files in `data/source/`.
- A `search()` function that pulls relevant chunks for a question.
- An `llm_client` that can either:
  - Call the **OpenAI API** when an `OPENAI_API_KEY` is available, or
  - Fall back to a **local stub** that just echoes the prompt (useful for offline testing).

---

## Project layout

```text
RAG Mini Chat (Stage 2)/
├─ data/
│  └─ source/
│     ├─ product_overview.txt
│     ├─ support_faq.txt
│     └─ Special_Order_Overview.txt
├─ index/
│  └─ chunks.parquet          # created by build_index2.py
└─ rag/
   ├─ build_index2.py         # build text index from data/source
   ├─ retrieval.py            # search() + answer_question()
   ├─ llm_client.py           # ask_llm(): OpenAI or local stub
   ├─ chat_cli_stage2.py      # command-line chat loop
   └─ __init__.py

```
## Data → Index → Chat (Overview)

```mermaid
flowchart LR
  A["Text files in data/source"] --> B["build_index2.py (create chunks.parquet)"]
  B --> C["search() in retrieval.py (find relevant chunks)"]
  C --> D["ask_llm() in llm_client.py (call OpenAI or local stub)"]
  D --> E["chat_cli_stage2.py (CLI conversation)"]