# RAG Mini Bot — Stage 1 (Local Retrieval Only or local TF-IDF search)

Small “retrieval-first” helper that searches over local text files and returns the
most relevant passages. No LLM yet — this stage is all about understanding the
**R**etrieval part of RAG. 

What Stage 1 Does
	1.	You drop .txt documents into data/source/.
	2.	build_index.py:
	•	reads all .txt files
	•	splits text into small chunks
	•	builds a TF-IDF index
	•	saves chunks + index into the index/ folder
	3.	chat_cli.py:
	•	takes your question from the terminal
	•	encodes it with the same TF-IDF vectorizer
	•	finds the most similar chunk(s) from your documents
	•	prints the best-matching passage and its source file

No language model, no generation — just retrieval. This keeps the mechanics
transparent and easy to debug.
---

## Project Structure

```text
rag_mini_bot/  (or RAG Mini Chat/)
├── data/
│   └── source/           # Your input .txt files live here
│       ├── faq.txt
│       └── notes_project.txt
├── index/                # Created by build_index.py (saved index files)
│   ├── chunks.parquet    # or chunks.csv – text chunks + metadata
│   ├── tfidf_matrix.joblib
│   └── vectorizer.joblib
├── build_index.py        # Reads source docs, chunks text, builds TF-IDF index
├── chat_cli.py           # Simple terminal “chat” that does retrieval
└── requirements.txt

```
## Architecture Diagram (Mermaid)

```mermaid
flowchart TD
    A[Text files in data/source] --> B[build_index.py]
    B --> C[Chunks + TF-IDF index in index/]
    D[User question in chat_cli.py] --> E[Encode with TF-IDF vectorizer]
    E --> F[Similarity search over chunks]
    F --> G[Return top matching passage + source]

