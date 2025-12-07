# chat_cli_stage3.py
from pathlib import Path
import json
import datetime

from retrieval import search
from llm_client import ask_llm
import yaml   # pip install pyyaml

CONFIG_PATH = Path(__file__).parent / "config.yml"
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

def load_config():
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def log_turn(session_id: str, question: str, answer: str, hits: list, cfg: dict):
    log_rec = {
        "ts": datetime.datetime.now().isoformat(),
        "session_id": session_id,
        "question": question,
        "answer": answer,
        "hits": hits,        # doc_id, chunk_id, text, score
        "retrieval_cfg": cfg.get("retrieval", {}),
    }
    log_file = LOG_DIR / f"{session_id}.jsonl"
    with log_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(log_rec) + "\n")

def main():
    cfg = load_config()
    session_id = datetime.datetime.now().strftime("rag3_%Y%m%d_%H%M%S")

    print("RAG Mini Chat (Stage 3)")
    print("Ask questions about the documents in data/source.")
    print("Commands: :config, :quit")
    print()

    while True:
        question = input("You: ").strip()
        if not question:
            continue
        if question.lower() in {"exit", "quit", "q", ":q"}:
            print("Goodbye!")
            break
        if question == ":config":
            print("Current config:", json.dumps(cfg, indent=2))
            continue

        # --- 1) retrieval
        k = cfg["retrieval"]["k"]
        hits = search(question, k=k)

        if not hits:
            answer = "I couldn’t find anything in the documents for that question yet."
            print(f"Assistant: {answer}\n")
            log_turn(session_id, question, answer, [], cfg)
            continue

        # --- 2) build context
        max_chars = cfg["retrieval"]["max_context_chars"]
        context_parts = []
        total = 0
        for i, h in enumerate(hits, start=1):
            snippet = f"[{i}] ({h['doc_id']}#chunk{h['chunk_id']})\n{h['text']}\n"
            if total + len(snippet) > max_chars:
                break
            context_parts.append(snippet)
            total += len(snippet)
        context_block = "\n".join(context_parts)

        # --- 3) ask LLM
        prompt = (
            "Use ONLY the context below to answer.\n\n"
            f"{context_block}\n\n"
            f"Question: {question}\n"
            "Answer (be concise and grounded in the snippets):"
        )
        answer = ask_llm(prompt, cfg)   # you can ignore cfg inside if you want

        # --- 4) display
        print(f"\nAssistant: {answer}\n")
        if cfg["display"]["show_sources"]:
            print("Sources:")
            for i, h in enumerate(hits, start=1):
                line = f"  [{i}] {h['doc_id']} (chunk {h['chunk_id']})"
                if cfg["display"]["show_scores"] and "score" in h:
                    line += f" — score={h['score']:.3f}"
                print(line)
            print()

        # --- 5) log
        log_turn(session_id, question, answer, hits, cfg)

if __name__ == "__main__":
    main()
