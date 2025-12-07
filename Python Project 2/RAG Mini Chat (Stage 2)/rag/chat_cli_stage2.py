# rag/chat_cli_stage2.py
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from retrieval import answer_question


def main() -> None:
    print("RAG Mini Chat (Stage 2)")
    print("Ask questions about the documents in data/source.")
    print("Type 'exit', 'quit', or 'q' to leave.\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue

        if question.lower() in {"exit", "quit", "q"}:
            print("Goodbye!")
            break

        answer = answer_question(question)
        print(f"\nAssistant: {answer}\n")


if __name__ == "__main__":
    main()