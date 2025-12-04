# chat_cli_stage2.py

from retrieval import answer_question


def main() -> None:
    print("RAG Mini Chat (Stage 2)")
    print("Ask questions about the documents in data/source.")
    print("Type 'exit', 'quit', or 'q' to leave.\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAssistant: Bye! 👋")
            break

        if not question:
            continue

        if question.lower() in {"exit", "quit", "q"}:
            print("Assistant: Bye! 👋")
            break

        answer = answer_question(question)
        print(f"\nAssistant: {answer}\n")


if __name__ == "__main__":
    main()