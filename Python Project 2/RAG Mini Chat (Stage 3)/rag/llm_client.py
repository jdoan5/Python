# llm_client.py  (local-only version, no OpenAI calls)

from __future__ import annotations
import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env just in case you want other settings later
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# Always local: no OpenAI client
client = None


def ask_llm(prompt: str, cfg: dict | None = None) -> str:
    """
    Small wrapper around the LLM.
    cfg is optional (used in Stage 3 for model / temperature tweaks).
    """

    if os.getenv("DEBUG_LLM", "0") == "1":
        print("DEBUG (llm_client.ask_llm): using LOCAL stub, len(prompt) =", len(prompt))

    # You can format this however you like
    return (
        "Stub answer (local mode).\n\n"
        "I would answer based on this prompt/context:\n\n"
        f"{prompt}"
    )