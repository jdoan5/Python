# rag/llm_client.py
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI  # new OpenAI Python SDK


# --- Load .env --------------------------------------------------------------

HERE = Path(__file__).resolve().parent
DOTENV_PATH = HERE / ".env"

# Load variables from rag/.env
load_dotenv(DOTENV_PATH)

raw_key = os.getenv("OPENAI_API_KEY")
model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

print(f"DEBUG (llm_client): .env path used: {DOTENV_PATH}")
print(f"DEBUG (llm_client): raw OPENAI_API_KEY present? {bool(raw_key)}")
print(f"DEBUG (llm_client): OPENAI_MODEL = {model!r}")

# --- Create client or fall back to stub ------------------------------------

client: OpenAI | None

if raw_key:
    client = OpenAI(api_key=raw_key)
    print("DEBUG (llm_client): Created real OpenAI client.")
else:
    client = None
    print("DEBUG (llm_client): No API key – using stub answers.")


def ask_llm(prompt: str) -> str:
    """
    Send a prompt to OpenAI if we have a client.
    Otherwise return a stub answer (so the rest of the app still works).
    """
    global client

    print(f"DEBUG (llm_client.ask_llm): client is None? {client is None}")

    # Fallback path (no key / misconfig)
    if client is None:
        return "Stub answer (no API key):\n" + prompt

    # Real API call
    try:
        response = client.responses.create(
            model=model,
            input=prompt,
        )
        # Responses API: first output → first content block → text
        return response.output[0].content[0].text
    except Exception as e:
        # In case of quota issues or any other error, show something useful
        return f"[LLM error] {e}"