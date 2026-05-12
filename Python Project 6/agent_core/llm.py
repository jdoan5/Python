from __future__ import annotations

import os
from functools import lru_cache

import anthropic
from dotenv import load_dotenv

load_dotenv()


@lru_cache(maxsize=1)
def get_client() -> anthropic.Anthropic:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY not set. Add it to your environment or a .env file."
        )
    return anthropic.Anthropic(api_key=api_key)


DEFAULT_MODEL = "claude-opus-4-7"
