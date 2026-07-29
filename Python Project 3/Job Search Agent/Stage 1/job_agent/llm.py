"""Anthropic client setup shared by all pipeline stages."""

from __future__ import annotations

import os
from functools import lru_cache

import anthropic
from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODEL = os.environ.get("JOB_AGENT_MODEL", "claude-opus-4-7")


@lru_cache(maxsize=1)
def get_client() -> anthropic.Anthropic:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Create a .env file in the project root "
            "with: ANTHROPIC_API_KEY=sk-ant-..."
        )
    return anthropic.Anthropic(api_key=api_key)
