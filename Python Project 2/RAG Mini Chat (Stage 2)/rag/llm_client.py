# llm_client.py – simple wrapper so other code can call the LLM
from __future__ import annotations

import os
from textwrap import shorten

try:
    # Optional: if you have OpenAI installed, you can use it
    from openai import OpenAI
    _HAS_OPENAI = True
except ImportError:
    OpenAI = None
    _HAS_OPENAI = False


def _call_openai(prompt: str) -> str:
    """
    Real call to OpenAI if OPENAI_API_KEY is set and openai is installed.
    Otherwise we fall back to a stub response.
    """
    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key or not _HAS_OPENAI:
        # Fallback: stub so the app still works for learning/demo
        snippet = shorten(prompt, width=180, placeholder="...")
        return f"[LLM stub] No API key / SDK. I would answer based on this prompt:\n{snippet}"

    client = OpenAI(api_key=api_key)

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a concise, helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=300,
    )
    return resp.choices[0].message.content.strip()


def ask_llm(prompt: str) -> str:
    """
    Public helper used by retrieval.answer_question().
    """
    return _call_openai(prompt)