# llm_client.py
from __future__ import annotations

import os
from dotenv import load_dotenv
from openai import OpenAI

# Load .env so OPENAI_API_KEY is available
load_dotenv()

# Create client (will read OPENAI_API_KEY from env)
client = OpenAI()


def ask_llm(prompt: str) -> str:
    """
    Call the real OpenAI model to answer based on the prompt.
    """
    try:
        response = client.responses.create(
            model="gpt-4.1-mini",   # you can swap to another model if you want
            input=prompt,
        )

        # Extract the text from the first output
        message = response.output[0].content[0].text
        return message

    except Exception as exc:
        # Fallback so your CLI doesn’t crash hard
        return f"[LLM error] {exc}"