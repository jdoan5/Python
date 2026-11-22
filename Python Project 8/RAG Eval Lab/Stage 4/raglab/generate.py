"""The generator's requests (PLAN.md, "What the generator sees").

One request per (condition, question, sample):
- model claude-haiku-5-5, the answer prompt as `system`;
- one user message: one custom-content document per chunk, then the question;
- thinking adaptive, or disabled in the thinking-off arm (accepted only at
  effort high or below);
- output_config.effort.
Nothing else: no temperature, top_p or top_k, no prefill, no tools, no
fallbacks, no cache marker (the answer prompt is too short to cache).
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

from raglab import params
from raglab.blocks import WindowDoc
from raglab.windows import Condition

STAGE = Path(__file__).resolve().parents[1]
PROMPTS = STAGE / "prompts"
NO_THINKING_EFFORTS = ("low", "medium", "high")
FORBIDDEN = ("temperature", "top_p", "top_k", "tools", "tool_choice", "fallbacks", "stream")


def prompt(name: str) -> str:
    return (PROMPTS / name).read_text(encoding="utf-8")


def api_document(doc: WindowDoc, titled: bool) -> dict:
    out = doc.api_document()
    if not titled:
        out.pop("title")
    return out


def generation_body(condition: Condition, question: str, documents: Sequence[WindowDoc], max_tokens: int,
                    answer_prompt: str, closed_prompt: str) -> dict:
    if condition.closed:
        if documents:
            raise ValueError("the closed-book arm carries no documents")
        system, content = closed_prompt, [{"type": "text", "text": question}]
    else:
        if not documents:
            raise ValueError(f"{condition.name}: a window with no documents can't be asked")
        system = answer_prompt
        content = [api_document(d, condition.titles) for d in documents] + [{"type": "text", "text": question}]
    if condition.thinking == "disabled" and condition.effort not in NO_THINKING_EFFORTS:
        raise ValueError("thinking can be disabled only at effort high or below")
    body = {
        "model": params.GENERATOR,
        "max_tokens": int(max_tokens),
        "system": system,
        "messages": [{"role": "user", "content": content}],
        "thinking": {"type": condition.thinking},
        "output_config": {"effort": condition.effort},
    }
    check_generation_body(body)
    return body


def check_generation_body(body: Mapping) -> None:
    """What the API (version 2026-10-21, Haiku 5.5) would refuse, refused here first."""
    if body.get("model") != params.GENERATOR:
        raise ValueError(f"the generator is {params.GENERATOR}")
    bad = [k for k in FORBIDDEN if k in body]
    if bad:
        raise ValueError(f"never sent: {bad}")
    messages = body.get("messages") or []
    if len(messages) != 1 or messages[0].get("role") != "user":
        raise ValueError("one user message, no prefill")
    content = messages[0]["content"]
    docs = [c for c in content if c.get("type") == "document"]
    if docs and not all((c.get("citations") or {}).get("enabled") for c in docs):
        raise ValueError("citations must be enabled on every document or none")
    if any(c.get("type") == "text" and c.get("citations") for c in content):
        raise ValueError("a user text block with citations is refused on API version 2026-10-21")
    if "format" in (body.get("output_config") or {}) and docs:
        raise ValueError("structured outputs can't be combined with citations")
    thinking = (body.get("thinking") or {}).get("type")
    if thinking not in ("adaptive", "disabled"):
        raise ValueError("Haiku 5.5 takes adaptive thinking or disabled thinking only")
    if thinking == "disabled" and (body.get("output_config") or {}).get("effort") not in NO_THINKING_EFFORTS:
        raise ValueError("thinking can be disabled only at effort high or below")
