"""Helpers shared across Streamlit pages."""

from __future__ import annotations

import streamlit as st

from agent_core.agent import AgentResult
from agent_core.llm import get_client


def require_api_key() -> None:
    """Stop the page with a friendly error if ANTHROPIC_API_KEY is missing."""
    try:
        get_client()
    except RuntimeError as e:
        st.error(str(e))
        st.info(
            "Add your key to a `.env` file in the project root: "
            "`ANTHROPIC_API_KEY=sk-ant-...`"
        )
        st.stop()


def render_result(result: AgentResult, output_label: str = "Output") -> None:
    """Render an AgentResult with output, stats, and a collapsible tool-call log."""
    st.subheader(output_label)
    st.markdown(result.final_text)

    cols = st.columns(4)
    cols[0].metric("Iterations", result.iterations)
    cols[1].metric("Tool calls", len(result.tool_calls))
    cols[2].metric(
        "Output tokens", result.usage.get("output_tokens", 0)
    )
    cols[3].metric(
        "Cache hits", result.usage.get("cache_read_input_tokens", 0)
    )

    if result.tool_calls:
        with st.expander(f"Tool calls ({len(result.tool_calls)})"):
            for i, call in enumerate(result.tool_calls, 1):
                status = "error" if call["error"] else "ok"
                st.markdown(f"**{i}. `{call['name']}` — {status}**")
                st.json(call["input"])
                output_preview = call["output"]
                if len(output_preview) > 2000:
                    output_preview = output_preview[:2000] + "\n... (truncated)"
                st.code(output_preview, language="text")
