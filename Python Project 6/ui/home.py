"""Landing page — explains the apps and the setup state."""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

st.title("AI Agents Portfolio")

st.markdown(
    """
Three Claude-powered agents sharing one tool-use loop. Pick an app from
the sidebar to try it.

- **GitHub Issue Triage** — reads an issue, searches a local repo,
  drafts a maintainer-style reply with file:line citations.
- **Job Application Co-pilot** — scrapes a posting, scores fit against
  your resume, drafts a tailored cover letter, tracks in SQLite.
- **Research Paper Digest** — pulls recent arXiv papers in a category,
  groups by theme, writes critical 2–3 sentence summaries.
"""
)

st.subheader("Setup state")

api_key_set = bool(os.environ.get("ANTHROPIC_API_KEY"))
resume_path = Path.home() / ".job_copilot" / "resume.txt"

col1, col2 = st.columns(2)
with col1:
    if api_key_set:
        st.success("ANTHROPIC_API_KEY is set")
    else:
        st.error("ANTHROPIC_API_KEY is missing")
        st.caption("Add it to `.env` in the project root: `ANTHROPIC_API_KEY=sk-ant-...`")

with col2:
    if resume_path.exists():
        size_kb = resume_path.stat().st_size / 1024
        st.success(f"Resume found ({size_kb:.1f} KB)")
        st.caption(str(resume_path))
    else:
        st.warning("No resume yet — needed for Job Co-pilot")
        st.caption(f"Save a plain-text resume to `{resume_path}`")

st.subheader("How it works")

st.markdown(
    """
Every agent uses the same loop in `agent_core/agent.py`:

1. Call Claude with a cached system prompt and the available tools.
2. If Claude wants to call a tool, execute it and feed the result back.
3. Repeat until Claude returns `stop_reason == "end_turn"`.

Each app contributes a small set of domain tools (search, fetch, save,
list) and a system prompt that defines the agent's job. The shared core
handles prompt caching, adaptive thinking, usage tracking, and the
iteration cap.
"""
)

st.subheader("Cost note")

st.markdown(
    """
Each agent run hits the Anthropic API and costs roughly:

- **Triage**: $0.05–0.15 per issue (depends on repo size).
- **Job Co-pilot**: $0.05–0.10 per posting.
- **Paper Digest**: $0.10–0.30 per digest (10 papers).

Prompt caching brings the second run on the same system prompt down to
~10% of the first. Watch the "Cache hits" metric after each run.
"""
)
