"""Streamlit page for the GitHub Issue Triage agent."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from apps.github_triage.agent import build_agent
from ui._common import render_result, require_api_key

require_api_key()

st.title("GitHub Issue Triage")
st.caption(
    "Paste an issue and point the agent at a local repo. It searches the "
    "codebase, reads relevant files, classifies the issue, and drafts a reply."
)

with st.form("triage_form"):
    issue_text = st.text_area(
        "Issue (title + body)",
        height=200,
        placeholder=(
            "Title: search_codebase returns no matches\n\n"
            "Body: I called search_codebase with a string I know exists in the repo "
            "and got 'No matches'. Tried with both 'def' and 'class' — nothing."
        ),
    )
    col1, col2 = st.columns([3, 1])
    with col1:
        repo_path = st.text_input(
            "Repository path (absolute or relative)",
            value=str(Path.cwd()),
        )
    with col2:
        max_iters = st.number_input("Max iterations", min_value=3, max_value=30, value=15)

    submitted = st.form_submit_button("Triage", type="primary")

if submitted:
    repo = Path(repo_path).expanduser().resolve()
    if not issue_text.strip():
        st.error("Issue text is empty.")
        st.stop()
    if not repo.is_dir():
        st.error(f"Not a directory: {repo}")
        st.stop()

    agent = build_agent(repo_root=repo, verbose=False)
    agent.max_iterations = int(max_iters)

    with st.status("Agent is searching the repo and drafting a reply...", expanded=False) as status:
        try:
            result = agent.run(issue_text)
            status.update(label="Done", state="complete")
        except Exception as e:
            status.update(label="Failed", state="error")
            st.exception(e)
            st.stop()

    render_result(result, output_label="Triage")
