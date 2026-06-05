"""Streamlit page for the Research Paper Digest agent."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from apps.paper_digest.agent import build_agent
from apps.paper_digest.tools import VALID_CATEGORIES
from ui._common import render_result, require_api_key

require_api_key()

DEFAULT_DIR = Path.home() / ".paper_digest"

st.title("Research Paper Digest")
st.caption(
    "Pick an arXiv category. The agent pulls the most recent submissions, "
    "groups them by theme, writes critical 2–3 sentence summaries, and "
    "saves the digest as markdown."
)

with st.form("digest_form"):
    col1, col2, col3 = st.columns([2, 1, 2])
    with col1:
        category = st.selectbox(
            "arXiv category",
            options=sorted(VALID_CATEGORIES),
            index=sorted(VALID_CATEGORIES).index("cs.AI") if "cs.AI" in VALID_CATEGORIES else 0,
        )
    with col2:
        count = st.number_input("Papers", min_value=3, max_value=25, value=10)
    with col3:
        out_dir = st.text_input("Output dir", value=str(DEFAULT_DIR))

    submitted = st.form_submit_button("Generate digest", type="primary")

if submitted:
    out_path = Path(out_dir).expanduser()
    out_path.mkdir(parents=True, exist_ok=True)

    agent = build_agent(digest_dir=out_path, verbose=False)

    with st.status(
        f"Fetching {count} recent {category} papers and writing the digest...",
        expanded=False,
    ) as status:
        try:
            result = agent.run(
                f"Produce a weekly digest for arXiv category {category}. "
                f"Fetch about {count} recent papers, then save the digest."
            )
            status.update(label="Done", state="complete")
        except Exception as e:
            status.update(label="Failed", state="error")
            st.exception(e)
            st.stop()

    render_result(result, output_label="Digest")

st.divider()
st.subheader("Saved digests")

out_path = Path(out_dir if "out_dir" in dir() else DEFAULT_DIR).expanduser()
if out_path.exists():
    files = sorted(out_path.glob("digest_*.md"), reverse=True)[:20]
    if files:
        selected = st.selectbox("Open a past digest", options=[f.name for f in files])
        if selected:
            content = (out_path / selected).read_text(encoding="utf-8")
            st.markdown(content)
    else:
        st.info("No digests saved yet.")
else:
    st.info(f"Output directory does not exist yet: {out_path}")
