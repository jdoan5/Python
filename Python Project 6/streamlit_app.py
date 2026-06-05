"""Multi-page Streamlit entrypoint for the AI Agents portfolio.

Run with:
    streamlit run streamlit_app.py

Pages are defined in ui/. Each page is a standalone module whose top-level
code is executed every time the user navigates to it.
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="AI Agents Portfolio",
    page_icon="A",
    layout="wide",
    initial_sidebar_state="expanded",
)

PAGES = {
    "Overview": [
        st.Page("ui/home.py", title="Home", default=True),
    ],
    "Agents": [
        st.Page("ui/github_triage.py", title="GitHub Issue Triage"),
        st.Page("ui/job_copilot.py", title="Job Application Co-pilot"),
        st.Page("ui/paper_digest.py", title="Research Paper Digest"),
    ],
}

nav = st.navigation(PAGES)
nav.run()
