"""Streamlit page for the Job Application Co-pilot agent."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st

from apps.job_copilot.agent import build_agent
from ui._common import render_result, require_api_key

require_api_key()

DEFAULT_RESUME = Path.home() / ".job_copilot" / "resume.txt"
DEFAULT_DB = Path.home() / ".job_copilot" / "applications.db"

st.title("Job Application Co-pilot")

tab_apply, tab_history = st.tabs(["Draft application", "History"])

with tab_apply:
    st.caption(
        "Paste a job posting URL. The agent fetches it, reads your resume, "
        "scores fit honestly, and drafts a 3-paragraph cover letter."
    )

    with st.form("apply_form"):
        url = st.text_input("Job posting URL", placeholder="https://jobs.example.com/postings/12345")
        resume_path = st.text_input("Resume path", value=str(DEFAULT_RESUME))
        db_path = st.text_input("Tracker DB path", value=str(DEFAULT_DB))
        submitted = st.form_submit_button("Draft", type="primary")

    if submitted:
        resume = Path(resume_path).expanduser()
        db = Path(db_path).expanduser()
        if not url.startswith(("http://", "https://")):
            st.error("URL must start with http:// or https://")
            st.stop()
        if not resume.exists():
            st.error(f"Resume not found at {resume}")
            st.info(f"Save your resume as plain text to `{resume}` and try again.")
            st.stop()

        resume.parent.mkdir(parents=True, exist_ok=True)
        db.parent.mkdir(parents=True, exist_ok=True)

        agent = build_agent(resume_path=resume, db_path=db, verbose=False)

        with st.status("Fetching posting, scoring fit, drafting letter...", expanded=False) as status:
            try:
                result = agent.run(
                    f"Please evaluate this job posting and draft a cover letter: {url}"
                )
                status.update(label="Done", state="complete")
            except Exception as e:
                status.update(label="Failed", state="error")
                st.exception(e)
                st.stop()

        render_result(result, output_label="Application Draft")

with tab_history:
    db_for_history = Path(
        st.text_input("Tracker DB path", value=str(DEFAULT_DB), key="hist_db")
    ).expanduser()

    if not db_for_history.exists():
        st.info("No applications recorded yet. Draft one in the other tab first.")
    else:
        with sqlite3.connect(db_for_history) as conn:
            df = pd.read_sql_query(
                "SELECT id, company, role, fit_score, status, created_at, notes "
                "FROM applications ORDER BY created_at DESC",
                conn,
            )

        if df.empty:
            st.info("No applications recorded yet.")
        else:
            col1, col2, col3 = st.columns(3)
            col1.metric("Total applications", len(df))
            col2.metric("Avg fit score", f"{df['fit_score'].mean():.1f}/10")
            col3.metric(
                "Strong fits (>=7)", int((df["fit_score"] >= 7).sum())
            )

            status_filter = st.multiselect(
                "Filter by status",
                options=sorted(df["status"].unique().tolist()),
                default=[],
            )
            if status_filter:
                df = df[df["status"].isin(status_filter)]

            st.dataframe(df, use_container_width=True, hide_index=True)
