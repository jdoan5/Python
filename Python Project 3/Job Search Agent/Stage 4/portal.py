"""Stage 4 — deployable portal: auth gate + both apps behind one port.

Wraps Stage 2 (tailor) and Stage 3 (tracker) as pages of a single Streamlit
app, behind a shared-password gate. This is the process a container runs.

Run locally from the project root ("Job Search Agent/"):
    APP_PASSWORD=changeme .venv/bin/streamlit run "Stage 4/portal.py"

Auth model — deliberate scope: a single shared password (APP_PASSWORD env
var) compared with hmac.compare_digest, session-scoped via st.session_state.
That is appropriate for a personal tool whose only secret-burning user is
you. It is NOT multi-user auth: no accounts, no rate limiting, no lockout.
If APP_PASSWORD is unset the portal refuses to serve (fail closed) unless
ALLOW_UNAUTHENTICATED=1 is set explicitly for local development.
"""

from __future__ import annotations

import hmac
import os
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STAGE2_APP = PROJECT_ROOT / "Stage 2" / "app.py"
STAGE3_APP = PROJECT_ROOT / "Stage 3" / "dashboard.py"

st.set_page_config(page_title="Job Search Agent", page_icon="J", layout="wide")


def gate() -> bool:
    """Password gate. Returns True when the session may proceed."""
    password = os.environ.get("APP_PASSWORD", "")

    if not password:
        if os.environ.get("ALLOW_UNAUTHENTICATED") == "1":
            return True  # explicit local-dev opt-out
        st.error(
            "APP_PASSWORD is not set (or is empty) — refusing to serve. "
            "Set the APP_PASSWORD environment variable to a non-empty value "
            "(or ALLOW_UNAUTHENTICATED=1 for local development only)."
        )
        st.stop()

    if st.session_state.get("auth_ok"):
        return True

    st.title("Job Search Agent")
    with st.form("login"):
        entered = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in")
    if submitted:
        if hmac.compare_digest(entered.encode("utf-8"), password.encode("utf-8")):
            st.session_state["auth_ok"] = True
            st.rerun()
        st.error("Wrong password.")
    return False


if not gate():
    st.stop()

with st.sidebar:
    if st.button("Sign out"):
        # Pop only the auth flag: a draft sitting unapproved in Stage 2's
        # session state must survive an accidental sign-out (single-user
        # tool — the next sign-in is the same person).
        st.session_state.pop("auth_ok", None)
        st.rerun()

nav = st.navigation(
    {
        "Apps": [
            st.Page(str(STAGE2_APP), title="Tailor an application", default=True),
            st.Page(str(STAGE3_APP), title="Application tracker"),
        ],
    }
)
nav.run()
