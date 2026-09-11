"""Stage 6 — the Stage 4 portal on Streamlit Community Cloud, for $0.

Community Cloud clones the whole repo and runs this file, so the platform
differences are handled here and ONLY here — Stages 1-4 are untouched:

- Secrets arrive via st.secrets (the app dashboard), not environment
  variables. They are bridged into os.environ before job_agent imports,
  because llm.py and portal-style auth read the environment.
- The job_agent package is not pip-installed from a requirements line
  (Community Cloud installs requirements.txt only); instead "Stage 1" is
  put on sys.path — the pages then import job_agent from source.
- The sample inventory tracked in the repo is the default inventory, so
  the tailor page works the moment the app boots.

Auth model is Stage 4's, unchanged: one shared APP_PASSWORD compared with
hmac.compare_digest, session-scoped. Fails closed when unset.
"""

from __future__ import annotations

import hmac
import os
import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- st.secrets -> os.environ bridge (set BEFORE any job_agent import) ---
# Locally there is usually no secrets.toml; Streamlit raises on access, so
# fall back to plain environment variables (same behavior as Stage 4).
try:
    # ALLOW_UNAUTHENTICATED is deliberately NOT bridged: it is a local-dev
    # escape hatch, and a stray line in the cloud Secrets panel must never
    # be able to silently open the app (and the API key) to the internet.
    for key in ("ANTHROPIC_API_KEY", "APP_PASSWORD"):
        if key in st.secrets and key not in os.environ:
            os.environ[key] = str(st.secrets[key])
except Exception:
    pass  # no secrets file — environment variables rule, as in Stage 4

# This deployment is internet-facing and its password is shared on request, so
# the sidebar filesystem-path boxes are pinned to their defaults (see
# pinned_path() in Stage 2/app.py). setdefault, not a hard assignment, so a
# local `streamlit run` of this file can still opt out with =0.
os.environ.setdefault("JOB_AGENT_LOCK_PATHS", "1")

# job_agent lives in "Stage 1" as source; make it importable for the pages.
stage1 = str(PROJECT_ROOT / "Stage 1")
if stage1 not in sys.path:
    sys.path.insert(0, stage1)

st.set_page_config(page_title="Job Search Agent", page_icon="J", layout="wide")


def gate() -> bool:
    """Stage 4's password gate, verbatim semantics: fail closed."""
    password = os.environ.get("APP_PASSWORD", "")

    if not password:
        if os.environ.get("ALLOW_UNAUTHENTICATED") == "1":
            # Loud, not silent: this state must be unmistakable if it ever
            # appears anywhere but a laptop.
            st.warning("UNAUTHENTICATED MODE — local development only.")
            return True
        st.error(
            "APP_PASSWORD is not set (or is empty) — refusing to serve. "
            "On Community Cloud, add it in the app's Secrets settings."
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
        st.session_state.pop("auth_ok", None)
        st.rerun()

nav = st.navigation(
    {
        "Apps": [
            st.Page(
                str(PROJECT_ROOT / "Stage 2" / "app.py"),
                title="Tailor an application",
                default=True,
            ),
            st.Page(
                str(PROJECT_ROOT / "Stage 3" / "dashboard.py"),
                title="Application tracker",
            ),
        ],
    }
)
nav.run()
