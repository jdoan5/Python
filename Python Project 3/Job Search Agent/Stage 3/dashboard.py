"""Stage 3 — application tracking dashboard.

Streamlit dashboard over the tracker DB. Auto-syncs new sidecars from
Stage 1/output/ on load, then offers: funnel metrics, an editable status
board, fit/status charts, and the cross-application gap analysis ("what
skills keep blocking me").

Run from the project root ("Job Search Agent/"):
    .venv/bin/streamlit run "Stage 3/dashboard.py"

State notes (hard-won via review):
- The sync-once flag is keyed to the (db, drafts-dir) pair, so editing either
  path in the sidebar automatically re-syncs against the new pair.
- Save results and sync warnings are stashed in session_state and rendered
  AFTER st.rerun(), because anything written in the same run as a rerun is
  discarded before the user sees it.
- Cleared editor cells arrive as None/NaN; they are normalized to "" so
  deleting a note actually persists (None is the "no change" sentinel).
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# When mounted as a portal page (Stage 4), sys.path holds the portal's
# directory, not this one — make `import tracker` work in both modes.
_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from tracker import (  # noqa: E402
    DEFAULT_DB,
    DEFAULT_OUTPUT_DIR,
    VALID_STATUSES,
    TrackerError,
    gap_frequencies,
    load_applications,
    sync_output_dir,
    update_application,
)

# Same escaping discipline as Stage 2: sidecar content is model-derived.
_MD_SPECIALS = "\\`*_{}[]()#!|~$<>"


def esc(text: object) -> str:
    return "".join("\\" + ch if ch in _MD_SPECIALS else ch for ch in str(text))


def parse_path(raw: str, default: Path) -> Path:
    """User-typed path -> Path, falling back to the default on any error."""
    raw = raw.strip()
    if not raw:
        return default
    try:
        return Path(raw).expanduser()
    except (RuntimeError, ValueError):
        return default


# Standalone: `streamlit run` executes this file with __name__ == "__main__".
# Under the Stage 4 portal, Streamlit sets __name__ to "__page__" and the
# portal owns the page config, so skip it here.
if __name__ == "__main__":
    st.set_page_config(page_title="Application Tracker", page_icon="T", layout="wide")

with st.sidebar:
    st.title("Application Tracker")
    st.caption("Stage 3 — tracking + analytics over approved drafts")
    output_dir = parse_path(
        st.text_input("Drafts directory", value=str(DEFAULT_OUTPUT_DIR)), DEFAULT_OUTPUT_DIR
    )
    db_path = parse_path(
        st.text_input("Tracker database", value=str(DEFAULT_DB)), DEFAULT_DB
    )
    if st.button("Re-scan drafts folder"):
        st.session_state.pop("sync_done", None)

# ----- sync: once per (db, dir) pair, or after explicit re-scan ---------------

sync_key = (str(db_path), str(output_dir))
if st.session_state.get("sync_done") != sync_key:
    if not output_dir.is_dir():
        st.warning(f"Drafts directory does not exist: `{esc(output_dir)}`")
    try:
        added, skipped = sync_output_dir(db_path=db_path, output_dir=output_dir)
    except Exception as e:  # sqlite/OS errors on user-typed paths
        st.error(f"Could not open tracker database: {esc(e)}")
        st.stop()
    st.session_state["sync_done"] = sync_key
    st.session_state["sync_report"] = {"added": added, "skipped": skipped}

report = st.session_state.get("sync_report") or {}
if report.get("added"):
    st.success(f"Ingested {report['added']} new draft(s).")
if report.get("skipped"):
    st.warning(
        f"Skipped unreadable/foreign JSON file(s): {esc(', '.join(report['skipped']))} — "
        "if one was just approved, click *Re-scan drafts folder* (it may have "
        "been mid-write during the scan)."
    )

# Save results stashed before the last rerun (see Save-changes handler).
save_report = st.session_state.pop("save_report", None)
if save_report:
    if save_report.get("updated"):
        st.success(f"Updated {save_report['updated']} application(s).")
    for err in save_report.get("errors", []):
        st.error(esc(err))

try:
    applications = load_applications(db_path=db_path)
except Exception as e:
    st.error(f"Could not read tracker database: {esc(e)}")
    st.stop()

st.title("Job search pipeline")

if not applications:
    st.info(
        "No applications tracked yet. Approve a draft in Stage 2 (or the Stage 1 "
        "CLI), then click **Re-scan drafts folder** in the sidebar."
    )
    st.stop()

# ----- metrics row ------------------------------------------------------------

df = pd.DataFrame(applications)
total = len(df)
in_progress = int(df["status"].isin(["applied", "interviewing"]).sum())
interviews = int((df["status"] == "interviewing").sum())
offers = int((df["status"] == "offer").sum())
avg_fit = df["fit_score"].dropna().mean() if df["fit_score"].notna().any() else None

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Tracked", total)
m2.metric("In progress", in_progress)
m3.metric("Interviewing", interviews)
m4.metric("Offers", offers)
m5.metric("Avg fit", f"{avg_fit:.1f}/10" if avg_fit is not None else "—")

# ----- editable status board --------------------------------------------------

st.subheader("Applications")
board = df[["id", "company", "role", "fit_score", "status", "notes", "created_at"]].copy()

edited = st.data_editor(
    board,
    width="stretch",
    hide_index=True,
    num_rows="fixed",
    disabled=["id", "company", "role", "fit_score", "created_at"],
    column_config={
        "id": st.column_config.NumberColumn("ID", width="small"),
        "fit_score": st.column_config.NumberColumn("Fit", width="small"),
        # required=True: the status selectbox cannot be cleared to None.
        "status": st.column_config.SelectboxColumn(
            "Status", options=VALID_STATUSES, required=True
        ),
        "notes": st.column_config.TextColumn("Notes"),
        "created_at": st.column_config.TextColumn("Created", width="medium"),
    },
    key="board_editor",
)

if st.button("Save changes", type="primary"):
    updated, errors = 0, []
    original = board.set_index("id")
    for _, row in edited.iterrows():
        app_id = int(row["id"])
        orig = original.loc[app_id]
        new_status = row["status"] if row["status"] != orig["status"] else None
        # A cleared notes cell arrives as None/NaN — normalize to "" so the
        # deletion persists ("" is a real value; None means "don't touch").
        notes_val = "" if pd.isna(row["notes"]) else str(row["notes"])
        new_notes = notes_val if notes_val != (orig["notes"] or "") else None
        if new_status is None and new_notes is None:
            continue
        try:
            update_application(app_id, status=new_status, notes=new_notes, db_path=db_path)
            updated += 1
        except (TrackerError, sqlite3.Error) as e:
            errors.append(f"#{app_id}: {e}")
    if updated or errors:
        # Stash for the post-rerun render — messages written here would be
        # discarded by st.rerun() before the user could read them.
        st.session_state["save_report"] = {"updated": updated, "errors": errors}
        st.rerun()
    else:
        st.info("No changes to save.")

# ----- charts -----------------------------------------------------------------

col_status, col_fit = st.columns(2)
with col_status:
    st.subheader("Status breakdown")
    status_counts = df["status"].value_counts()
    # Known statuses in lifecycle order first, then anything unexpected —
    # a row must never silently vanish from the chart.
    order = [s for s in VALID_STATUSES if s in status_counts.index] + [
        s for s in status_counts.index if s not in VALID_STATUSES
    ]
    st.bar_chart(status_counts.reindex(order))
with col_fit:
    st.subheader("Fit score distribution")
    fit_counts = (
        df["fit_score"].dropna().astype(int).value_counts().sort_index()
    )
    if not fit_counts.empty:
        st.bar_chart(fit_counts)
    else:
        st.caption("No fit scores recorded.")

# ----- gap analysis -----------------------------------------------------------

st.subheader("Recurring gaps — what to learn next")
st.caption(
    "Requirements that keep showing up with no supporting evidence in your "
    "inventory, ranked by how many postings asked for them."
)
gaps = gap_frequencies(applications)
if gaps:
    for gap_text, count in gaps:
        st.markdown(f"- **{count}×** {esc(gap_text)}")
else:
    st.caption("No gaps recorded — either strong fits so far, or too few data points.")

# ----- draft previews ---------------------------------------------------------

with st.expander("Preview a saved draft"):
    # Keyed by unique id so duplicate company/role pairs can't collide.
    options = {
        f"#{app['id']} {app['company']} — {app['role']}": app["sidecar_path"]
        for app in applications
    }
    choice = st.selectbox("Application", options=list(options.keys()))
    if choice:
        md_path = Path(options[choice]).with_suffix(".md")
        if md_path.is_file():
            try:
                st.code(md_path.read_text(encoding="utf-8"), language="markdown")
            except OSError as e:
                st.warning(f"Could not read draft: {esc(e)}")
        else:
            st.caption("Markdown draft not found next to the sidecar.")
