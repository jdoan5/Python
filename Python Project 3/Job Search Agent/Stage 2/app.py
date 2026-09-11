"""Stage 2 — browser UI for the Job Search Agent.

Streamlit front-end over the Stage 1 pipeline. Same four stages, same
grounding guarantees; the human-in-the-loop gate becomes explicit
Approve / Reject buttons instead of a terminal y/n prompt.

Run from the project root ("Job Search Agent/"):
    .venv/bin/streamlit run "Stage 2/app.py"

Architecture note: the CLI passes a blocking `approve` callback into
run_pipeline(). That shape can't work in Streamlit's rerun model, so this app
calls the stage functions directly — extract_requirements -> gather_evidence
-> draft_application — stores the result in st.session_state, renders the
review screen, and only calls save_draft() when the Approve button fires on a
later rerun. Nothing touches disk before that click.

Security note: model output is derived from an untrusted job posting, so
every model-derived string is markdown-escaped before rendering. Unescaped
st.markdown would let a hostile posting inject e.g. image tags whose URLs the
browser fetches automatically (zero-click exfiltration of draft content).
"""

from __future__ import annotations

import os
from pathlib import Path

import anthropic
import pandas as pd
import streamlit as st

from job_agent.fetcher import FetchError, fetch_posting
from job_agent.llm import DEFAULT_MODEL
from job_agent.pipeline import (
    GroundingError,
    PipelineError,
    draft_application,
    extract_requirements,
    gather_evidence,
    save_draft,
)
from job_agent.retrieval import Bm25Index, InventoryError, load_inventory

STAGE1_ROOT = Path(__file__).resolve().parent.parent / "Stage 1"
# Env overrides let a container point these at a mounted volume (Stage 4).
DEFAULT_INVENTORY = Path(
    os.environ.get("JOB_AGENT_INVENTORY", str(STAGE1_ROOT / "data" / "experience_inventory.yaml"))
)
DEFAULT_OUTPUT = Path(os.environ.get("JOB_AGENT_OUTPUT_DIR", str(STAGE1_ROOT / "output")))

# Deployed builds (Stage 4 container, Stage 6 cloud) set this. The sidebar path
# boxes then render read-only AND any submitted value is discarded here, on the
# server: `disabled=True` is a client-side hint a crafted websocket frame can
# ignore, so the pinning below is what actually confines the filesystem.
PATHS_LOCKED = os.environ.get("JOB_AGENT_LOCK_PATHS", "0") == "1"


def pinned_path(raw: str, default: Path) -> Path:
    """User-typed path -> Path; forced to `default` when paths are locked."""
    if PATHS_LOCKED:
        return default
    raw = raw.strip()
    if not raw:
        return default
    try:
        return Path(raw).expanduser()
    except (RuntimeError, ValueError):
        return default

# Markdown specials that must be neutralized in model/posting-derived text.
_MD_SPECIALS = "\\`*_{}[]()#!|~$<>"


def esc(text: object) -> str:
    """Backslash-escape markdown so model-derived text renders as plain text."""
    return "".join("\\" + ch if ch in _MD_SPECIALS else ch for ch in str(text))


# Standalone: `streamlit run` executes this file with __name__ == "__main__".
# Mounted as a page in the Stage 4 portal, Streamlit sets __name__ to
# "__page__" and the portal owns the page config, so skip it here.
if __name__ == "__main__":
    st.set_page_config(page_title="Job Search Agent", page_icon="J", layout="wide")


# ----- sidebar: configuration ------------------------------------------------

with st.sidebar:
    st.title("Job Search Agent")
    st.caption("Stage 2 — browser UI over the Stage 1 pipeline")

    inventory_raw = st.text_input(
        "Experience inventory", value=str(DEFAULT_INVENTORY), disabled=PATHS_LOCKED
    )
    inventory_path = pinned_path(inventory_raw, DEFAULT_INVENTORY)

    output_raw = st.text_input(
        "Output directory", value=str(DEFAULT_OUTPUT), disabled=PATHS_LOCKED
    )
    output_dir = pinned_path(output_raw, DEFAULT_OUTPUT)

    model = st.text_input("Model", value=DEFAULT_MODEL).strip() or DEFAULT_MODEL

    st.divider()
    if os.environ.get("ANTHROPIC_API_KEY"):
        st.success("API key loaded")
    else:
        st.error("ANTHROPIC_API_KEY missing")
        st.caption(
            "Set the ANTHROPIC_API_KEY environment variable "
            "(running locally: add it to `.env` in the project root), then restart."
        )

    try:
        _entries_preview = load_inventory(inventory_path)
        st.success(f"Inventory: {len(_entries_preview)} entries")
    except InventoryError as e:
        st.error("Inventory problem")
        st.caption(esc(e))

st.title("Tailor an application")
st.caption(
    "Extract requirements → retrieve evidence from your real experience → "
    "draft grounded bullets → **you approve before anything is saved**."
)


# ----- input: paste text or fetch URL ----------------------------------------
# Streamlit renders BOTH tabs' widgets every run, so both fields can hold
# values at once. Precedence: pasted text wins, and the recorded source must
# follow the same rule so saved drafts are never misattributed.

tab_paste, tab_url = st.tabs(["Paste posting text", "Fetch from URL"])

with tab_paste:
    pasted = st.text_area(
        "Job posting text",
        height=260,
        placeholder="Paste the full posting here (most reliable — job boards often block bots).",
    )

with tab_url:
    url = st.text_input("Posting URL", placeholder="https://jobs.example.com/backend-123")

posting_text = pasted.strip()
url = url.strip()
if posting_text:
    source_label = "pasted text"
elif url:
    source_label = url
else:
    source_label = ""

if posting_text and url:
    st.warning(
        "Both inputs are filled — the **pasted text** will be analyzed; "
        "the URL will be ignored. Clear the paste box to fetch the URL instead."
    )

analyze = st.button("Analyze posting", type="primary", disabled=not source_label)


# ----- run stages 1-3 on click ----------------------------------------------

if analyze:
    # NOTE: do NOT invalidate the previous analysis here. If this run fails at
    # any point, the prior draft (possibly awaiting approval) must survive.

    try:
        index = Bm25Index(load_inventory(inventory_path))
    except InventoryError as e:
        st.error(f"Inventory error: {esc(e)}")
        st.stop()

    if not posting_text and url:
        try:
            with st.status("Fetching posting...", expanded=False):
                posting_text = fetch_posting(url)
        except FetchError as e:
            # Rendered OUTSIDE the status container so it's actually visible.
            st.error(f"Fetch failed: {esc(e)}")
            st.info("Tip: use the *Paste posting text* tab instead — it always works.")
            st.stop()

    try:
        with st.status("Stage 1 — extracting requirements...", expanded=False):
            requirements = extract_requirements(posting_text, model=model)

        with st.status("Stage 2 — gathering evidence (agentic)...", expanded=True) as status:
            def log_search(query: str, hits: int) -> None:
                st.write(f"searched `{esc(query)}` — {hits} hit(s)")

            evidence = gather_evidence(requirements, index, model=model, on_search=log_search)
            status.update(label=f"Stage 2 — {len(evidence.entries)} evidence entries retrieved",
                          state="complete")

        with st.status("Stage 3 — drafting application...", expanded=False):
            draft = draft_application(requirements, evidence, model=model)

    except GroundingError as e:
        st.error(f"Draft rejected by the grounding validator: {esc(e)}")
        st.info(
            "The model cited experience that retrieval never returned — the "
            "anti-fabrication check caught it. Click *Analyze posting* to retry."
        )
        st.stop()
    except PipelineError as e:
        st.error(f"Pipeline error: {esc(e)}")
        st.stop()
    except anthropic.RateLimitError:
        st.error("Rate limited by the API — wait a minute and retry.")
        st.stop()
    except anthropic.AuthenticationError:
        st.error("Invalid API key. Check ANTHROPIC_API_KEY in .env and restart.")
        st.stop()
    except anthropic.APIError as e:
        st.error(f"API error: {esc(e)}")
        st.stop()
    except Exception as e:  # last resort: keep the app alive, keep prior state
        st.error(f"Unexpected error ({type(e).__name__}): {esc(e)}")
        st.stop()

    # Success — only now supersede any previous analysis / saved marker.
    st.session_state.pop("saved_path", None)
    st.session_state["analysis"] = {
        "requirements": requirements,
        "evidence": evidence,
        "draft": draft,
        "source": source_label,
    }


# ----- review screen + the HITL gate -----------------------------------------

if st.session_state.pop("rejected", False):
    st.warning("Draft rejected — nothing was written to disk.")

analysis = st.session_state.get("analysis")

if analysis and not st.session_state.get("saved_path"):
    requirements = analysis["requirements"]
    evidence = analysis["evidence"]
    draft = analysis["draft"]

    st.divider()
    st.subheader(f"{esc(requirements.role)} @ {esc(requirements.company)}")
    st.caption(f"Seniority: {esc(requirements.seniority)} · Source: {esc(analysis['source'])}")

    score_icon = "🟢" if draft.fit_score >= 7 else "🟡" if draft.fit_score >= 5 else "🔴"
    col_score, col_rationale = st.columns([1, 4])
    col_score.metric("Fit", f"{score_icon} {draft.fit_score}/10")
    col_rationale.markdown(f"**{esc(draft.fit_rationale)}**")

    st.markdown("#### Requirement coverage")
    coverage = pd.DataFrame(
        [
            {
                "Requirement": m.requirement,
                "Strength": m.strength,
                "Evidence": ", ".join(m.evidence_ids) or "—",
                "Note": m.note,
            }
            for m in draft.matches
        ]
    )
    # st.dataframe does not render markdown in cells, so raw values are safe here.
    st.dataframe(coverage, width="stretch", hide_index=True)

    st.markdown("#### Drafted bullets")
    for i, bullet in enumerate(draft.bullets, 1):
        st.markdown(f"**{i}.** {esc(bullet.text)}")
        st.caption(
            f"evidence: {esc(', '.join(bullet.evidence_ids))} · "
            f"targets: {esc(bullet.targets_requirement)}"
        )

    if draft.gaps:
        st.markdown("#### Honest gaps")
        for gap in draft.gaps:
            st.markdown(f"- {esc(gap)}")

    if draft.cover_note:
        st.markdown("#### Cover note")
        st.info(esc(draft.cover_note))

    if requirements.red_flags:
        with st.expander(f"Posting red flags ({len(requirements.red_flags)})"):
            for flag in requirements.red_flags:
                st.markdown(f"- {esc(flag)}")

    with st.expander(f"Evidence trail — {len(evidence.searches)} searches, "
                     f"{len(evidence.entries)} entries retrieved"):
        for query in evidence.searches:
            st.markdown(f"- searched: `{esc(query)}`")
        st.markdown("**Retrieved entries** (the only experience the draft may cite):")
        for entry in evidence.entries.values():
            st.markdown(f"- `{esc(entry.entry_id)}` ({esc(entry.kind)}): {esc(entry.text)}")

    st.divider()
    col_approve, col_reject, _ = st.columns([1, 1, 3])
    if col_approve.button("Approve & save", type="primary"):
        try:
            path = save_draft(output_dir, requirements, evidence, draft, analysis["source"])
        except OSError as e:
            st.error(f"Could not write draft: {esc(e)}")
            st.stop()
        st.session_state["saved_path"] = str(path)
        st.rerun()
    if col_reject.button("Reject"):
        st.session_state.pop("analysis", None)
        st.session_state["rejected"] = True  # message rendered after the rerun
        st.rerun()

elif st.session_state.get("saved_path"):
    saved = Path(st.session_state["saved_path"])
    st.divider()
    st.success(f"Draft approved and saved: `{esc(saved)}`")
    try:
        saved_text = saved.read_text(encoding="utf-8")
    except OSError as e:
        saved_text = None
        st.warning(f"Saved file could not be re-read for preview: {esc(e)}")
    if saved_text is not None:
        st.download_button(
            "Download markdown",
            data=saved_text,
            file_name=saved.name,
            mime="text/markdown",
        )
        with st.expander("Preview saved draft"):
            # st.code, not st.markdown: the file embeds model text and must
            # not be re-rendered as live markup.
            st.code(saved_text, language="markdown")
    if st.button("Start a new analysis"):
        st.session_state.pop("analysis", None)
        st.session_state.pop("saved_path", None)
        st.rerun()
