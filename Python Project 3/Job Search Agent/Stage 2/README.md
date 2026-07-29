# Stage 2 — Browser UI

Streamlit front-end over the Stage 1 pipeline. Same four stages, same
grounding guarantees — the human-in-the-loop gate becomes **Approve / Reject
buttons** instead of a terminal prompt.

## Why the UI doesn't reuse `run_pipeline()`

Stage 1's `run_pipeline()` takes a blocking `approve` callback — perfect for
a CLI prompt, impossible in Streamlit's rerun model (the script re-executes
top-to-bottom on every interaction). So the app calls the stage functions
directly:

```
Analyze click:  extract_requirements() -> gather_evidence() -> draft_application()
                            └── result parked in st.session_state
Approve click:  save_draft()          # the ONLY place anything touches disk
Reject click:   session cleared       # nothing written
```

This is the payoff of exposing stages as functions: two front-ends (CLI, web)
share one pipeline with zero changes to Stage 1 code.

## Run it

From the project root (`Job Search Agent/`):

```bash
source .venv/bin/activate
pip install -e "Stage 1" streamlit pandas   # first time only
streamlit run "Stage 2/app.py"
```

Opens `http://localhost:8501`. Paste a posting (most reliable) or fetch a
URL, click **Analyze posting**, review the fit score / coverage table /
bullets / gaps, then Approve or Reject. Approved drafts land in
`Stage 1/output/` alongside CLI-produced ones.

## PyCharm run configuration

Run → Edit Configurations → **+** → Python:
- **Module name**: `streamlit`
- **Parameters**: `run "Stage 2/app.py"`
- **Working directory**: the `Job Search Agent` folder
- **Interpreter**: the shared `.venv`
