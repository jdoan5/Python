# Stage 3 — Application Tracker & Analytics

Tracks every approved draft through the application lifecycle
(`drafted → applied → interviewing → offer / rejected / withdrawn`) and
answers the meta-question the whole project builds toward: **which gaps keep
costing me interviews — what should I learn next?**

## Zero-coupling integration

Stage 3 required **no changes** to Stages 1 or 2. `save_draft()` already
writes a JSON sidecar next to every approved markdown draft; Stage 3 ingests
those sidecars from `Stage 1/output/` into SQLite on dashboard load. The
filesystem is the integration contract:

```
Stage 1 CLI  ─┐
              ├── save_draft() → output/*.md + *.json ──→ Stage 3 sync → SQLite
Stage 2 UI   ─┘                                            (status + notes live here)
```

Re-syncing is idempotent (`sidecar_path` is UNIQUE) and never touches
user-edited status/notes.

## Run it

```bash
cd "Job Search Agent"
source .venv/bin/activate
streamlit run "Stage 3/dashboard.py"
```

Dashboard sections:
- **Metrics** — tracked / applied / interviewing / offers / average fit
- **Applications board** — edit status and notes inline, then *Save changes*
- **Charts** — status breakdown, fit-score distribution
- **Recurring gaps** — must-have requirements with no evidence across
  postings, ranked by frequency: your learn-next list
- **Draft preview** — read any saved draft without leaving the dashboard

## Configuration (deploy-friendly)

| Env var | Default | Purpose |
|---|---|---|
| `JOB_TRACKER_OUTPUT_DIR` | `../Stage 1/output` | where sidecars are read from |
| `JOB_TRACKER_DB` | `Stage 3/tracker.db` | SQLite location (mount a volume in prod) |

Both are also editable in the sidebar at runtime.

## Tests

```bash
cd "Stage 3"
../.venv/bin/python -m pytest -q
```

Covers sidecar parsing (including malformed files), idempotent sync,
preservation of user edits across re-syncs, status validation, and gap
normalization/ranking.

## Production path (Stages 4–5 preview)

- **Stage 4 — containerize + deploy**: one Dockerfile running Stage 2 and
  Stage 3 behind auth (they burn your API key); SQLite on a persistent
  volume. Fly.io / Render / Railway, or Streamlit Community Cloud for the
  zero-infra version. *Not Vercel* — Streamlit needs a long-running server
  and the pipeline's 30-60s LLM calls exceed serverless limits.
- **Stage 5 — Terraform**: provision the same deployment as code (e.g. AWS
  App Runner + Secrets Manager). Terraform is infrastructure-as-code, not a
  host — it's the DevOps chapter of the story.
