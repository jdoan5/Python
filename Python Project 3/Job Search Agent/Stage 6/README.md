# Stage 6 — Live demo on Streamlit Community Cloud ($0)

The Stage 4 portal, hosted free. Stage 5 proved the enterprise path
(Terraform → ECR → ECS Fargate → ALB, ~$50/month whether or not anyone
visits — destroyed on purpose, redeployable in ~10 minutes). Stage 6 asks
the platform-matching question one more time: a Streamlit app needs a
persistent WebSocket server, Community Cloud provides exactly that for
public repos, so the always-on demo costs **$0 in infrastructure**.

```
streamlit_app.py       entry point: st.secrets → env bridge, sys.path
                       bootstrap for job_agent, then Stage 4's gate + nav
../../requirements.txt runtime deps — at the REPO ROOT, not here: Community
                       Cloud passes the requirements path to pip unquoted,
                       so a path containing spaces ("Python Project 3/…")
                       breaks the install. Root path has no spaces.
```

Stages 1–4 are untouched — every platform difference lives in the entry
file. The sample inventory tracked in the repo is the default, so the
tailor page works on first boot.

## Deploy runbook (~5 minutes, all in the browser)

1. Sign in at [share.streamlit.io](https://share.streamlit.io) with GitHub.
2. **Create app** → *Deploy a public app from GitHub*.
3. Fill in:
   - **Repository**: `jdoan5/Python`
   - **Branch**: `main`
   - **Main file path**: `Python Project 3/Job Search Agent/Stage 6/streamlit_app.py`
4. **Advanced settings** before deploying:
   - Python version: 3.12
   - **Secrets** — paste (TOML, real values, no quotes needed around keys):

     ```toml
     ANTHROPIC_API_KEY = "sk-ant-…"
     APP_PASSWORD = "choose-a-demo-password"
     ```

5. **Deploy**. First build takes a few minutes; the app gets a stable
   `https://<name>.streamlit.app` URL you can put on the portfolio card.

## Costs and controls — the honest picture

- **Infrastructure: $0.** The app sleeps after ~12h of inactivity; the
  first visitor wakes it (~1 minute spin-up).
- **Per-use: Anthropic API tokens.** Each "Analyze posting" run costs a
  few cents on the key in Secrets. The password gate exists precisely so
  strangers cannot burn credits — share the password selectively, rotate
  it in Secrets anytime (Manage app → Settings → Secrets).
- **State is ephemeral.** Approved drafts and the tracker DB live on the
  app instance's disk and reset on reboot/redeploy — same deliberate
  trade-off as Fargate in Stage 5. Real state would mean external storage
  (S3/Postgres), which is future work, not demo work.

## Known limitations

- Community Cloud requires the repo to be **public** — true here already;
  the inventory it ships is the public sample, secrets stay in the
  dashboard, never in git.
- One small shared instance (1 GB): fine for a demo, not a fleet.
- The wake-from-sleep pause is visible to the first visitor. A demo GIF on
  the portfolio card covers the impatient case.
