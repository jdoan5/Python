# Stage 4 — Containerize & Deploy

One container, one port, both apps (Stage 2 tailor + Stage 3 tracker) behind
a password gate. Everything mutable — drafts, tracker DB, your real
inventory — lives on a mounted volume at `/data`, so the container stays
disposable.

```
portal.py            auth gate (APP_PASSWORD) + st.navigation over Stages 2/3
Dockerfile           python:3.12-slim, non-root, healthcheck, /data volume
docker-compose.yml   local production-like run
fly.toml             Fly.io config (scale-to-zero, persistent volume)
```

## Why auth is non-negotiable here

A deployed portal runs on **your** Anthropic API key. Without a gate, anyone
who finds the URL can burn your credits. The portal **fails closed**: if
`APP_PASSWORD` is unset it refuses to serve (set `ALLOW_UNAUTHENTICATED=1`
only for local dev). Comparison uses `hmac.compare_digest`; the session flag
lives in Streamlit session state. Scope is deliberate: a shared password is
right-sized for a single-user personal tool — it is not multi-user auth.

## Run locally without Docker

```bash
cd "Job Search Agent"
APP_PASSWORD=changeme .venv/bin/streamlit run "Stage 4/portal.py"
```

Stages 2 and 3 still run standalone exactly as before — the portal is
additive (their `set_page_config` calls are guarded by `__name__`, which
Streamlit sets to `"__page__"` for mounted pages).

## Run locally with Docker

```bash
cd "Job Search Agent/Stage 4"
ANTHROPIC_API_KEY=sk-ant-... APP_PASSWORD=changeme docker compose up --build
```

(Or put both vars in `Stage 4/.env` — compose reads the `.env` next to the
compose file, and that path is dockerignored so it can never end up in the
image.)

Open http://localhost:8501 (bound to loopback only). Drafts and the tracker
DB persist in the `job_agent_data` volume across rebuilds.

**Your inventory is not baked into the image** — it's personal data. The
container reads it from `/data/inventory.yaml` (the image default for
`JOB_AGENT_INVENTORY`); copy it onto the volume once:

```bash
docker compose cp ../Stage\ 1/data/experience_inventory.yaml portal:/data/inventory.yaml
```

## Deploy to Fly.io

Steps are in the comments at the top of [fly.toml](fly.toml). Highlights:
scale-to-zero when idle (near-free), a 1GB persistent volume for `/data`,
secrets via `fly secrets set` (never in the image), HTTPS forced.

**Alternatives:** Render/Railway work the same way (Dockerfile + volume +
env secrets). Streamlit Community Cloud is the zero-infra option but has no
persistent disk — the tracker DB would reset on redeploys, so prefer a
container host for this project. **Not Vercel**: serverless timeouts kill
30-60s LLM calls and SQLite doesn't persist there.

## Environment variables

| Var | Required | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | yes | the pipeline's API access |
| `APP_PASSWORD` | yes (prod) | portal gate; unset = refuses to serve |
| `ALLOW_UNAUTHENTICATED` | no | `1` = skip gate, local dev only |
| `JOB_AGENT_INVENTORY` | no | inventory path (container default `/data/inventory.yaml`; local default: the Stage 1 sample) |
| `JOB_AGENT_OUTPUT_DIR` | no | where approved drafts go (container default `/data/output`) |
| `JOB_TRACKER_DB` | no | tracker SQLite path (container default `/data/tracker.db`) |
| `JOB_AGENT_MODEL` | no | override the model id |

## Stage 5 preview — Terraform on AWS

You already have an AWS account (us-east-1). Stage 5 will provision this
same container with Terraform: **ECR** (image registry) + **App Runner**
(managed container service — the AWS analog of what Fly does here) +
**Secrets Manager** (API key + password) + IAM roles. Two safety notes
before then:

1. **Create an IAM user for daily use** — don't work as the root account.
2. **Set a billing alarm** (Billing → Budgets → zero-spend budget) before
   creating any resource, so a forgotten service can't surprise you.
