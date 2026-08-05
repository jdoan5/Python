# Calorie Tracker — Stage 6 (Vercel + Neon)

The Stage 4 app deployed **serverlessly, for $0**: Django runs as a single
Vercel Python function, static files ship to Vercel's CDN, and Postgres
lives on Neon's free tier. Live URL, no servers, no monthly bill.

## Why Vercel works here (and didn't for the Streamlit projects)

Serverless fits **request/response** apps. This tracker's requests are
millisecond ORM queries over plain HTTP — no WebSockets, no long-running
work, and Stage 4 already externalized all state behind `DATABASE_URL`.
The Streamlit projects needed a persistent WebSocket per user and 30-60s
LLM calls: exactly what serverless kills. Same question every time:
*match the workload to the platform.*

```
wsgi_app.py               the ONE serverless function (root-level, because
                          api/ is already the DRF app package)
vercel.json               routes: /static/* → CDN, everything else → Django
build_files.sh            build step: pip install + collectstatic
config/settings/vercel.py prod.py hardening, serverless-adapted
```

## Deploy runbook (once, ~15 minutes)

**Account you create yourself** (free, no card): [Vercel](https://vercel.com)
— sign up with your GitHub account. A separate Neon account is NOT needed:
Vercel's Storage tab provisions the Postgres for you.

**1. Create the database inside Vercel.** Dashboard → your project →
**Storage** → **Create Database** → **Neon (Postgres)** → free plan. Vercel
auto-injects connection env vars into the project. Verify a `DATABASE_URL`
entry exists under Settings → Environment Variables (if the integration only
injected `POSTGRES_URL`-style names, copy the value into a new `DATABASE_URL`
variable — our settings read exactly that name). Copy the **connection
string** from the Storage tab's Connect panel for step 2.

*(Alternative: a separate [neon.tech](https://neon.tech) account works
identically — same database company — you just manage it outside Vercel.)*

**2. Migrate the schema** — run migrations from your Mac against Neon
(serverless has no entrypoint; migrating from the outside is the simple,
explicit pattern):

```bash
cd "Stage 6"
source .venv/bin/activate
DJANGO_SETTINGS_MODULE=config.settings.vercel \
  DJANGO_SECRET_KEY=migrate-only \
  DATABASE_URL='<your Neon connection string>' \
  python manage.py migrate
```

Create your login the same way (`createsuperuser` with the same env vars).

**3. Vercel: deploy from this folder** (CLI keeps the monorepo simple —
this folder becomes the project root):

```bash
npx vercel login          # authenticate in the browser
npx vercel                # first deploy: accept defaults when prompted
```

**4. Set the environment variables** in the Vercel dashboard (Project →
Settings → Environment Variables), then redeploy:

| Name | Value |
|---|---|
| `DJANGO_SETTINGS_MODULE` | `config.settings.vercel` |
| `DJANGO_SECRET_KEY` | `python3 -c 'import secrets;print(secrets.token_urlsafe(50))'` |
| `DATABASE_URL` | auto-injected by the Storage integration (verify; see step 1) |

```bash
npx vercel --prod
```

Your app is live at `https://<project>.vercel.app` — sign in with the
superuser from step 2, or sign up fresh. The Stage 5 Bruno collection runs
against it too: change `baseUrl` in a new Bruno environment.

## Honest limitations (also good interview answers)

- **Cold starts** — an idle function takes ~1-3s on the first request, then
  it's warm. Fine for a portfolio demo; a paid always-warm instance or a
  container host fixes it if it ever matters.
- **Per-instance throttling** — the `/api/token/` 5/min cap uses Django's
  in-memory cache, which is per warm instance here, not global. A shared
  cache (managed Redis) restores the global cap; noted in
  `settings/vercel.py`.
- **No filesystem** — anything that writes files locally would silently
  vanish between invocations. This app writes only to Postgres, which is
  why it ports cleanly.

## Costs

Vercel Hobby: $0 (personal projects). Neon free tier: 0.5GB Postgres, $0 —
it may suspend the DB after inactivity (first request wakes it, adding a
few hundred ms). Nothing here can surprise-bill you, unlike the ~$50/mo
ECS stack from the Job Search Agent's Stage 5.
