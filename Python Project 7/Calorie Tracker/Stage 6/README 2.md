# Calorie Tracker — Stage 4 (Production Posture)

Stage 3's app, deployment-shaped: environment-split settings, Postgres,
WhiteNoise static files, gunicorn, and Docker — the same playbook used for
the Job Search Agent deploy, applied to Django.

```
config/settings/
  base.py        shared truth (apps, middleware, DRF, app settings)
  dev.py         DEBUG, SQLite, insecure-key fallback  ← manage.py default
  prod.py        fail-loud env config, Postgres, security headers ← wsgi.py default
Dockerfile       python:3.14-slim, non-root, cached dep layer, healthcheck
entrypoint.sh    wait-for-Postgres → migrate → exec gunicorn
docker-compose.yml  web + postgres:17, healthchecked, loopback-only port
```

## Run it — three ways

**1. Dev, unchanged from Stage 3** (SQLite, runserver):

```bash
source .venv/bin/activate
python manage.py migrate && python manage.py runserver
python manage.py test        # 44 tests, dev settings
```

**2. Prod settings locally, no Docker** (the fastest way to see the split):

```bash
DJANGO_SETTINGS_MODULE=config.settings.prod \
  DJANGO_SECRET_KEY=$(python3 -c 'import secrets;print(secrets.token_urlsafe(50))') \
  DATABASE_URL=sqlite:///db.sqlite3 \
  DJANGO_ALLOWED_HOSTS=localhost \
  DJANGO_SECURE_SSL_REDIRECT=1 \
  python manage.py check --deploy    # 0 issues with SSL redirect on; without
                                     # it, W008 is the one expected warning
                                     # (plain-HTTP compose has no TLS to redirect to)
```

**3. The real thing** (gunicorn + Postgres in Docker; Docker Desktop must be running):

```bash
DJANGO_SECRET_KEY=$(python3 -c 'import secrets;print(secrets.token_urlsafe(50))') \
  docker compose up --build
# http://127.0.0.1:8000 — data persists in the calorie_tracker_pgdata volume
```

## Concepts this stage teaches

- **Settings split** — `base/dev/prod` with `DJANGO_SETTINGS_MODULE`
  selecting; dev optimizes for convenience, prod **fails loudly** on missing
  config (`os.environ["DJANGO_SECRET_KEY"]` — a KeyError at boot beats a
  quietly-insecure server).
- **`check --deploy`** — Django's built-in production auditor; prod.py
  passes with zero issues (HSTS, secure cookies, nosniff, referrer policy).
- **12-factor database config** — one `DATABASE_URL` env var via
  `dj-database-url`; SQLite in dev, Postgres in prod, no code changes.
- **WhiteNoise** — hashed+compressed static files served by the app process;
  the standard answer when there's no nginx/CDN tier.
- **Entrypoint discipline** — wait for the DB (psycopg retry loop, not
  `sleep 5`), migrate, then `exec` gunicorn so signals reach the server.
  Single-instance simplification: with replicas, migrations move to a
  release step (two containers racing one migration is the classic failure).
- **Compose healthchecks** — web waits on `service_healthy`, not just
  "started"; Postgres's `pg_isready` is the gate. The web container probes
  its own `/healthz/` — a dedicated unauthenticated endpoint, because the
  login page would 301 under SSL redirect and 400 under strict
  ALLOWED_HOSTS, marking healthy containers unhealthy.
- **stdout logging** — prod.py configures console logging (Django's
  DEBUG=False default silently drops 500 tracebacks when ADMINS is unset)
  and gunicorn ships access logs to stdout: `docker logs` shows everything.

## Migrations note (SQLite → Postgres)

The compose Postgres starts empty and `entrypoint.sh` migrates it on first
boot — your dev SQLite data does not move. That's deliberate: schema is
portable (migrations), data migration is a separate concern
(`dumpdata`/`loaddata` if you ever want it — one command each way).

## Next steps beyond the tracker

The deploy targets from the Job Search Agent project apply unchanged: this
image runs on Fly.io (volume-free — state lives in managed Postgres) or ECS
Fargate + ALB via the same Terraform shapes, with `DATABASE_URL` pointing at
Fly Postgres / RDS.
