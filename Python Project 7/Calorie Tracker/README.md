# Calorie Tracker (Django)

A staged Django learning project: log meals, watch a running daily total
against a calorie goal, and browse historical intake.

## Stages

- **Stage 1 (built)** — core Django: one app, one model, forms with
  validation, function- and class-based views, ORM aggregation for daily
  totals, templates with inheritance, the admin, a seed command, and a
  17-test suite. SQLite, dev server.
- **Stage 2 (built)** — users & auth: signup/login/logout, `Meal.user`
  foreign key, login-required views, user-scoped querysets, per-view
  isolation tests. 21 tests.
- **Stage 3 (built)** — Django REST Framework: token + session auth, a
  user-scoped ModelViewSet (cross-user access = 404), /api/meals/summary/
  aggregation endpoint feeding a Chart.js trend, CSV export. 41 tests.
- **Stage 4 (built)** — production posture: settings split (base/dev/prod),
  Postgres via DATABASE_URL, WhiteNoise + gunicorn, hardened prod settings
  (clean `check --deploy`), Docker + compose with healthchecks. 44 tests.

- **Stage 5 (built)** — Bruno API collection: token-auth chaining, an
  idempotent CRUD chain, negative tests — 21 assertions run headless via
  `bru` CLI against the Stage 4 Docker stack.
- **Stage 6 (built)** — serverless deploy: Django as a Vercel Python
  function + Neon Postgres + CDN static files, $0/month. Deploy runbook in
  the stage README (needs your Vercel/Neon accounts).

Each stage is a self-contained folder with its own venv and README (Stage 5
is a file-based API collection rather than an app copy), same as the other
staged projects in this repo.
