# Calorie Tracker — Stage 5 (Bruno API Collection)

The Stage 3/4 REST API, exercised end-to-end by a [Bruno](https://www.usebruno.com/)
collection. Unlike Postman workspaces, Bruno collections are **plain text
files** — this whole folder commits to git, opens in the Bruno desktop app,
and runs headless in CI via the CLI.

```
environments/local.bru   baseUrl + demo credentials (local test user only)
01-auth/                 POST /api/token/ -> chains {{token}} to every request
02-meals/                create -> list -> get -> patch -> summary -> delete
03-negative/             anon 403, naive datetime 400, future 400, calorie bounds 400
```

Latest run: **11/11 requests, 21/21 assertions, 5/5 scripted tests, ~700ms**
against the Stage 4 Docker stack.

## Run it

1. Start the Stage 4 stack (see `../Stage 4/README.md`), then create the
   test user once:

```bash
cd "../Stage 4"
DJANGO_SECRET_KEY=x docker compose exec -T web python manage.py shell -c "
from django.contrib.auth import get_user_model
u,_ = get_user_model().objects.get_or_create(username='bruno')
u.set_password('bruno-test-pass-1'); u.save()"
```

2. Run the collection (CLI, no install needed):

```bash
cd "../Stage 5"
npx -y @usebruno/cli run --env local
```

Or open the folder in the Bruno desktop app and click through requests.

## What the collection demonstrates

- **Auth chaining** — `01-auth` exchanges credentials for a token and stores
  it with `bru.setVar()` (a *runtime* var — see gotcha below), which every
  later request uses via `Authorization: Token {{token}}`.
- **A full CRUD chain** — the created meal's id flows through get/patch/
  delete; the summary test asserts the patched calories are included; the
  final delete leaves the account clean, making the collection **idempotent**
  (run it as many times as you like).
- **Fresh dates every run** — `eaten_at` is computed in pre-request scripts
  (now minus 1h, UTC), so the collection never goes stale against the
  "no future meals" and "aware datetimes only" API rules.
- **Negative testing** — the API contract's edges, pinned: 403 for
  anonymous, 400s for naive datetimes, future times, and absurd calories.

## Gotcha worth remembering (found while building this)

`bru.setEnvVar()` **persists into the environment file** — the first version
of this collection wrote the live API token and meal id into
`environments/local.bru`, a file destined for git. `bru.setVar()` is the
right tool for chained request values: runtime-only, gone when the run ends.
The env file holds only static config; secrets never belong in it.

(The `bruno`/`bruno-test-pass-1` credentials in `local.bru` are a
deliberate exception: a throwaway user that exists only in your local
Docker Postgres.)

## Note on the token throttle

Each run makes one `/api/token/` request, and Stage 3 throttles that
endpoint at 5/min — so more than five back-to-back runs inside a minute
will start failing with 429s on auth. That's the throttle doing its job.
