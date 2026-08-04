# Calorie Tracker — Stage 3 (REST API, Charts, CSV)

Stage 2's app, now with a programmatic surface: a Django REST Framework API
(the single most job-posting-frequent Django skill), a Chart.js daily-trend
chart driven by the API, and CSV export.

## Run it

```bash
cd "Stage 3"
source .venv/bin/activate
python manage.py migrate                    # adds the authtoken table
python manage.py runserver
python manage.py test                       # 41 tests
```

## The API

| Endpoint | What |
|---|---|
| `POST /api/token/` | exchange `username`+`password` for a token |
| `GET/POST /api/meals/` | list (paginated) / create |
| `GET/PATCH/DELETE /api/meals/<id>/` | detail — **404 for other users' meals** |
| `GET /api/meals/summary/` | daily totals, oldest-first, chart-ready |

Try it from the terminal:

```bash
TOKEN=$(curl -s -X POST http://127.0.0.1:8000/api/token/ \
  -d "username=YOU&password=YOURPASS" | python3 -c "import sys,json;print(json.load(sys.stdin)['token'])")

curl -s http://127.0.0.1:8000/api/meals/ -H "Authorization: Token $TOKEN"

curl -s -X POST http://127.0.0.1:8000/api/meals/ \
  -H "Authorization: Token $TOKEN" -H "Content-Type: application/json" \
  -d '{"name":"Pho","meal_type":"lunch","calories":550,"eaten_at":"2026-01-15T12:00:00-05:00"}'
```

Or just open `http://127.0.0.1:8000/api/meals/` in the browser while signed
in — DRF's **browsable API** renders forms for every endpoint (session auth).

## Concepts this stage teaches

- **ModelSerializer** — ModelForm's JSON twin; model validators (the
  calories 1-5000 bounds) are copied onto serializer fields automatically;
  `validate_eaten_at()` mirrors Stage 1's form rule.
- **ModelViewSet + DefaultRouter** — full CRUD with correct status codes
  from ~15 lines; `get_queryset()` re-applies the Stage 2 scope rule.
- **404-not-403 for cross-user objects** — scoping the queryset means other
  users' meal IDs *don't exist* from your perspective; a 403 would confirm
  the ID is real (an enumeration leak).
- **Ownership is never client data** — `user` isn't a serializer field;
  `perform_create()` stamps `request.user`. There's a test that tries to
  forge `user` in the payload and asserts it's ignored.
- **Two auth schemes, one API** — SessionAuthentication (browser, chart,
  browsable API) + TokenAuthentication (scripts, mobile) configured in
  `REST_FRAMEWORK` settings.
- **The page as API client** — history.html's chart `fetch()`es
  `/api/meals/summary/` with session credentials; the same endpoint serves
  curl with a token.
- **Plain-Django file download** — `export_csv` needs no DRF: set
  `content_type`, add `Content-Disposition`, hand the response to
  `csv.writer`.

## Security notes (found by this stage's adversarial review)

- **`/api/token/` is throttled** (5/min) because DRF deliberately ships
  `ObtainAuthToken` with `throttle_classes = ()` — an unmetered
  password-guessing oracle otherwise.
- **Password change revokes the API token.** Django invalidates sessions on
  password change; DRF tokens are permanent DB rows that would survive —
  exactly wrong for the stolen-password scenario. See
  `TokenRevokingPasswordChangeView`.
- **`eaten_at` must carry a UTC offset** (e.g. `2026-01-15T12:00:00-05:00`).
  Naive datetimes are rejected rather than silently interpreted in the
  server's timezone; note it also must not be in the future.

## Next: Stage 4

Production posture — Postgres, settings split, Docker, deploy (the Job
Search Agent Stage 4/5 playbook applied to Django).
