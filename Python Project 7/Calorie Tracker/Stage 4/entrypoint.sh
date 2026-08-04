#!/bin/sh
# Wait for Postgres, apply migrations, then hand off to the CMD (gunicorn).
#
# Running migrations at container start is the simple deployment story for a
# single-instance app. With multiple replicas you'd run migrations as a
# separate release step instead (two containers racing the same migration is
# the classic failure) — noted in the README.
set -e

echo "Waiting for the database..."
python - <<'PY'
import os, sys, time
import psycopg

url = os.environ.get("DATABASE_URL", "")
if not url.startswith(("postgres://", "postgresql://")):
    sys.exit(0)  # sqlite or unset: nothing to wait for

for attempt in range(30):
    try:
        psycopg.connect(url, connect_timeout=3).close()
        sys.exit(0)
    except psycopg.OperationalError:
        time.sleep(1)
print("Database never became reachable.", file=sys.stderr)
sys.exit(1)
PY

echo "Applying migrations..."
python manage.py migrate --noinput

exec "$@"
