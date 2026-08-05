#!/usr/bin/env bash
# Vercel static build step: collect Django's static files into staticfiles/,
# which Vercel serves from its CDN.
set -euo pipefail

# Vercel's build image ships a PEP 668 "externally managed" Python (managed
# by uv) — bare pip refuses to install; the override is the documented
# escape hatch for throwaway build containers like this one.
pip3 install --break-system-packages -r requirements.txt

# Build-time fallbacks: our settings FAIL LOUDLY on missing env (by design),
# but collectstatic needs no real secret and no real database — and
# DATABASE_URL doesn't exist until the Storage integration is added.
export DJANGO_SETTINGS_MODULE="${DJANGO_SETTINGS_MODULE:-config.settings.vercel}"
export DJANGO_SECRET_KEY="${DJANGO_SECRET_KEY:-build-only-not-a-secret}"
export DATABASE_URL="${DATABASE_URL:-sqlite:///build-only.sqlite3}"

python3 manage.py collectstatic --noinput
