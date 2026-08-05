#!/usr/bin/env bash
# Vercel static build step: collect Django's static files (admin CSS/JS etc.)
# into staticfiles/, which Vercel then serves from its CDN — the lambda never
# handles /static/ requests. Env vars (DJANGO_SECRET_KEY, DATABASE_URL) are
# available at build time because they're set in the Vercel project settings.
set -euo pipefail
pip3 install -r requirements.txt
python3 manage.py collectstatic --noinput
