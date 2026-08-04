"""Production settings — fail loudly on missing config, secure by default.

Required environment:
    DJANGO_SECRET_KEY       no fallback here, on purpose
    DATABASE_URL            e.g. postgres://user:pass@host:5432/dbname
    DJANGO_ALLOWED_HOSTS    comma-separated

Optional:
    DJANGO_CSRF_TRUSTED_ORIGINS  comma-separated, scheme included
    DJANGO_SECURE_SSL_REDIRECT   "1" when TLS terminates in front of the app
                                 (default off so plain-HTTP compose works)

Run `python manage.py check --deploy` against this module before shipping —
it audits exactly these settings.
"""

import dj_database_url

from .base import *  # noqa: F401,F403

DEBUG = False

# KeyError at boot beats a quietly-insecure server: no fallback.
SECRET_KEY = os.environ["DJANGO_SECRET_KEY"]

ALLOWED_HOSTS = [
    h.strip()
    for h in os.environ.get("DJANGO_ALLOWED_HOSTS", "").split(",")
    if h.strip()
]

# Postgres via a single URL (the 12-factor convention every host speaks).
DATABASES = {
    "default": dj_database_url.config(
        env="DATABASE_URL",
        conn_max_age=60,          # keep connections between requests
        conn_health_checks=True,
    )
}

# Hashed + compressed static files, served by WhiteNoise.
STORAGES = {
    "default": {"BACKEND": "django.core.files.storage.FileSystemStorage"},
    "staticfiles": {
        "BACKEND": "whitenoise.storage.CompressedManifestStaticFilesStorage",
    },
}

# --- Security headers (what `check --deploy` looks for) ---
CSRF_COOKIE_SECURE = True
SESSION_COOKIE_SECURE = True
SECURE_CONTENT_TYPE_NOSNIFF = True
SECURE_HSTS_SECONDS = 31536000  # 1 year; only sent over HTTPS anyway
SECURE_HSTS_INCLUDE_SUBDOMAINS = True
SECURE_HSTS_PRELOAD = True
SECURE_REFERRER_POLICY = "same-origin"

# Behind a TLS-terminating proxy (ALB, Fly, nginx): set to "1" and Django
# will redirect plain HTTP and trust X-Forwarded-Proto. Off by default so
# the local docker-compose (no TLS) still works.
if os.environ.get("DJANGO_SECURE_SSL_REDIRECT") == "1":
    SECURE_SSL_REDIRECT = True
    SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")

CSRF_TRUSTED_ORIGINS = [
    o.strip()
    for o in os.environ.get("DJANGO_CSRF_TRUSTED_ORIGINS", "").split(",")
    if o.strip()
]
