"""Vercel settings — prod.py's hardening, adapted to serverless.

Inherits everything from prod (fail-loud SECRET_KEY/DATABASE_URL, security
headers, stdout logging) and overrides only what serverless changes:

- TLS is terminated at Vercel's edge and *every* public request is already
  HTTPS, so Django-level SSL redirect stays off; the X-Forwarded-Proto
  header keeps request.is_secure() (and Secure cookies) working.
- Static files are served by Vercel's CDN (see vercel.json routes), not by
  the function, so WhiteNoise and manifest storage are dropped — a lambda
  has no staticfiles/ directory at runtime, and ManifestStaticFilesStorage
  would crash template rendering looking for its manifest.
"""

from .prod import *  # noqa: F401,F403

# Vercel deployment URLs are *.vercel.app; extras still come from env.
ALLOWED_HOSTS = [".vercel.app", "localhost"] + [
    h.strip()
    for h in os.environ.get("DJANGO_ALLOWED_HOSTS", "").split(",")
    if h.strip()
]

CSRF_TRUSTED_ORIGINS = ["https://*.vercel.app"] + [
    o.strip()
    for o in os.environ.get("DJANGO_CSRF_TRUSTED_ORIGINS", "").split(",")
    if o.strip()
]

# Edge handles HTTPS; trust the forwarded proto so is_secure() is accurate.
SECURE_SSL_REDIRECT = False
SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")

# CDN serves /static/ — plain storage, no WhiteNoise, no manifest.
STORAGES = {
    "default": {"BACKEND": "django.core.files.storage.FileSystemStorage"},
    "staticfiles": {
        "BACKEND": "django.contrib.staticfiles.storage.StaticFilesStorage",
    },
}
MIDDLEWARE = [m for m in MIDDLEWARE if "whitenoise" not in m.lower()]  # noqa: F405

# Serverless caveat, documented in the README: the token throttle uses the
# default in-memory cache, which is per-function-instance on Vercel — the
# 5/min cap applies per warm instance, not globally. A shared cache (e.g.
# managed Redis) would restore a global cap; out of scope for this stage.
