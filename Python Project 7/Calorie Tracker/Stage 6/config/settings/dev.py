"""Development settings — convenient, insecure on purpose, SQLite.

This is what manage.py uses by default. Never point a deployment at it.
"""

from .base import *  # noqa: F401,F403

DEBUG = True

# Dev-only fallback; harmless because DEBUG never leaves this module.
SECRET_KEY = os.environ.get(
    "DJANGO_SECRET_KEY",
    "django-insecure-dev-only-do-not-use-in-production",
)

ALLOWED_HOSTS = ["localhost", "127.0.0.1"]

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
    }
}
