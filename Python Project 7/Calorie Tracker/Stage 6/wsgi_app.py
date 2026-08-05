"""Vercel serverless entry point.

Vercel's Python builder looks for a WSGI/ASGI callable named `app`. The
whole Django project becomes ONE serverless function; vercel.json routes
every non-static path here. (This file lives at the root, not in api/,
because api/ is already our DRF app package — the zero-config convention
would have turned every module in it into its own function.)
"""

import os

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings.vercel")

from django.core.wsgi import get_wsgi_application  # noqa: E402

app = get_wsgi_application()
