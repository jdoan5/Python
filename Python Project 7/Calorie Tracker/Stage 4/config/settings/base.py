"""Base settings — everything shared by dev and prod.

The settings-split pattern: base.py holds the common truth; dev.py and
prod.py import * from it and override only what differs. Which module loads
is chosen by DJANGO_SETTINGS_MODULE (manage.py defaults to dev, wsgi.py to
prod — env var always wins).

NOTE: this file is one directory deeper than the old settings.py, so
BASE_DIR needs one more .parent.
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'meals',
    'accounts',
    'rest_framework',
    'rest_framework.authtoken',  # DB-backed API tokens (adds a migration)
    'api',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    # WhiteNoise serves collected static files from the app process itself —
    # the standard answer when there's no nginx/CDN in front. Must sit right
    # after SecurityMiddleware.
    'whitenoise.middleware.WhiteNoiseMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'config.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / 'templates'],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'config.wsgi.application'

AUTH_PASSWORD_VALIDATORS = [
    {'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator'},
    {'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator'},
    {'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator'},
    {'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator'},
]

LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'America/New_York'
USE_I18N = True
USE_TZ = True

STATIC_URL = 'static/'
STATIC_ROOT = BASE_DIR / 'staticfiles'   # collectstatic target (prod)

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# --- Calorie Tracker ---
try:
    DAILY_CALORIE_GOAL = int(os.environ.get("DAILY_CALORIE_GOAL", "2000"))
except ValueError:
    DAILY_CALORIE_GOAL = 2000
if DAILY_CALORIE_GOAL <= 0:  # a zero/negative goal breaks the progress math
    DAILY_CALORIE_GOAL = 2000

# --- Auth flow ---
LOGIN_URL = 'login'
LOGIN_REDIRECT_URL = 'meals:today'
LOGOUT_REDIRECT_URL = 'login'

# --- Django REST Framework ---
REST_FRAMEWORK = {
    'DEFAULT_AUTHENTICATION_CLASSES': [
        'rest_framework.authentication.SessionAuthentication',
        'rest_framework.authentication.TokenAuthentication',
    ],
    'DEFAULT_PERMISSION_CLASSES': [
        'rest_framework.permissions.IsAuthenticated',
    ],
    'DEFAULT_PAGINATION_CLASS': 'rest_framework.pagination.PageNumberPagination',
    'PAGE_SIZE': 20,
    # ObtainAuthToken overrides default throttles with (), so the cap is
    # applied via ScopedRateThrottle on our subclass (api.views).
    'DEFAULT_THROTTLE_RATES': {'token_obtain': '5/min'},
}
