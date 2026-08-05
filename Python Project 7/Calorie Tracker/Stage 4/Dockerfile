# Calorie Tracker — production image.
#
# Build from the Stage 4 folder:
#   docker build -t calorie-tracker .
# Normally driven by docker-compose.yml (which also runs Postgres).
#
# Lessons carried from the Job Search Agent deploy: dependency layer cached
# separately from code, non-root runtime user, JSON-form COPY, healthcheck,
# no secrets in the image.

FROM python:3.14-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DJANGO_SETTINGS_MODULE=config.settings.prod

RUN useradd --create-home appuser
WORKDIR /app

# Dependency layer first — rebuilds only when requirements change.
COPY ["requirements.txt", "requirements.txt"]
RUN pip install --no-cache-dir -r requirements.txt

COPY ["manage.py", "manage.py"]
COPY ["config", "config"]
COPY ["meals", "meals"]
COPY ["accounts", "accounts"]
COPY ["api", "api"]
COPY ["templates", "templates"]
COPY ["entrypoint.sh", "entrypoint.sh"]

# collectstatic needs the settings module but no DB and no real secret —
# a throwaway key keeps the build free of secrets.
RUN DJANGO_SECRET_KEY=build-time-only DATABASE_URL=sqlite:///build.sqlite3 \
    python manage.py collectstatic --noinput \
    && chmod +x entrypoint.sh \
    && chown -R appuser:appuser /app/staticfiles

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=25s CMD \
    python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz/', timeout=4)"

ENTRYPOINT ["./entrypoint.sh"]
CMD ["gunicorn", "config.wsgi", "--bind", "0.0.0.0:8000", "--workers", "2", \
     "--access-logfile", "-"]
