# Stage 4 — container image for the Job Search Agent portal.
#
# Build from the PROJECT ROOT (the folder containing "Stage 1".."Stage 4"):
#   docker build -f "Stage 4/Dockerfile" -t job-search-agent .
#
# Run:
#   docker run -p 127.0.0.1:8501:8501 \
#     -e ANTHROPIC_API_KEY=sk-ant-... \
#     -e APP_PASSWORD=choose-a-password \
#     -v job_agent_data:/data \
#     job-search-agent
#
# Everything mutable lives under /data (drafts, tracker DB, inventory) so the
# container itself stays disposable. The image bakes only the PUBLIC sample
# inventory (tracked in the repo); the entrypoint seeds it to
# /data/inventory.yaml when none exists, so volume-less hosts (ECS Fargate)
# boot functional. To use a private inventory, put it on the volume — it is
# never overwritten:
#   docker cp my_inventory.yaml <container>:/data/inventory.yaml

FROM python:3.12-slim

RUN useradd --create-home appuser
WORKDIR /app

# Dependency layer: install ONLY the dependency list (exported from
# pyproject with stdlib tomllib) so this layer caches across code changes.
# The job-search-agent package itself is installed once, after the full COPY
# below — installing "./Stage 1" here would build an empty wheel (no source
# copied yet) and silently mask failures.
COPY ["Stage 1/pyproject.toml", "Stage 1/pyproject.toml"]
RUN python -c "import tomllib; print('\n'.join(tomllib.load(open('Stage 1/pyproject.toml','rb'))['project']['dependencies']))" > /tmp/requirements.txt \
    && pip install --no-cache-dir -r /tmp/requirements.txt "streamlit~=1.50" "pandas~=2.3"

COPY ["Stage 1", "Stage 1"]
COPY ["Stage 2", "Stage 2"]
COPY ["Stage 3", "Stage 3"]
COPY ["Stage 4", "Stage 4"]

# Application code stays root-owned (read-only to the runtime user);
# only /data is writable, and the entrypoint chowns it at start.
RUN pip install --no-cache-dir "./Stage 1" \
    && chmod +x "Stage 4/entrypoint.sh"

ENV JOB_AGENT_INVENTORY=/data/inventory.yaml \
    JOB_AGENT_OUTPUT_DIR=/data/output \
    JOB_TRACKER_OUTPUT_DIR=/data/output \
    JOB_TRACKER_DB=/data/tracker.db \
    PYTHONUNBUFFERED=1

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s CMD \
    python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=4)"

# Entrypoint runs as root just long enough to chown the volume, then drops
# to appuser via setpriv (see entrypoint.sh).
ENTRYPOINT ["/app/Stage 4/entrypoint.sh"]

CMD ["streamlit", "run", "Stage 4/portal.py", \
     "--server.port=8501", "--server.address=0.0.0.0", \
     "--server.headless=true", "--browser.gatherUsageStats=false"]
