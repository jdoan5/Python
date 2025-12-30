# Dockerfile
FROM python:3.11-slim

# 1) Basic env settings
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# 2) System deps (you can trim these later if not needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# 3) Workdir
WORKDIR /app

# 4) Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5) Copy project code (including model artifacts)
COPY . .

# 6) Expose API port
EXPOSE 8000

# 7) Entrypoint: run FastAPI with uvicorn
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]