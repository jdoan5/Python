# Hello-Prod FastAPI — Python

A tiny but **production-flavored** FastAPI service.

This project is intentionally small, but it’s structured the way you’d build a real API: versioned routes, environment-based settings, and a simple health endpoint that could sit behind a load balancer.

---

## Features

- ✅ **FastAPI** app with versioned routes (`/api/v1/...`)
- ✅ **Health check** endpoint ready for uptime probes
- ✅ **Hello** endpoint that returns a friendly message plus environment info
- ✅ **Config via environment variables / `.env`** using `pydantic-settings`
- ✅ Centralized **error handler** for unexpected exceptions
- ✅ Optional **tests** using `TestClient` (FastAPI’s test utilities)

---

## Tech Stack

- **Python 3.11+**
- **FastAPI**
- **Uvicorn**
- **pydantic-settings** (for configuration)

---

## Project Structure

```text
Hello-Prod-FastAPI/
├─ app/
│  ├─ __init__.py
│  ├─ main.py          # FastAPI app factory, routes wiring, error handler
│  ├─ config.py        # Settings pulled from environment / .env
│  └─ api/
│     ├─ __init__.py
│     └─ v1.py         # Versioned API routes (health, hello)
├─ tests/
│  └─ test_health.py   # Example test for /api/v1/health
├─ requirements.txt
├─ README.md
└─ run_dev.sh          # Optional helper to start the dev server
