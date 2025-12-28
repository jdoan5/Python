# Hello-Prod FastAPI

A tiny **production-style FastAPI starter**.

It’s intentionally simple, but shows how you would structure a real service:

- Versioned API router (`/api/v1`)
- Centralized configuration with `settings`
- Health check endpoint for monitoring
- Centralized exception handler
- Ready for `uvicorn` + tests (`pytest`)

---

## Project structure

```text
Hello-Prod-FastAPI/
├─ app/
│  ├─ __init__.py
│  ├─ config.py          # Settings (app name, environment, debug flag, etc.)
│  ├─ main.py            # FastAPI application factory + root route + error handler
│  └─ api/
│     ├─ __init__.py
│     └─ v1.py           # Versioned API router (e.g. /api/v1/health)
│
├─ tests/
│  └─ test_health.py     # Basic health-check test
│
├─ requirements.txt      # Python dependencies
├─ run_dev.sh            # Helper script for local dev (optional)
└─ README.md             # This file
```
## Requirements
	•	Python 3.11+ (3.10 also fine)
	•	pip and venv available on your system