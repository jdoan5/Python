from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .config import settings
from .api.v1 import router as v1_router


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        version="1.0.0",
        debug=settings.debug,
    )

    # Include versioned API
    app.include_router(v1_router)

    @app.get("/", summary="Root")
    async def root():
        return {
            "message": "Hello-Prod FastAPI is running OK",
            "docs": "/docs",
            "env": settings.environment,
        }

    # Simple example of centralized error handling
    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        # In a real prod app you’d log this to structured logs / APM
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "error_type": type(exc).__name__},
        )

    return app


app = create_app()