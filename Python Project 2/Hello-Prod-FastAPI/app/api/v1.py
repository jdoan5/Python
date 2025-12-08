# versioned routes

from fastapi import APIRouter
from datetime import datetime, timezone

from app.config import settings

router = APIRouter(prefix="/api/vi", tags=["v1"])

@router.get("/health", summary="Health Check")
async def health_check():
    """Lightweight probe for uptime checks / load balancer health."""
    return {
        "status": "ok",
        "app": settings.app_name,
        "env": settings.environment,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@router.get("/hello", summary="Hello endpoint")
async def hello(name: str = "World"):
    """Tiny demo endpoint that feels like 'Hello, production!'"""
    return {
        "message": f"Hello, {name}",
        "env": settings.environment,
    }