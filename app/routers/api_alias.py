"""API alias router to handle /api/* paths that redirect to actual endpoints."""

from fastapi import APIRouter
from fastapi.responses import RedirectResponse

router = APIRouter(prefix="/api", tags=["api-alias"])

@router.get("/health")
async def api_health_alias():
    """Redirect /api/health to /health."""
    return RedirectResponse(url="/health", status_code=307)

@router.post("/demo/run")
async def demo_run_alias():
    """Redirect /api/demo/run to /draft/ (placeholder for demo functionality)."""
    return RedirectResponse(url="/draft/", status_code=307)

@router.get("/demo/status")
async def demo_status_alias():
    """Redirect /api/demo/status to /debug/strict-report for demo verification."""
    return RedirectResponse(url="/debug/strict-report", status_code=307)
