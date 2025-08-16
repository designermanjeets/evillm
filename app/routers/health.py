"""Health check router for system monitoring."""

from datetime import datetime
from typing import Dict, Any
from fastapi import APIRouter, Depends
import structlog

from app.config.settings import get_settings

logger = structlog.get_logger(__name__)
router = APIRouter()


@router.get("/")
async def health_check() -> Dict[str, Any]:
    """Basic health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "0.1.0",
        "service": "Logistics Email AI"
    }


@router.get("/detailed")
async def detailed_health_check() -> Dict[str, Any]:
    """Detailed health check with component status."""
    settings = get_settings()
    
    # Check database health
    from app.database.engine import check_database_connection, get_database_info
    from app.database.migrations import get_current_migration, get_pending_migrations
    
    # Check storage health
    from app.storage.health import check_storage_health
    
    db_healthy = await check_database_connection()
    db_info = await get_database_info()
    current_migration = await get_current_migration()
    pending_migrations = await get_pending_migrations()
    
    # Determine overall database status
    if not db_healthy:
        db_status = "unhealthy"
    elif pending_migrations:
        db_status = "migrations_pending"
    else:
        db_status = "healthy"
    
    components = {
        "database": {
            "status": db_status,
            "connection": "connected" if db_healthy else "disconnected",
            "current_migration": current_migration,
            "pending_migrations": len(pending_migrations) if pending_migrations else 0,
            "info": db_info
        },
        "search": "healthy",    # TODO: Add actual search health check
        "storage": await check_storage_health(),
        "agents": "healthy",    # TODO: Add actual agent health check
    }
    
    # Overall status depends on critical components
    overall_status = "healthy"
    if db_status != "healthy":
        overall_status = "degraded"
    
    return {
        "status": overall_status,
        "timestamp": datetime.utcnow().isoformat(),
        "version": "0.1.0",
        "environment": settings.environment,
        "components": components,
        "uptime": "0s"  # TODO: Add actual uptime calculation
    }


@router.get("/ready")
async def readiness_check() -> Dict[str, Any]:
    """Readiness check for Kubernetes health probes."""
    # TODO: Add actual readiness checks
    return {
        "status": "ready",
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("/live")
async def liveness_check() -> Dict[str, Any]:
    """Liveness check for Kubernetes health probes."""
    # TODO: Add actual liveness checks
    return {
        "status": "alive",
        "timestamp": datetime.utcnow().isoformat()
    }
