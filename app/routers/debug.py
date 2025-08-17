"""Debug routes for development and strict mode verification."""

from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.responses import JSONResponse
from typing import Dict, Any
import structlog

from app.middleware.tenant_isolation import get_tenant_id
from app.config.settings import get_settings

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/debug", tags=["debug"])


@router.get("/strict-report")
async def get_strict_report(request: Request) -> Dict[str, Any]:
    """Get strict mode configuration report (dev-only)."""
    try:
        # Get tenant ID from request state
        tenant_id = getattr(request.state, "tenant_id", "test-tenant")
        settings = get_settings()
        
        # Check if we're in development mode
        if settings.environment == "production":
            raise HTTPException(status_code=403, detail="Debug endpoints disabled in production")
        
        # Get strict mode configuration
        strict_config = {
            "strict_mode": settings.security.strict_mode,
            "require": {
                "bm25": settings.search.require_bm25,
                "vector": settings.search.require_vector
            },
            "llm_allow_stub": settings.llm.allow_stub,
            "security": {
                "redact_pii_in_logs": settings.security.redact_pii_in_logs,
                "cors_allowed_origins": settings.security.cors_allowed_origins,
                "csp": settings.security.csp,
                "rate_limit_per_minute": settings.security.rate_limit_per_minute
            },
            "quality_thresholds": {
                "hit_at_5": settings.search.quality_thresholds_hit_at_5,
                "mrr_at_10": settings.search.quality_thresholds_mrr_at_10,
                "ndcg_at_10": settings.search.quality_thresholds_ndcg_at_10
            },
            "eval": {
                "enabled": settings.eval_enabled,
                "threshold": settings.eval_threshold
            }
        }
        
        logger.info("Strict mode report requested", tenant_id=tenant_id, config=strict_config)
        
        return {
            "status": "success",
            "tenant_id": tenant_id,
            "environment": settings.environment,
            "strict_config": strict_config
        }
        
    except Exception as e:
        logger.error("Failed to generate strict mode report", error=str(e), tenant_id=tenant_id)
        raise HTTPException(status_code=500, detail="Failed to generate strict mode report")


@router.get("/health/detailed")
async def get_detailed_health(request: Request) -> Dict[str, Any]:
    """Get detailed health status for all dependencies (dev-only)."""
    try:
        # Get tenant ID from request state
        tenant_id = getattr(request.state, "tenant_id", "test-tenant")
        settings = get_settings()
        
        # Check if we're in development mode
        if settings.environment == "production":
            raise HTTPException(status_code=403, detail="Debug endpoints disabled in production")
        
        # Mock health checks for demo purposes
        # In production, these would be real health checks
        health_status = {
            "bm25": {
                "status": "healthy",
                "endpoint": f"{settings.search.opensearch_host}:{settings.search.opensearch_port}",
                "last_check": "2024-01-01T00:00:00Z"
            },
            "vector": {
                "status": "healthy",
                "endpoint": f"{settings.search.qdrant_host}:{settings.search.qdrant_port}",
                "last_check": "2024-01-01T00:00:00Z"
            },
            "llm": {
                "status": "healthy",
                "provider": settings.llm.provider,
                "last_check": "2024-01-01T00:00:00Z"
            },
            "database": {
                "status": "healthy",
                "endpoint": f"{settings.database.host}:{settings.database.port}",
                "last_check": "2024-01-01T00:00:00Z"
            }
        }
        
        logger.info("Detailed health check requested", tenant_id=tenant_id)
        
        return {
            "status": "success",
            "tenant_id": tenant_id,
            "timestamp": "2024-01-01T00:00:00Z",
            "services": health_status
        }
        
    except Exception as e:
        logger.error("Failed to get detailed health status", error=str(e), tenant_id=tenant_id)
        raise HTTPException(status_code=500, detail="Failed to get detailed health status")
