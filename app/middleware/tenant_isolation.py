"""Tenant isolation middleware for multi-tenant security."""

from typing import Callable, Optional
from fastapi import Request, Response, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
import structlog
import uuid

logger = structlog.get_logger(__name__)

DEV_DEFAULT_TENANT = "demo"

def resolve_tenant_id(request) -> Optional[str]:
    """Resolve tenant ID from multiple sources with dev-friendly fallbacks."""
    # 1) header wins for APIs
    tid = request.headers.get("X-Tenant-ID")

    # 2) query/cookie support for UI
    if not tid:
        tid = request.query_params.get("tenant") or request.cookies.get("tenant_id")

    if not tid:
        # dev default for UI landing when nothing provided
        if (getattr(request.app.state, "env", "dev") == "dev"):
            tid = DEV_DEFAULT_TENANT
        else:
            return None

    # Accept UUIDs as-is
    try:
        uuid.UUID(str(tid))
        return str(tid)
    except Exception:
        pass

    # In dev, allow slug -> deterministic UUID (no DB needed)
    if getattr(request.app.state, "env", "dev") == "dev":
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"tenant:{tid}"))

    return None


class TenantIsolationMiddleware(BaseHTTPMiddleware):
    """Middleware for enforcing tenant isolation."""
    
    def __init__(self, app, header_name: str = "X-Tenant-ID"):
        super().__init__(app)
        self.header_name = header_name
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and enforce tenant isolation."""
        # Get tenant ID from multiple sources
        tenant_id = resolve_tenant_id(request)
        
        # Skip tenant check for health and public endpoints
        if self._is_public_endpoint(request.url.path):
            return await call_next(request)
        
        # Validate tenant ID
        if not tenant_id:
            logger.warning(
                "Missing tenant ID",
                path=request.url.path,
                method=request.method,
            )
            return Response(
                content='{"detail": "Missing or invalid tenant"}',
                status_code=status.HTTP_401_UNAUTHORIZED,
                media_type="application/json"
            )
        
        # Add tenant ID to request state
        request.state.tenant_id = tenant_id
        
        # Log tenant access
        logger.info(
            "Tenant access",
            tenant_id=tenant_id,
            path=request.url.path,
            method=request.method,
        )
        
        # Process request
        response = await call_next(request)
        
        # Add tenant ID to response headers for debugging
        response.headers["X-Tenant-Resolved"] = tenant_id
        
        return response
    
    def _is_public_endpoint(self, path: str) -> bool:
        """Check if endpoint is public (no tenant required)."""
        public_paths = [
            "/health",
            "/docs",
            "/redoc",
            "/openapi.json",
            "/",
        ]
        return any(path.startswith(public_path) for public_path in public_paths)


def get_tenant_id(request: Request) -> str:
    """Get tenant ID from request state."""
    tenant_id = getattr(request.state, "tenant_id", None)
    if not tenant_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Tenant ID not found in request"
        )
    return tenant_id


def require_tenant_id(func):
    """Decorator to require tenant ID in function."""
    async def wrapper(*args, **kwargs):
        # This would be used in dependency injection
        # For now, just return the function
        return await func(*args, **kwargs)
    return wrapper
