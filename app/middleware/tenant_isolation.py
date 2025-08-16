"""Tenant isolation middleware for multi-tenant security."""

from typing import Callable, Optional
from fastapi import Request, Response, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
import structlog

logger = structlog.get_logger(__name__)


class TenantIsolationMiddleware(BaseHTTPMiddleware):
    """Middleware for enforcing tenant isolation."""
    
    def __init__(self, app, header_name: str = "X-Tenant-ID"):
        super().__init__(app)
        self.header_name = header_name
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and enforce tenant isolation."""
        # Get tenant ID from header
        tenant_id = request.headers.get(self.header_name)
        
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
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Tenant ID is required"
            )
        
        # Validate tenant ID format (UUID)
        if not self._is_valid_uuid(tenant_id):
            logger.warning(
                "Invalid tenant ID format",
                tenant_id=tenant_id,
                path=request.url.path,
                method=request.method,
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid tenant ID format"
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
        response.headers["X-Tenant-ID"] = tenant_id
        
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
    
    def _is_valid_uuid(self, uuid_string: str) -> bool:
        """Validate UUID format."""
        try:
            import uuid
            uuid.UUID(uuid_string)
            return True
        except ValueError:
            return False


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
