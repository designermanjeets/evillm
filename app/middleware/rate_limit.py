"""Rate limiting middleware for security."""

import time
from typing import Dict, Tuple
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
import structlog

logger = structlog.get_logger(__name__)


class RateLimitMiddleware:
    """Rate limiting middleware with per-IP and per-tenant limits."""
    
    def __init__(self, app, requests_per_minute: int = 60):
        self.app = app
        self.requests_per_minute = requests_per_minute
        self.ip_requests: Dict[str, list] = {}
        self.tenant_requests: Dict[str, list] = {}
    
    async def __call__(self, scope, receive, send):
        """Process request with rate limiting."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        # Create a request object for easier handling
        request = Request(scope, receive)
        
        # Get client IP
        client_ip = self._get_client_ip(request)
        
        # Get tenant ID from header
        tenant_id = request.headers.get("X-Tenant-ID", "anonymous")
        
        # Check rate limits
        if not self._check_rate_limit(client_ip, tenant_id):
            logger.warning("Rate limit exceeded", 
                          client_ip=client_ip,
                          tenant_id=tenant_id,
                          path=request.url.path)
            
            # Send rate limit response
            response = JSONResponse(
                status_code=429,
                content={
                    "error": "rate_limit_exceeded",
                    "message": "Too many requests",
                    "retry_after": 60
                },
                headers={"Retry-After": "60"}
            )
            
            await response(scope, receive, send)
            return
        
        # Process request through the app
        await self.app(scope, receive, send)
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address."""
        # Check for forwarded headers
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        # Check for real IP header
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # Use client host
        return request.client.host if request.client else "unknown"
    
    def _check_rate_limit(self, client_ip: str, tenant_id: str) -> bool:
        """Check if request is within rate limits."""
        current_time = time.time()
        window_start = current_time - 60  # 1 minute window
        
        # Clean old requests
        self._cleanup_old_requests(client_ip, tenant_id, window_start)
        
        # Check IP rate limit
        if not self._check_ip_limit(client_ip, current_time):
            return False
        
        # Check tenant rate limit
        if not self._check_tenant_limit(tenant_id, current_time):
            return False
        
        # Record request
        self._record_request(client_ip, tenant_id, current_time)
        
        return True
    
    def _cleanup_old_requests(self, client_ip: str, tenant_id: str, window_start: float):
        """Remove old requests outside the time window."""
        if client_ip in self.ip_requests:
            self.ip_requests[client_ip] = [
                req_time for req_time in self.ip_requests[client_ip]
                if req_time > window_start
            ]
        
        if tenant_id in self.tenant_requests:
            self.tenant_requests[tenant_id] = [
                req_time for req_time in self.tenant_requests[tenant_id]
                if req_time > window_start
            ]
    
    def _check_ip_limit(self, client_ip: str, current_time: float) -> bool:
        """Check IP-based rate limit."""
        if client_ip not in self.ip_requests:
            return True
        
        recent_requests = len(self.ip_requests[client_ip])
        return recent_requests < self.requests_per_minute
    
    def _check_tenant_limit(self, tenant_id: str, current_time: float) -> bool:
        """Check tenant-based rate limit."""
        if tenant_id not in self.tenant_requests:
            return True
        
        recent_requests = len(self.tenant_requests[tenant_id])
        return recent_requests < self.requests_per_minute
    
    def _record_request(self, client_ip: str, tenant_id: str, current_time: float):
        """Record a new request."""
        if client_ip not in self.ip_requests:
            self.ip_requests[client_ip] = []
        self.ip_requests[client_ip].append(current_time)
        
        if tenant_id not in self.tenant_requests:
            self.tenant_requests[tenant_id] = []
        self.tenant_requests[tenant_id].append(current_time)
