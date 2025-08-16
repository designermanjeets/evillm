"""Logging middleware for structured logging with trace IDs."""

import time
import uuid
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
import structlog

logger = structlog.get_logger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for structured logging with trace IDs."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and log structured information."""
        # Generate trace ID
        trace_id = str(uuid.uuid4())
        
        # Add trace ID to request state
        request.state.trace_id = trace_id
        
        # Start timing
        start_time = time.time()
        
        # Log request
        logger.info(
            "Request started",
            trace_id=trace_id,
            method=request.method,
            url=str(request.url),
            client_ip=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent"),
        )
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate processing time
            process_time = time.time() - start_time
            
            # Log response
            logger.info(
                "Request completed",
                trace_id=trace_id,
                method=request.method,
                url=str(request.url),
                status_code=response.status_code,
                process_time=process_time,
            )
            
            # Add trace ID to response headers
            response.headers["X-Trace-ID"] = trace_id
            response.headers["X-Process-Time"] = str(process_time)
            
            return response
            
        except Exception as exc:
            # Calculate processing time
            process_time = time.time() - start_time
            
            # Log error
            logger.error(
                "Request failed",
                trace_id=trace_id,
                method=request.method,
                url=str(request.url),
                error=str(exc),
                process_time=process_time,
                exc_info=exc,
            )
            
            # Re-raise exception
            raise
