"""Main FastAPI application for Logistics Email AI."""

import time
from contextlib import asynccontextmanager
from typing import Dict, Any
import os

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import structlog

from app.config.settings import get_settings
from app.routers import health, draft, eval_router, ingestion, search_metrics, search_qa, ui, debug, api_alias
from app.middleware.logging import LoggingMiddleware
from app.middleware.tenant_isolation import TenantIsolationMiddleware
from app.middleware.rate_limit import RateLimitMiddleware

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting Logistics Email AI application")
    settings = get_settings()
    logger.info("Application settings loaded", 
                environment=settings.environment,
                log_level=settings.monitoring.log_level)
    
    # Initialize search QA service with settings
    from app.services.search_qa import initialize_search_qa
    search_config = {
        "quality_thresholds_hit_at_5": settings.search.quality_thresholds_hit_at_5,
        "quality_thresholds_mrr_at_10": settings.search.quality_thresholds_mrr_at_10,
        "quality_thresholds_ndcg_at_10": settings.search.quality_thresholds_ndcg_at_10
    }
    initialize_search_qa(search_config)
    logger.info("Search QA service initialized", config=search_config)
    
    yield
    
    # Shutdown
    logger.info("Shutting down Logistics Email AI application")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()
    
    app = FastAPI(
        title="Logistics Email AI",
        description="Intelligent email processing with grounded responses",
        version="0.1.0",
        docs_url="/docs" if settings.environment != "production" else None,
        redoc_url="/redoc" if settings.environment != "production" else None,
        lifespan=lifespan
    )
    
    # Add CORS middleware with strict mode settings
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.security.cors_allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Add environment to app state for tenant resolution
    app.state.env = os.getenv("ENV", "dev")
    
    # Add middleware
    app.add_middleware(LoggingMiddleware)
    app.add_middleware(TenantIsolationMiddleware)
    app.add_middleware(RateLimitMiddleware, requests_per_minute=60)
    
    # Add CSP headers middleware
    @app.middleware("http")
    async def add_csp_headers(request: Request, call_next):
        """Add Content Security Policy headers."""
        response = await call_next(request)
        response.headers["Content-Security-Policy"] = settings.security.csp
        return response
    
    # Add exception handlers
    from app.exceptions import SearchDependencyUnavailable, LlmDependencyUnavailable, NoEvidenceFoundError, EvalGateBlockedError
    
    @app.exception_handler(SearchDependencyUnavailable)
    async def search_dependency_handler(request: Request, exc: SearchDependencyUnavailable):
        """Handle search dependency unavailability."""
        logger.warning("Search dependency unavailable", 
                      service=exc.service,
                      action=exc.action,
                      trace_id=exc.trace_id,
                      tenant_id=exc.tenant_id)
        return JSONResponse(
            status_code=503,
            content=exc.to_dict()
        )
    
    @app.exception_handler(LlmDependencyUnavailable)
    async def llm_dependency_handler(request: Request, exc: LlmDependencyUnavailable):
        """Handle LLM dependency unavailability."""
        logger.warning("LLM dependency unavailable", 
                      provider=exc.provider,
                      trace_id=exc.trace_id,
                      tenant_id=exc.tenant_id)
        return JSONResponse(
            status_code=503,
            content=exc.to_dict()
        )
    
    @app.exception_handler(NoEvidenceFoundError)
    async def no_evidence_handler(request: Request, exc: NoEvidenceFoundError):
        """Handle no evidence found errors."""
        logger.warning("No evidence found", 
                      query=exc.query,
                      trace_id=exc.trace_id,
                      tenant_id=exc.tenant_id)
        return JSONResponse(
            status_code=404,
            content=exc.to_dict()
        )
    
    @app.exception_handler(EvalGateBlockedError)
    async def eval_gate_blocked_handler(request: Request, exc: EvalGateBlockedError):
        """Handle evaluation gate blocked errors."""
        logger.warning("Draft blocked by evaluation gate", 
                      scores=exc.scores,
                      reasons=exc.reasons,
                      trace_id=exc.trace_id,
                      tenant_id=exc.tenant_id)
        return JSONResponse(
            status_code=422,
            content=exc.to_dict()
        )
    
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        """Global exception handler."""
        logger.error("Unhandled exception", 
                    exc_info=exc,
                    path=request.url.path,
                    method=request.method)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"}
        )
    
    # Include routers
    app.include_router(health.router, prefix="/health", tags=["health"])
    app.include_router(draft.router, prefix="/draft", tags=["draft"])
    app.include_router(eval_router.router, prefix="/eval", tags=["evaluation"])
    app.include_router(ingestion.router, prefix="/ingestion", tags=["ingestion"])
    app.include_router(search_metrics.router, prefix="/search-metrics", tags=["search-metrics"])
    app.include_router(search_qa.router, prefix="/search-qa", tags=["search-qa"])
    app.include_router(ui.router, tags=["ui"])
    app.include_router(debug.router, tags=["debug"])
    app.include_router(api_alias.router, tags=["api-alias"])
    
    return app


app = create_app()


@app.get("/")
async def root() -> Dict[str, Any]:
    """Root endpoint."""
    return {
        "message": "Logistics Email AI",
        "version": "0.1.0",
        "status": "running"
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
