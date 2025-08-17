"""UI routes for the demo application."""

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from typing import Optional
import structlog

from app.middleware.tenant_isolation import resolve_tenant_id

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/ui", tags=["ui"])

# Initialize templates
templates = Jinja2Templates(directory="app/templates")


@router.get("/upload", response_class=HTMLResponse)
async def upload_page(request: Request):
    """Upload page with drag-and-drop interface."""
    try:
        tenant_id = resolve_tenant_id(request)
        return templates.TemplateResponse("upload.html", {
            "request": request,
            "tenant_id": tenant_id,
            "page_title": "Upload Documents"
        })
    except Exception as e:
        logger.error("Failed to render upload page", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to render upload page")


@router.get("/search", response_class=HTMLResponse)
async def search_page(request: Request):
    """Search page with query interface and filters."""
    try:
        tenant_id = resolve_tenant_id(request)
        return templates.TemplateResponse("search.html", {
            "request": request,
            "tenant_id": tenant_id,
            "page_title": "Search Documents"
        })
    except Exception as e:
        logger.error("Failed to render search page", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to render search page")


@router.get("/draft", response_class=HTMLResponse)
async def draft_page(request: Request):
    """Draft page with prompt interface and streaming."""
    try:
        tenant_id = resolve_tenant_id(request)
        return templates.TemplateResponse("draft.html", {
            "request": request,
            "tenant_id": tenant_id,
            "page_title": "Generate Draft"
        })
    except Exception as e:
        logger.error("Failed to render draft page", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to render draft page")


@router.get("/", response_class=HTMLResponse)
async def index_page(request: Request):
    """Main index page with navigation."""
    try:
        tenant_id = resolve_tenant_id(request)
        return templates.TemplateResponse("index.html", {
            "request": request,
            "tenant_id": tenant_id,
            "page_title": "EvilLLM - Document Intelligence"
        })
    except Exception as e:
        logger.error("Failed to render index page", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to render index page")
