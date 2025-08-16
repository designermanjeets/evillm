"""Ingestion router for email processing pipeline."""

import asyncio
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import structlog

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ..ingestion import IngestionPipeline, run_ingestion_batch
from ..ingestion.models import BatchManifest, IngestionMetrics, ProcessingStatus
from ..config.settings import get_settings
from ..middleware.tenant_isolation import get_tenant_id

logger = structlog.get_logger(__name__)
router = APIRouter()


class IngestionRequest(BaseModel):
    """Request model for starting ingestion."""
    source_type: str = Field(..., description="Source type: 'dropbox', 'manifest', 'imap'")
    source_path: str = Field(..., description="Path to source files or manifest")
    batch_size: int = Field(default=100, description="Number of emails per batch")
    tenant_id: Optional[str] = Field(None, description="Tenant ID (auto-detected if not provided)")


class IngestionResponse(BaseModel):
    """Response model for ingestion operations."""
    batch_id: str
    status: ProcessingStatus
    message: str
    metrics: Optional[IngestionMetrics] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class BatchStatusResponse(BaseModel):
    """Response model for batch status."""
    batch_id: str
    status: ProcessingStatus
    progress: Dict[str, Any]
    metrics: Optional[IngestionMetrics] = None
    last_updated: datetime


@router.post("/ingest", response_model=IngestionResponse)
async def start_ingestion(
    request: IngestionRequest,
    background_tasks: BackgroundTasks,
    tenant_id: str = Depends(get_tenant_id)
):
    """Start a new ingestion batch."""
    try:
        logger.info("Starting ingestion request", 
                   source_type=request.source_type,
                   source_path=request.source_path,
                   tenant_id=tenant_id)
        
        # Validate source path
        if request.source_type == "dropbox":
            source_path = Path(request.source_path)
            if not source_path.exists():
                raise HTTPException(status_code=400, detail="Source path does not exist")
            
            # Create manifest from dropbox
            manifest = await _create_manifest_from_dropbox(source_path, tenant_id)
        elif request.source_type == "manifest":
            manifest = await _load_manifest_from_file(request.source_path, tenant_id)
        else:
            raise HTTPException(status_code=400, detail="Unsupported source type")
        
        # Start ingestion in background
        background_tasks.add_task(
            run_ingestion_batch,
            manifest=manifest,
            tenant_id=tenant_id
        )
        
        return IngestionResponse(
            batch_id=manifest.batch_id,
            status=ProcessingStatus.PROCESSING,
            message="Ingestion started successfully"
        )
        
    except Exception as exc:
        logger.error("Failed to start ingestion", exc_info=exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/status/{batch_id}", response_model=BatchStatusResponse)
async def get_batch_status(
    batch_id: str,
    tenant_id: str = Depends(get_tenant_id)
):
    """Get the status of an ingestion batch."""
    try:
        # This would typically query the database for batch status
        # For now, return a mock response
        return BatchStatusResponse(
            batch_id=batch_id,
            status=ProcessingStatus.PROCESSING,
            progress={
                "total_emails": 100,
                "processed_emails": 45,
                "failed_emails": 2,
                "quarantined_emails": 1
            },
            last_updated=datetime.utcnow()
        )
    except Exception as exc:
        logger.error("Failed to get batch status", exc_info=exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/batches", response_model=List[BatchStatusResponse])
async def list_batches(
    tenant_id: str = Depends(get_tenant_id),
    limit: int = Query(default=10, le=100),
    offset: int = Query(default=0, ge=0)
):
    """List recent ingestion batches for the tenant."""
    try:
        # This would typically query the database for batch history
        # For now, return a mock response
        return [
            BatchStatusResponse(
                batch_id=f"batch_{i}",
                status=ProcessingStatus.COMPLETED,
                progress={
                    "total_emails": 100,
                    "processed_emails": 100,
                    "failed_emails": 0,
                    "quarantined_emails": 0
                },
                last_updated=datetime.utcnow()
            )
            for i in range(1, min(limit + 1, 6))
        ]
    except Exception as exc:
        logger.error("Failed to list batches", exc_info=exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/retry/{batch_id}")
async def retry_failed_emails(
    batch_id: str,
    tenant_id: str = Depends(get_tenant_id)
):
    """Retry processing of failed emails in a batch."""
    try:
        logger.info("Retrying failed emails", batch_id=batch_id, tenant_id=tenant_id)
        
        # This would typically query failed emails and retry them
        # For now, return success
        return {"message": "Retry initiated", "batch_id": batch_id}
        
    except Exception as exc:
        logger.error("Failed to retry batch", exc_info=exc)
        raise HTTPException(status_code=500, detail=str(exc))


async def _create_manifest_from_dropbox(dropbox_path: Path, tenant_id: str) -> BatchManifest:
    """Create a batch manifest from a dropbox folder."""
    from ..ingestion.models import generate_batch_id
    
    # Find all email files
    email_files = []
    for file_path in dropbox_path.rglob("*.eml"):
        email_files.append({
            "file_path": str(file_path),
            "file_name": file_path.name,
            "file_size": file_path.stat().st_size
        })
    
    if not email_files:
        raise HTTPException(status_code=400, detail="No .eml files found in dropbox")
    
    return BatchManifest(
        batch_id=generate_batch_id(),
        tenant_id=tenant_id,
        source_type="dropbox",
        source_path=str(dropbox_path),
        total_files=len(email_files),
        file_manifest=email_files,
        created_at=datetime.utcnow()
    )


async def _load_manifest_from_file(manifest_path: str, tenant_id: str) -> BatchManifest:
    """Load a batch manifest from a JSON file."""
    try:
        with open(manifest_path, 'r') as f:
            manifest_data = json.load(f)
        
        # Validate and create manifest
        manifest = BatchManifest(
            batch_id=manifest_data.get("batch_id"),
            tenant_id=tenant_id,
            source_type=manifest_data.get("source_type", "manifest"),
            source_path=manifest_path,
            total_files=manifest_data.get("total_files", 0),
            file_manifest=manifest_data.get("file_manifest", []),
            created_at=datetime.utcnow()
        )
        
        return manifest
        
    except Exception as exc:
        logger.error("Failed to load manifest", exc_info=exc)
        raise HTTPException(status_code=400, detail=f"Invalid manifest file: {str(exc)}")
