#!/usr/bin/env python3
"""Command-line interface for email ingestion pipeline."""

import asyncio
import argparse
import json
import sys
from pathlib import Path
from typing import Optional
import structlog

from .ingestion import run_ingestion_batch
from .ingestion.models import BatchManifest
from .config.settings import get_settings

logger = structlog.get_logger(__name__)


async def run_dropbox_ingestion(
    dropbox_path: str,
    tenant_id: str,
    batch_size: int = 100
) -> None:
    """Run ingestion from a dropbox folder."""
    dropbox = Path(dropbox_path)
    
    if not dropbox.exists():
        logger.error("Dropbox path does not exist", path=dropbox_path)
        sys.exit(1)
    
    if not dropbox.is_dir():
        logger.error("Dropbox path is not a directory", path=dropbox_path)
        sys.exit(1)
    
    # Find all email files
    email_files = []
    for file_path in dropbox.rglob("*.eml"):
        email_files.append({
            "file_path": str(file_path),
            "file_name": file_path.name,
            "file_size": file_path.stat().st_size
        })
    
    if not email_files:
        logger.error("No .eml files found in dropbox", path=dropbox_path)
        sys.exit(1)
    
    logger.info("Found email files", count=len(email_files), path=dropbox_path)
    
    # Create manifest
    manifest = BatchManifest(
        batch_id=f"cli_batch_{int(asyncio.get_event_loop().time())}",
        tenant_id=tenant_id,
        source_type="dropbox",
        source_path=str(dropbox),
        total_files=len(email_files),
        file_manifest=email_files,
        created_at=asyncio.get_event_loop().time()
    )
    
    # Run ingestion
    try:
        metrics = await run_ingestion_batch(manifest, tenant_id)
        logger.info("Ingestion completed successfully", metrics=metrics.dict())
    except Exception as exc:
        logger.error("Ingestion failed", exc_info=exc)
        sys.exit(1)


async def run_manifest_ingestion(
    manifest_path: str,
    tenant_id: str
) -> None:
    """Run ingestion from a manifest file."""
    manifest_file = Path(manifest_path)
    
    if not manifest_file.exists():
        logger.error("Manifest file does not exist", path=manifest_path)
        sys.exit(1)
    
    try:
        with open(manifest_file, 'r') as f:
            manifest_data = json.load(f)
        
        # Validate manifest
        required_fields = ["batch_id", "file_manifest"]
        for field in required_fields:
            if field not in manifest_data:
                logger.error("Missing required field in manifest", field=field)
                sys.exit(1)
        
        # Create manifest object
        manifest = BatchManifest(
            batch_id=manifest_data["batch_id"],
            tenant_id=tenant_id,
            source_type=manifest_data.get("source_type", "manifest"),
            source_path=str(manifest_file),
            total_files=len(manifest_data["file_manifest"]),
            file_manifest=manifest_data["file_manifest"],
            created_at=asyncio.get_event_loop().time()
        )
        
        logger.info("Loaded manifest", batch_id=manifest.batch_id, file_count=manifest.total_files)
        
        # Run ingestion
        metrics = await run_ingestion_batch(manifest, tenant_id)
        logger.info("Ingestion completed successfully", metrics=metrics.dict())
        
    except json.JSONDecodeError as exc:
        logger.error("Invalid JSON in manifest file", exc_info=exc)
        sys.exit(1)
    except Exception as exc:
        logger.error("Ingestion failed", exc_info=exc)
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Email Ingestion Pipeline CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Ingest from dropbox folder
  python -m app.cli dropbox /path/to/emails --tenant tenant1
  
  # Ingest from manifest file
  python -m app.cli manifest /path/to/manifest.json --tenant tenant1
  
  # Ingest with custom batch size
  python -m app.cli dropbox /path/to/emails --tenant tenant1 --batch-size 50
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Ingestion command")
    
    # Dropbox command
    dropbox_parser = subparsers.add_parser("dropbox", help="Ingest from dropbox folder")
    dropbox_parser.add_argument("path", help="Path to dropbox folder containing .eml files")
    dropbox_parser.add_argument("--tenant", required=True, help="Tenant ID")
    dropbox_parser.add_argument("--batch-size", type=int, default=100, help="Batch size for processing")
    
    # Manifest command
    manifest_parser = subparsers.add_parser("manifest", help="Ingest from manifest file")
    manifest_parser.add_argument("path", help="Path to manifest JSON file")
    manifest_parser.add_argument("--tenant", required=True, help="Tenant ID")
    
    # Global options
    parser.add_argument("--log-level", default="INFO", 
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Log level")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Configure logging
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer()
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Set log level
    import logging
    logging.basicConfig(level=getattr(logging, args.log_level))
    
    # Run appropriate command
    try:
        if args.command == "dropbox":
            asyncio.run(run_dropbox_ingestion(args.path, args.tenant, args.batch_size))
        elif args.command == "manifest":
            asyncio.run(run_manifest_ingestion(args.path, args.tenant))
        else:
            logger.error("Unknown command", command=args.command)
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Ingestion interrupted by user")
        sys.exit(0)
    except Exception as exc:
        logger.error("Unexpected error", exc_info=exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
