"""Checkpoint manager for email ingestion pipeline."""

import asyncio
import json
import os
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from pathlib import Path
import structlog

from .models import CheckpointInfo, ProcessingStatus

logger = structlog.get_logger(__name__)


class CheckpointManager:
    """Manages checkpoints for batch processing."""
    
    def __init__(self, checkpoint_dir: str = "./data/ingestion/checkpoints"):
        """Initialize checkpoint manager."""
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.active_checkpoints = {}  # batch_id -> CheckpointInfo
    
    async def create_checkpoint(self, batch_id: str, tenant_id: str, total_emails: int) -> CheckpointInfo:
        """Create a new checkpoint for batch processing."""
        try:
            checkpoint_id = f"checkpoint_{batch_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            
            checkpoint = CheckpointInfo(
                checkpoint_id=checkpoint_id,
                batch_id=batch_id,
                tenant_id=tenant_id,
                total_emails=total_emails,
                processed_emails=0,
                failed_emails=0,
                quarantined_emails=0
            )
            
            # Store in memory
            self.active_checkpoints[batch_id] = checkpoint
            
            # Save to disk
            await self._save_checkpoint(checkpoint)
            
            logger.info("Checkpoint created",
                       checkpoint_id=checkpoint_id,
                       batch_id=batch_id,
                       total_emails=total_emails)
            
            return checkpoint
            
        except Exception as exc:
            logger.error("Failed to create checkpoint",
                        batch_id=batch_id,
                        exc_info=exc)
            raise
    
    async def update_checkpoint(self, batch_id: str, email_file: str, position: int):
        """Update checkpoint with progress information."""
        try:
            if batch_id not in self.active_checkpoints:
                logger.warning("No active checkpoint found for batch", batch_id=batch_id)
                return
            
            checkpoint = self.active_checkpoints[batch_id]
            checkpoint.processed_emails += 1
            checkpoint.last_processed_email = email_file
            checkpoint.last_processed_position = position
            checkpoint.updated_at = datetime.utcnow()
            
            # Save to disk
            await self._save_checkpoint(checkpoint)
            
            logger.debug("Checkpoint updated",
                        batch_id=batch_id,
                        processed_emails=checkpoint.processed_emails,
                        last_file=email_file)
            
        except Exception as exc:
            logger.error("Failed to update checkpoint",
                        batch_id=batch_id,
                        exc_info=exc)
    
    async def mark_email_failed(self, batch_id: str, error_type: str = None):
        """Mark an email as failed in the checkpoint."""
        try:
            if batch_id not in self.active_checkpoints:
                return
            
            checkpoint = self.active_checkpoints[batch_id]
            checkpoint.failed_emails += 1
            checkpoint.error_count += 1
            checkpoint.updated_at = datetime.utcnow()
            
            # Save to disk
            await self._save_checkpoint(checkpoint)
            
        except Exception as exc:
            logger.error("Failed to mark email as failed",
                        batch_id=batch_id,
                        exc_info=exc)
    
    async def mark_email_quarantined(self, batch_id: str):
        """Mark an email as quarantined in the checkpoint."""
        try:
            if batch_id not in self.active_checkpoints:
                return
            
            checkpoint = self.active_checkpoints[batch_id]
            checkpoint.quarantined_emails += 1
            checkpoint.updated_at = datetime.utcnow()
            
            # Save to disk
            await self._save_checkpoint(checkpoint)
            
        except Exception as exc:
            logger.error("Failed to mark email as quarantined",
                        batch_id=batch_id,
                        exc_info=exc)
    
    async def complete_checkpoint(self, batch_id: str):
        """Mark checkpoint as completed."""
        try:
            if batch_id not in self.active_checkpoints:
                return
            
            checkpoint = self.active_checkpoints[batch_id]
            checkpoint.is_complete = True
            checkpoint.completed_at = datetime.utcnow()
            checkpoint.updated_at = datetime.utcnow()
            
            # Save to disk
            await self._save_checkpoint(checkpoint)
            
            # Move to completed state
            await self._archive_checkpoint(checkpoint)
            
            # Remove from active checkpoints
            del self.active_checkpoints[batch_id]
            
            logger.info("Checkpoint completed",
                       batch_id=batch_id,
                       total_processed=checkpoint.processed_emails)
            
        except Exception as exc:
            logger.error("Failed to complete checkpoint",
                        batch_id=batch_id,
                        exc_info=exc)
    
    async def should_skip_email(self, batch_id: str, email_file: str) -> bool:
        """Check if an email should be skipped based on checkpoint."""
        try:
            if batch_id not in self.active_checkpoints:
                return False
            
            checkpoint = self.active_checkpoints[batch_id]
            
            # Check if this file was already processed
            if checkpoint.last_processed_email == email_file:
                return True
            
            # Check if we're resuming and this file is before our last position
            # This is a simplified check - in a real implementation, you'd need
            # to maintain a more sophisticated file ordering
            return False
            
        except Exception as exc:
            logger.error("Failed to check if email should be skipped",
                        batch_id=batch_id,
                        email_file=email_file,
                        exc_info=exc)
            return False
    
    async def get_checkpoint(self, batch_id: str) -> Optional[CheckpointInfo]:
        """Get checkpoint information for a batch."""
        try:
            # Check active checkpoints first
            if batch_id in self.active_checkpoints:
                return self.active_checkpoints[batch_id]
            
            # Check disk for completed checkpoints
            checkpoint_file = self.checkpoint_dir / f"{batch_id}.json"
            if checkpoint_file.exists():
                checkpoint = await self._load_checkpoint(checkpoint_file)
                return checkpoint
            
            return None
            
        except Exception as exc:
            logger.error("Failed to get checkpoint",
                        batch_id=batch_id,
                        exc_info=exc)
            return None
    
    async def list_checkpoints(self, tenant_id: Optional[str] = None) -> List[CheckpointInfo]:
        """List all checkpoints, optionally filtered by tenant."""
        try:
            checkpoints = []
            
            # Add active checkpoints
            for checkpoint in self.active_checkpoints.values():
                if tenant_id is None or checkpoint.tenant_id == tenant_id:
                    checkpoints.append(checkpoint)
            
            # Add completed checkpoints from disk
            for checkpoint_file in self.checkpoint_dir.glob("*.json"):
                try:
                    checkpoint = await self._load_checkpoint(checkpoint_file)
                    if tenant_id is None or checkpoint.tenant_id == tenant_id:
                        checkpoints.append(checkpoint)
                except Exception:
                    # Skip corrupted checkpoint files
                    continue
            
            # Sort by creation date
            checkpoints.sort(key=lambda x: x.created_at, reverse=True)
            
            return checkpoints
            
        except Exception as exc:
            logger.error("Failed to list checkpoints", exc_info=exc)
            return []
    
    async def cleanup_old_checkpoints(self, max_age_days: int = 7):
        """Clean up old checkpoint files."""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=max_age_days)
            cleaned_count = 0
            
            for checkpoint_file in self.checkpoint_dir.glob("*.json"):
                try:
                    # Check file modification time
                    file_mtime = datetime.fromtimestamp(checkpoint_file.stat().st_mtime)
                    
                    if file_mtime < cutoff_date:
                        checkpoint_file.unlink()
                        cleaned_count += 1
                        
                except Exception:
                    # Skip files that can't be processed
                    continue
            
            if cleaned_count > 0:
                logger.info("Cleaned up old checkpoints",
                           cleaned_count=cleaned_count,
                           max_age_days=max_age_days)
            
        except Exception as exc:
            logger.error("Failed to cleanup old checkpoints", exc_info=exc)
    
    async def _save_checkpoint(self, checkpoint: CheckpointInfo):
        """Save checkpoint to disk."""
        try:
            checkpoint_file = self.checkpoint_dir / f"{checkpoint.batch_id}.json"
            
            # Convert to dict for JSON serialization
            checkpoint_dict = {
                'checkpoint_id': checkpoint.checkpoint_id,
                'batch_id': checkpoint.batch_id,
                'tenant_id': checkpoint.tenant_id,
                'total_emails': checkpoint.total_emails,
                'processed_emails': checkpoint.processed_emails,
                'failed_emails': checkpoint.failed_emails,
                'quarantined_emails': checkpoint.quarantined_emails,
                'last_processed_email': checkpoint.last_processed_email,
                'last_processed_position': checkpoint.last_processed_position,
                'is_complete': checkpoint.is_complete,
                'error_count': checkpoint.error_count,
                'created_at': checkpoint.created_at.isoformat(),
                'updated_at': checkpoint.updated_at.isoformat(),
                'completed_at': checkpoint.completed_at.isoformat() if checkpoint.completed_at else None
            }
            
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint_dict, f, indent=2)
            
        except Exception as exc:
            logger.error("Failed to save checkpoint",
                        checkpoint_id=checkpoint.checkpoint_id,
                        exc_info=exc)
    
    async def _load_checkpoint(self, checkpoint_file: Path) -> CheckpointInfo:
        """Load checkpoint from disk."""
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint_dict = json.load(f)
            
            # Parse datetime fields
            created_at = datetime.fromisoformat(checkpoint_dict['created_at'])
            updated_at = datetime.fromisoformat(checkpoint_dict['updated_at'])
            completed_at = None
            if checkpoint_dict.get('completed_at'):
                completed_at = datetime.fromisoformat(checkpoint_dict['completed_at'])
            
            return CheckpointInfo(
                checkpoint_id=checkpoint_dict['checkpoint_id'],
                batch_id=checkpoint_dict['batch_id'],
                tenant_id=checkpoint_dict['tenant_id'],
                total_emails=checkpoint_dict['total_emails'],
                processed_emails=checkpoint_dict['processed_emails'],
                failed_emails=checkpoint_dict['failed_emails'],
                quarantined_emails=checkpoint_dict['quarantined_emails'],
                last_processed_email=checkpoint_dict.get('last_processed_email'),
                last_processed_position=checkpoint_dict.get('last_processed_position', 0),
                is_complete=checkpoint_dict.get('is_complete', False),
                error_count=checkpoint_dict.get('error_count', 0),
                created_at=created_at,
                updated_at=updated_at,
                completed_at=completed_at
            )
            
        except Exception as exc:
            logger.error("Failed to load checkpoint",
                        checkpoint_file=str(checkpoint_file),
                        exc_info=exc)
            raise
    
    async def _archive_checkpoint(self, checkpoint: CheckpointInfo):
        """Archive completed checkpoint."""
        try:
            # Move to completed directory
            completed_dir = self.checkpoint_dir / "completed"
            completed_dir.mkdir(exist_ok=True)
            
            # Rename checkpoint file
            source_file = self.checkpoint_dir / f"{checkpoint.batch_id}.json"
            target_file = completed_dir / f"{checkpoint.batch_id}_completed.json"
            
            if source_file.exists():
                source_file.rename(target_file)
            
        except Exception as exc:
            logger.error("Failed to archive checkpoint",
                        checkpoint_id=checkpoint.checkpoint_id,
                        exc_info=exc)


class IdempotencyManager:
    """Manages idempotency for email processing."""
    
    def __init__(self):
        """Initialize idempotency manager."""
        self.processed_emails = set()  # Set of processed email identifiers
        self.processing_locks = {}  # email_id -> lock
    
    async def is_email_processed(self, email_identifier: str) -> bool:
        """Check if an email has already been processed."""
        return email_identifier in self.processed_emails
    
    async def mark_email_processed(self, email_identifier: str):
        """Mark an email as processed."""
        self.processed_emails.add(email_identifier)
    
    async def acquire_processing_lock(self, email_id: str) -> bool:
        """Acquire a lock for processing an email."""
        try:
            if email_id in self.processing_locks:
                return False
            
            # Create a lock for this email
            self.processing_locks[email_id] = asyncio.Lock()
            return True
            
        except Exception as exc:
            logger.error("Failed to acquire processing lock",
                        email_id=email_id,
                        exc_info=exc)
            return False
    
    async def release_processing_lock(self, email_id: str):
        """Release the processing lock for an email."""
        try:
            if email_id in self.processing_locks:
                del self.processing_locks[email_id]
            
        except Exception as exc:
            logger.error("Failed to release processing lock",
                        email_id=email_id,
                        exc_info=exc)
    
    async def get_processing_stats(self) -> Dict[str, Any]:
        """Get idempotency processing statistics."""
        try:
            return {
                'total_processed_emails': len(self.processed_emails),
                'active_processing_locks': len(self.processing_locks),
                'processed_emails_sample': list(self.processed_emails)[:10]  # Sample of processed emails
            }
            
        except Exception as exc:
            logger.error("Failed to get processing stats", exc_info=exc)
            return {}
    
    async def cleanup_old_records(self, max_age_hours: int = 24):
        """Clean up old processing records."""
        try:
            # This is a simplified cleanup - in a real implementation,
            # you'd want to track timestamps for processed emails
            # and remove old ones based on age
            
            # For now, just clear the sets periodically
            if len(self.processed_emails) > 10000:  # Arbitrary threshold
                self.processed_emails.clear()
                logger.info("Cleared processed emails cache")
            
        except Exception as exc:
            logger.error("Failed to cleanup old records", exc_info=exc)
