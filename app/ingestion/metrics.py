"""Metrics collector for email ingestion pipeline."""

import asyncio
import time
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import structlog

from .models import IngestionMetrics, ProcessingStatus, ErrorType

logger = structlog.get_logger(__name__)


class MetricsCollector:
    """Collects and manages metrics for the ingestion pipeline."""
    
    def __init__(self):
        """Initialize metrics collector."""
        self.current_batch_id = None
        self.batch_start_time = None
        self.batch_end_time = None
        
        # Performance metrics
        self.total_emails = 0
        self.total_attachments = 0
        self.total_chunks = 0
        self.processed_emails = 0
        self.failed_emails = 0
        self.quarantined_emails = 0
        self.duplicate_emails = 0
        
        # Timing metrics
        self.email_processing_times = []
        self.attachment_processing_times = []
        self.chunking_times = []
        
        # Error tracking
        self.error_buckets = {}
        self.retry_counts = {}
        
        # Quality metrics
        self.dedup_ratio = 0.0
        self.ocr_rate = 0.0
        self.success_rate = 0.0
        
        # Throughput metrics
        self.docs_per_second = 0.0
        self.average_latency = 0.0
        
        # Last update time
        self.last_updated = datetime.utcnow()
    
    def start_batch(self, batch_id: str):
        """Start collecting metrics for a new batch."""
        try:
            self.current_batch_id = batch_id
            self.batch_start_time = datetime.utcnow()
            self.batch_end_time = None
            
            # Reset batch-specific metrics
            self.processed_emails = 0
            self.failed_emails = 0
            self.quarantined_emails = 0
            self.duplicate_emails = 0
            
            # Reset timing arrays
            self.email_processing_times = []
            self.attachment_processing_times = []
            self.chunking_times = []
            
            logger.info("Metrics collection started for batch", batch_id=batch_id)
            
        except Exception as exc:
            logger.error("Failed to start metrics collection", exc_info=exc)
    
    def complete_batch(self):
        """Complete metrics collection for the current batch."""
        try:
            self.batch_end_time = datetime.utcnow()
            
            # Calculate final metrics
            self._calculate_batch_metrics()
            
            logger.info("Metrics collection completed for batch",
                       batch_id=self.current_batch_id,
                       total_processed=self.processed_emails)
            
        except Exception as exc:
            logger.error("Failed to complete metrics collection", exc_info=exc)
    
    def record_email_processed(self, processing_time: float, has_attachments: bool, 
                             attachment_count: int, chunk_count: int):
        """Record metrics for a processed email."""
        try:
            self.processed_emails += 1
            self.total_emails += 1
            self.total_attachments += attachment_count
            self.total_chunks += chunk_count
            
            # Record processing time
            self.email_processing_times.append(processing_time)
            
            # Update last modified
            self.last_updated = datetime.utcnow()
            
        except Exception as exc:
            logger.error("Failed to record email processed metrics", exc_info=exc)
    
    def record_email_failed(self, error_type: ErrorType, retry_count: int = 0):
        """Record metrics for a failed email."""
        try:
            self.failed_emails += 1
            self.total_emails += 1
            
            # Update error buckets
            error_key = error_type.value if hasattr(error_type, 'value') else str(error_type)
            self.error_buckets[error_key] = self.error_buckets.get(error_key, 0) + 1
            
            # Update retry counts
            self.retry_counts[error_key] = self.retry_counts.get(error_key, 0) + retry_count
            
            # Update last modified
            self.last_updated = datetime.utcnow()
            
        except Exception as exc:
            logger.error("Failed to record email failed metrics", exc_info=exc)
    
    def record_email_quarantined(self, reason: str):
        """Record metrics for a quarantined email."""
        try:
            self.quarantined_emails += 1
            self.total_emails += 1
            
            # Update error buckets
            self.error_buckets[f"quarantined_{reason}"] = self.error_buckets.get(f"quarantined_{reason}", 0) + 1
            
            # Update last modified
            self.last_updated = datetime.utcnow()
            
        except Exception as exc:
            logger.error("Failed to record email quarantined metrics", exc_info=exc)
    
    def record_duplicate_email(self, duplicate_type: str):
        """Record metrics for a duplicate email."""
        try:
            self.duplicate_emails += 1
            
            # Update error buckets
            self.error_buckets[f"duplicate_{duplicate_type}"] = self.error_buckets.get(f"duplicate_{duplicate_type}", 0) + 1
            
            # Update last modified
            self.last_updated = datetime.utcnow()
            
        except Exception as exc:
            logger.error("Failed to record duplicate email metrics", exc_info=exc)
    
    def record_attachment_processed(self, processing_time: float, ocr_required: bool):
        """Record metrics for a processed attachment."""
        try:
            # Record processing time
            self.attachment_processing_times.append(processing_time)
            
            # Update OCR rate
            if ocr_required:
                # This is a simplified OCR rate calculation
                # In a real implementation, you'd track OCR tasks separately
                pass
            
            # Update last modified
            self.last_updated = datetime.utcnow()
            
        except Exception as exc:
            logger.error("Failed to record attachment processed metrics", exc_info=exc)
    
    def record_chunking_completed(self, chunking_time: float, chunk_count: int):
        """Record metrics for completed chunking."""
        try:
            # Record chunking time
            self.chunking_times.append(chunking_time)
            
            # Update last modified
            self.last_updated = datetime.utcnow()
            
        except Exception as exc:
            logger.error("Failed to record chunking metrics", exc_info=exc)
    
    def update_dedup_ratio(self, ratio: float):
        """Update deduplication ratio."""
        try:
            self.dedup_ratio = ratio
            self.last_updated = datetime.utcnow()
            
        except Exception as exc:
            logger.error("Failed to update dedup ratio", exc_info=exc)
    
    def update_ocr_rate(self, rate: float):
        """Update OCR rate."""
        try:
            self.ocr_rate = rate
            self.last_updated = datetime.utcnow()
            
        except Exception as exc:
            logger.error("Failed to update OCR rate", exc_info=exc)
    
    def get_metrics(self) -> IngestionMetrics:
        """Get current metrics."""
        try:
            # Calculate performance metrics
            total_processing_time = 0.0
            if self.batch_start_time and self.batch_end_time:
                total_processing_time = (self.batch_end_time - self.batch_start_time).total_seconds()
            
            # Calculate throughput
            if total_processing_time > 0:
                self.docs_per_second = self.processed_emails / total_processing_time
            
            # Calculate average latency
            if self.email_processing_times:
                self.average_latency = sum(self.email_processing_times) / len(self.email_processing_times)
            
            # Calculate success rate
            total_attempted = self.processed_emails + self.failed_emails + self.quarantined_emails
            if total_attempted > 0:
                self.success_rate = self.processed_emails / total_attempted
            
            return IngestionMetrics(
                docs_per_second=self.docs_per_second,
                total_processing_time=total_processing_time,
                average_latency=self.average_latency,
                total_emails=self.total_emails,
                total_attachments=self.total_attachments,
                total_chunks=self.total_chunks,
                dedup_ratio=self.dedup_ratio,
                ocr_rate=self.ocr_rate,
                success_rate=self.success_rate,
                error_count=self.failed_emails + self.quarantined_emails,
                quarantine_count=self.quarantined_emails,
                retry_count=sum(self.retry_counts.values()),
                error_buckets=self.error_buckets.copy(),
                batch_start_time=self.batch_start_time,
                batch_end_time=self.batch_end_time,
                last_updated=self.last_updated
            )
            
        except Exception as exc:
            logger.error("Failed to get metrics", exc_info=exc)
            return IngestionMetrics()
    
    def _calculate_batch_metrics(self):
        """Calculate final metrics for the completed batch."""
        try:
            # Calculate dedup ratio if we have duplicates
            if self.processed_emails > 0:
                self.dedup_ratio = self.duplicate_emails / (self.processed_emails + self.duplicate_emails)
            
            # Calculate OCR rate (simplified)
            # In a real implementation, you'd track actual OCR tasks
            if self.total_attachments > 0:
                # Placeholder calculation
                self.ocr_rate = 0.1  # Assume 10% of attachments need OCR
            
            # Calculate success rate
            total_attempted = self.processed_emails + self.failed_emails + self.quarantined_emails
            if total_attempted > 0:
                self.success_rate = self.processed_emails / total_attempted
            
        except Exception as exc:
            logger.error("Failed to calculate batch metrics", exc_info=exc)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get a performance summary of the current batch."""
        try:
            if not self.batch_start_time:
                return {}
            
            current_time = datetime.utcnow()
            elapsed_time = (current_time - self.batch_start_time).total_seconds()
            
            # Calculate current throughput
            current_docs_per_second = self.processed_emails / elapsed_time if elapsed_time > 0 else 0
            
            # Calculate estimated completion time
            remaining_emails = self.total_emails - self.processed_emails
            estimated_completion_time = remaining_emails / current_docs_per_second if current_docs_per_second > 0 else 0
            
            return {
                'batch_id': self.current_batch_id,
                'elapsed_time_seconds': elapsed_time,
                'processed_emails': self.processed_emails,
                'total_emails': self.total_emails,
                'progress_percentage': (self.processed_emails / self.total_emails * 100) if self.total_emails > 0 else 0,
                'current_throughput': current_docs_per_second,
                'estimated_completion_time': estimated_completion_time,
                'success_rate': self.success_rate,
                'error_rate': 1 - self.success_rate if self.success_rate is not None else 0
            }
            
        except Exception as exc:
            logger.error("Failed to get performance summary", exc_info=exc)
            return {}
    
    def reset_metrics(self):
        """Reset all metrics."""
        try:
            self.current_batch_id = None
            self.batch_start_time = None
            self.batch_end_time = None
            
            # Reset counters
            self.total_emails = 0
            self.total_attachments = 0
            self.total_chunks = 0
            self.processed_emails = 0
            self.failed_emails = 0
            self.quarantined_emails = 0
            self.duplicate_emails = 0
            
            # Reset timing arrays
            self.email_processing_times = []
            self.attachment_processing_times = []
            self.chunking_times = []
            
            # Reset error tracking
            self.error_buckets = {}
            self.retry_counts = {}
            
            # Reset quality metrics
            self.dedup_ratio = 0.0
            self.ocr_rate = 0.0
            self.success_rate = 0.0
            
            # Reset throughput metrics
            self.docs_per_second = 0.0
            self.average_latency = 0.0
            
            # Update timestamp
            self.last_updated = datetime.utcnow()
            
            logger.info("Metrics reset completed")
            
        except Exception as exc:
            logger.error("Failed to reset metrics", exc_info=exc)
    
    def export_metrics(self, format: str = "json") -> str:
        """Export metrics in the specified format."""
        try:
            metrics = self.get_metrics()
            
            if format.lower() == "json":
                import json
                return json.dumps(metrics.dict(), indent=2, default=str)
            elif format.lower() == "csv":
                # Simple CSV export
                csv_lines = ["metric,value"]
                metrics_dict = metrics.dict()
                for key, value in metrics_dict.items():
                    csv_lines.append(f"{key},{value}")
                return "\n".join(csv_lines)
            else:
                raise ValueError(f"Unsupported export format: {format}")
                
        except Exception as exc:
            logger.error("Failed to export metrics", exc_info=exc)
            return ""
