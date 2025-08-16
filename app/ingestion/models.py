"""Data models for the ingestion pipeline."""

from datetime import datetime
from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import uuid
import hashlib

from pydantic import BaseModel, Field


class ProcessingStatus(str, Enum):
    """Processing status for ingestion pipeline."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    QUARANTINED = "quarantined"
    SKIPPED = "skipped"


class ErrorType(str, Enum):
    """Error types for ingestion failures."""
    MALFORMED_MIME = "malformed_mime"
    OVERSIZED_EMAIL = "oversized_email"
    OVERSIZED_ATTACHMENT = "oversized_attachment"
    INVALID_MIMETYPE = "invalid_mimetype"
    PARSE_ERROR = "parse_error"
    STORAGE_ERROR = "storage_error"
    DATABASE_ERROR = "database_error"
    UNKNOWN = "unknown"


@dataclass
class EmailMetadata:
    """Email metadata extracted during parsing."""
    
    # Basic email information
    message_id: str
    subject: str
    from_address: str
    to_addresses: List[str]
    cc_addresses: List[str]
    bcc_addresses: List[str]
    date: datetime
    content_type: str
    tenant_id: str
    batch_id: str
    
    # Optional fields
    thread_id: Optional[str] = None
    in_reply_to: Optional[str] = None
    references: Optional[str] = None
    charset: Optional[str] = None
    language: Optional[str] = None
    processing_status: ProcessingStatus = ProcessingStatus.PENDING
    error_type: Optional[ErrorType] = None
    error_message: Optional[str] = None
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class EmailContent:
    """Email content after parsing and normalization."""
    
    # Raw content
    raw_content: bytes
    raw_content_hash: str
    
    # Normalized content
    normalized_text: str
    normalized_text_hash: str
    
    # Content metadata
    content_length: int
    word_count: int
    sentence_count: int
    paragraph_count: int
    
    # Language detection
    detected_language: str
    language_confidence: float
    
    # Normalization manifest
    normalization_manifest: Dict[str, Any]
    
    # Timestamps
    processed_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class AttachmentInfo:
    """Attachment information extracted from email."""
    
    # Basic attachment info
    filename: str
    content_type: str
    content_disposition: str
    content: bytes
    content_hash: str
    content_length: int
    
    # Optional fields
    content_id: Optional[str] = None
    text_extracted: bool = False
    extracted_text: Optional[str] = None
    extracted_text_hash: Optional[str] = None
    ocr_required: bool = False
    ocr_task_id: Optional[str] = None
    ocr_status: ProcessingStatus = ProcessingStatus.PENDING
    storage_key: Optional[str] = None
    storage_metadata: Optional[Dict[str, Any]] = None
    
    # Timestamps
    extracted_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class DedupResult:
    """Deduplication result for email content."""
    
    # Content identification
    content_hash: str
    is_duplicate: bool
    duplicate_type: str  # "exact", "near", "none"
    
    # Duplicate information
    original_content_hash: Optional[str] = None
    similarity_score: Optional[float] = None
    dedup_algorithm: Optional[str] = None
    
    # Lineage tracking
    reference_count: int = 1
    first_seen_at: datetime = field(default_factory=datetime.utcnow)
    last_seen_at: datetime = field(default_factory=datetime.utcnow)
    
    # Storage optimization
    storage_key_reused: Optional[str] = None
    storage_saved: int = 0  # bytes saved


@dataclass
class ChunkInfo:
    """Information about a text chunk."""
    
    # Chunk identification
    chunk_uid: str
    chunk_index: int
    chunk_hash: str
    
    # Content information
    content: str
    content_length: int
    token_count: int
    
    # Position information
    start_position: int
    end_position: int
    
    # Relationships
    email_id: str
    attachment_id: Optional[str] = None
    
    # Metadata
    chunk_type: str = "text"  # "text", "attachment", "mixed"
    semantic_boundary: bool = True
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ThreadInfo:
    """Thread information for email conversations."""
    
    # Thread identification
    thread_id: str
    thread_subject: str
    thread_subject_normalized: str
    tenant_id: str
    
    # Optional fields
    email_count: int = 0
    first_email_date: Optional[datetime] = None
    last_email_date: Optional[datetime] = None
    root_message_id: Optional[str] = None
    thread_depth: int = 0
    thread_breadth: int = 0
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class OCRTask:
    """OCR task for image/PDF processing."""
    
    # Task identification
    task_id: str
    email_id: str
    attachment_id: str
    content_type: str
    content_hash: str
    content_length: int
    
    # Optional fields
    status: ProcessingStatus = ProcessingStatus.PENDING
    priority: int = 0
    extracted_text: Optional[str] = None
    extracted_text_hash: Optional[str] = None
    confidence_score: Optional[float] = None
    error_type: Optional[ErrorType] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


@dataclass
class EmbeddingJob:
    """Job for generating embeddings from chunks."""
    
    # Job identification
    job_id: str
    chunk_uid: str
    tenant_id: str
    batch_id: str
    
    # Optional fields
    status: ProcessingStatus = ProcessingStatus.PENDING
    priority: int = 0
    model_name: str = "text-embedding-ada-002"  # Default OpenAI model
    token_count: int = 0
    embedding_vector: Optional[List[float]] = None
    embedding_hash: Optional[str] = None
    error_type: Optional[ErrorType] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


@dataclass
class CheckpointInfo:
    """Checkpoint information for batch processing."""
    
    # Checkpoint identification
    checkpoint_id: str
    batch_id: str
    tenant_id: str
    
    # Progress tracking
    total_emails: int
    processed_emails: int
    failed_emails: int
    quarantined_emails: int
    
    # Processing metadata
    last_processed_email: Optional[str] = None
    last_processed_position: int = 0
    
    # Status information
    is_complete: bool = False
    error_count: int = 0
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None


@dataclass
class BatchManifest:
    """Manifest for batch processing."""
    
    # Batch identification
    batch_id: str
    tenant_id: str
    source_type: str
    source_path: str
    total_files: int
    
    # Optional fields
    priority: int = 0
    description: Optional[str] = None
    batch_size: int = 100
    max_workers: int = 4
    status: ProcessingStatus = ProcessingStatus.PENDING
    progress: float = 0.0
    file_manifest: List[Dict[str, Any]] = field(default_factory=list)
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class IngestionMetrics(BaseModel):
    """Metrics for ingestion pipeline performance."""
    
    # Performance metrics
    docs_per_second: float = 0.0
    total_processing_time: float = 0.0
    average_latency: float = 0.0
    
    # Volume metrics
    total_emails: int = 0
    total_attachments: int = 0
    total_chunks: int = 0
    
    # Quality metrics
    dedup_ratio: float = 0.0
    ocr_rate: float = 0.0
    success_rate: float = 0.0
    
    # Error metrics
    error_count: int = 0
    quarantine_count: int = 0
    retry_count: int = 0
    
    # Error buckets
    error_buckets: Dict[str, int] = Field(default_factory=dict)
    
    # Timestamps
    batch_start_time: Optional[datetime] = None
    batch_end_time: Optional[datetime] = None
    last_updated: datetime = Field(default_factory=datetime.utcnow)


def generate_chunk_uid(email_id: str, chunk_index: int, content_hash: str) -> str:
    """Generate stable chunk UID."""
    return f"{email_id}_{chunk_index}_{content_hash[:8]}"


def generate_batch_id() -> str:
    """Generate unique batch ID."""
    return f"batch_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"


def generate_task_id() -> str:
    """Generate unique task ID."""
    return f"task_{uuid.uuid4().hex}"


def generate_job_id() -> str:
    """Generate unique job ID."""
    return f"job_{uuid.uuid4().hex}"


def generate_checkpoint_id(batch_id: str) -> str:
    """Generate checkpoint ID for batch."""
    return f"checkpoint_{batch_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
