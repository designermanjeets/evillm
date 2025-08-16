"""Ingestion configuration settings for Logistics Email AI."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings
import os


class IngestionSettings(BaseSettings):
    """Ingestion configuration settings."""
    
    # Source configuration
    dropbox_path: str = Field(default="./data/ingestion/dropbox", env="INGESTION_DROPBOX_PATH")
    manifest_path: str = Field(default="./data/ingestion/manifests", env="INGESTION_MANIFEST_PATH")
    batch_size: int = Field(default=100, env="INGESTION_BATCH_SIZE")
    max_workers: int = Field(default=4, env="INGESTION_MAX_WORKERS")
    
    # Processing configuration
    max_email_size: int = Field(default=50 * 1024 * 1024, env="INGESTION_MAX_EMAIL_SIZE")  # 50MB
    max_attachment_size: int = Field(default=25 * 1024 * 1024, env="INGESTION_MAX_ATTACHMENT_SIZE")  # 25MB
    chunk_size: int = Field(default=1000, env="INGESTION_CHUNK_SIZE")  # characters
    chunk_overlap: int = Field(default=200, env="INGESTION_CHUNK_OVERLAP")  # characters
    
    # Deduplication configuration
    simhash_threshold: float = Field(default=0.85, env="INGESTION_SIMHASH_THRESHOLD")
    minhash_threshold: float = Field(default=0.80, env="INGESTION_MINHASH_THRESHOLD")
    enable_near_dup_detection: bool = Field(default=True, env="INGESTION_ENABLE_NEAR_DUP")
    
    # OCR configuration
    ocr_enabled: bool = Field(default=True, env="INGESTION_OCR_ENABLED")
    ocr_batch_size: int = Field(default=50, env="INGESTION_OCR_BATCH_SIZE")
    ocr_timeout: int = Field(default=300, env="INGESTION_OCR_TIMEOUT")  # seconds
    
    # Allowed mimetypes for attachments
    allowed_attachment_mimetypes: List[str] = Field(
        default=[
            "text/plain",
            "text/html",
            "application/pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            "application/msword",
            "application/vnd.ms-excel",
            "application/vnd.ms-powerpoint",
            "image/jpeg",
            "image/png",
            "image/gif",
            "image/bmp",
            "image/tiff"
        ],
        env="INGESTION_ALLOWED_ATTACHMENT_MIMETYPES"
    )
    
    # Checkpoint configuration
    checkpoint_enabled: bool = Field(default=True, env="INGESTION_CHECKPOINT_ENABLED")
    checkpoint_interval: int = Field(default=10, env="INGESTION_CHECKPOINT_INTERVAL")  # emails
    checkpoint_retention: int = Field(default=7, env="INGESTION_CHECKPOINT_RETENTION")  # days
    
    # Quarantine configuration
    quarantine_enabled: bool = Field(default=True, env="INGESTION_QUARANTINE_ENABLED")
    quarantine_path: str = Field(default="./data/ingestion/quarantine", env="INGESTION_QUARANTINE_PATH")
    
    # Metrics configuration
    metrics_enabled: bool = Field(default=True, env="INGESTION_METRICS_ENABLED")
    metrics_interval: int = Field(default=60, env="INGESTION_METRICS_INTERVAL")  # seconds
    
    # Retry configuration
    max_retry_attempts: int = Field(default=3, env="INGESTION_MAX_RETRY_ATTEMPTS")
    retry_base_delay: float = Field(default=1.0, env="INGESTION_RETRY_BASE_DELAY")
    retry_max_delay: float = Field(default=60.0, env="INGESTION_RETRY_MAX_DELAY")
    
    # Language detection
    language_detection_enabled: bool = Field(default=True, env="INGESTION_LANGUAGE_DETECTION_ENABLED")
    default_language: str = Field(default="en", env="INGESTION_DEFAULT_LANGUAGE")
    
    # Threading configuration
    threading_enabled: bool = Field(default=True, env="INGESTION_THREADING_ENABLED")
    subject_similarity_threshold: float = Field(default=0.7, env="INGESTION_SUBJECT_SIMILARITY_THRESHOLD")
    
    @field_validator("batch_size")
    @classmethod
    def validate_batch_size(cls, v: int) -> int:
        """Validate batch size."""
        if v < 1 or v > 1000:
            raise ValueError("Batch size must be between 1 and 1000")
        return v
    
    @field_validator("simhash_threshold", "minhash_threshold")
    @classmethod
    def validate_similarity_threshold(cls, v: float) -> float:
        """Validate similarity threshold."""
        if v < 0.0 or v > 1.0:
            raise ValueError("Similarity threshold must be between 0.0 and 1.0")
        return v
    
    @field_validator("chunk_size")
    @classmethod
    def validate_chunk_size(cls, v: int) -> int:
        """Validate chunk size."""
        if v < 100 or v > 10000:
            raise ValueError("Chunk size must be between 100 and 10000 characters")
        return v
    
    @field_validator("chunk_overlap")
    @classmethod
    def validate_chunk_overlap(cls, v: int) -> int:
        """Validate chunk overlap."""
        if v < 0 or v > 1000:
            raise ValueError("Chunk overlap must be between 0 and 1000 characters")
        return v
    
    model_config = {
        "env_prefix": "INGESTION_",
        "case_sensitive": False
    }


def get_ingestion_settings() -> IngestionSettings:
    """Get ingestion settings instance."""
    return IngestionSettings()
