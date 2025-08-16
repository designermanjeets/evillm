"""OCR service configuration."""

from typing import List, Dict, Any
from pydantic import Field
from pydantic_settings import BaseSettings


class OCRSettings(BaseSettings):
    """OCR service configuration settings."""
    
    # Backend Configuration
    default_backend: str = Field(default="local", description="Default OCR backend")
    enable_tesseract: bool = Field(default=False, description="Enable Tesseract backend")
    enable_cloud_backends: bool = Field(default=False, description="Enable cloud OCR backends")
    
    # Processing Limits
    max_file_size_mb: int = Field(default=50, description="Maximum file size in MB")
    max_pages: int = Field(default=100, description="Maximum pages per document")
    max_concurrent_tasks: int = Field(default=5, description="Maximum concurrent OCR tasks")
    max_queue_size: int = Field(default=100, description="Maximum OCR queue size")
    
    # Timeout & Retry Configuration
    default_timeout_seconds: int = Field(default=30, description="Default OCR timeout in seconds")
    max_retries: int = Field(default=3, description="Maximum OCR retry attempts")
    retry_delay_base: float = Field(default=1.0, description="Base delay for retry backoff")
    retry_delay_max: float = Field(default=60.0, description="Maximum retry delay")
    
    # File Type Configuration
    allowed_mimetypes: List[str] = Field(
        default=[
            # Images
            "image/jpeg", "image/png", "image/gif", "image/bmp", "image/tiff",
            # Documents
            "application/pdf", "application/msword", 
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "text/plain"
        ],
        description="Allowed mimetypes for OCR processing"
    )
    
    # Size limits by file type
    size_limits_mb: Dict[str, int] = Field(
        default={
            "image/jpeg": 10,
            "image/png": 10,
            "image/gif": 5,
            "image/bmp": 10,
            "image/tiff": 25,
            "application/pdf": 50,
            "application/msword": 25,
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document": 25,
            "text/plain": 5
        },
        description="Size limits in MB by mimetype"
    )
    
    # Image Preprocessing
    enable_preprocessing: bool = Field(default=True, description="Enable image preprocessing")
    preprocessing_timeout: int = Field(default=10, description="Preprocessing timeout in seconds")
    enable_grayscale: bool = Field(default=True, description="Enable grayscale conversion")
    enable_binarization: bool = Field(default=True, description="Enable binarization")
    enable_deskewing: bool = Field(default=True, description="Enable image deskewing")
    
    # Metrics Configuration
    enable_detailed_metrics: bool = Field(default=True, description="Enable detailed OCR metrics")
    metrics_retention_days: int = Field(default=30, description="Metrics retention period")
    
    # Security Configuration
    quarantine_failed: bool = Field(default=True, description="Quarantine failed OCR attempts")
    log_pii: bool = Field(default=False, description="Log PII in OCR processing")
    
    class Config:
        env_prefix = "OCR_"
        case_sensitive = False


def get_ocr_settings() -> OCRSettings:
    """Get OCR configuration settings."""
    return OCRSettings()
