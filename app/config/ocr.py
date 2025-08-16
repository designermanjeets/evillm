"""OCR service configuration."""

import os
from typing import List, Dict, Any, Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings
import yaml
from pathlib import Path


class OCRSettings(BaseSettings):
    """OCR service configuration settings."""
    
    # Backend Configuration
    backend: str = Field(default="local_stub", description="OCR backend to use")
    enabled: bool = Field(default=True, description="Enable OCR processing")
    
    # Processing Limits
    timeout_seconds: int = Field(default=20, description="Default OCR timeout in seconds")
    max_retries: int = Field(default=2, description="Maximum OCR retry attempts")
    concurrency: int = Field(default=4, description="Maximum concurrent OCR tasks")
    max_pages: int = Field(default=20, description="Maximum pages per document")
    size_cap_mb: int = Field(default=15, description="Maximum file size in MB")
    
    # File Type Configuration
    allow_mimetypes: List[str] = Field(
        default=[
            "application/pdf", "image/png", "image/jpeg", "image/gif", "image/bmp", "image/tiff",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/msword", "text/plain"
        ],
        description="Allowed mimetypes for OCR processing"
    )
    
    # Image Preprocessing
    preprocess: Dict[str, Any] = Field(
        default={
            "grayscale": True,
            "binarize": True,
            "deskew": True,
            "timeout_seconds": 5
        },
        description="Image preprocessing configuration"
    )
    
    # Feature Flags
    feature_flags: Dict[str, bool] = Field(
        default={
            "ocr_idempotent_skip": True,
            "ocr_queue_inprocess": True,
            "ocr_redact_logs": True,
            "ocr_preprocessing": True,
            "ocr_tesseract_enabled": False
        },
        description="OCR feature flags"
    )
    
    @field_validator('backend')
    @classmethod
    def validate_backend(cls, v: str) -> str:
        """Validate OCR backend selection."""
        allowed_backends = ["local_stub", "tesseract", "aws_textract", "gcv", "azure_cv"]
        if v not in allowed_backends:
            raise ValueError(f"Invalid OCR backend: {v}. Allowed: {allowed_backends}")
        return v
    
    @field_validator('concurrency')
    @classmethod
    def validate_concurrency(cls, v: int) -> int:
        """Validate concurrency limits."""
        if v < 1 or v > 20:
            raise ValueError("OCR concurrency must be between 1 and 20")
        return v
    
    @field_validator('timeout_seconds')
    @classmethod
    def validate_timeout(cls, v: int) -> int:
        """Validate timeout values."""
        if v < 1 or v > 300:
            raise ValueError("OCR timeout must be between 1 and 300 seconds")
        return v
    
    class Config:
        env_prefix = "APP_OCR_"
        case_sensitive = False


def load_yaml_config() -> Dict[str, Any]:
    """Load configuration from YAML file."""
    config_path = Path(__file__).parent / "app.yaml"
    
    if not config_path.exists():
        return {}
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            return config.get('ocr', {})
    except Exception as e:
        print(f"Warning: Failed to load YAML config: {e}")
        return {}

def get_ocr_settings() -> OCRSettings:
    """Get OCR configuration settings with YAML fallback."""
    # Load YAML config first
    yaml_config = load_yaml_config()
    
    # Create settings with YAML defaults, environment overrides
    settings = OCRSettings(**yaml_config)
    
    return settings
