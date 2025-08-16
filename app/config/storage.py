"""Storage configuration settings for Logistics Email AI."""

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings
import os


class StorageSettings(BaseSettings):
    """Storage configuration settings."""
    
    # Storage provider configuration
    provider: str = Field(default="minio", env="STORAGE_PROVIDER")
    endpoint_url: str = Field(default="http://localhost:9000", env="STORAGE_ENDPOINT_URL")
    region: str = Field(default="us-east-1", env="STORAGE_REGION")
    bucket_name: str = Field(default="evillm", env="STORAGE_BUCKET_NAME")
    
    # Authentication
    access_key_id: str = Field(default="minioadmin", env="STORAGE_ACCESS_KEY_ID")
    secret_access_key: str = Field(default="minioadmin", env="STORAGE_SECRET_ACCESS_KEY")
    
    # Security settings
    encryption_algorithm: str = Field(default="AES256", env="STORAGE_ENCRYPTION_ALGORITHM")
    force_path_style: bool = Field(default=True, env="STORAGE_FORCE_PATH_STYLE")
    
    # Performance settings
    multipart_threshold: int = Field(default=5 * 1024 * 1024, env="STORAGE_MULTIPART_THRESHOLD")  # 5MB
    multipart_chunk_size: int = Field(default=10 * 1024 * 1024, env="STORAGE_MULTIPART_CHUNK_SIZE")  # 10MB
    max_concurrent_uploads: int = Field(default=10, env="STORAGE_MAX_CONCURRENT_UPLOADS")
    
    # Retry configuration
    max_retry_attempts: int = Field(default=3, env="STORAGE_MAX_RETRY_ATTEMPTS")
    retry_base_delay: float = Field(default=1.0, env="STORAGE_RETRY_BASE_DELAY")
    retry_max_delay: float = Field(default=60.0, env="STORAGE_RETRY_MAX_DELAY")
    
    # URL generation
    presigned_url_ttl: int = Field(default=900, env="STORAGE_PRESIGNED_URL_TTL")  # 15 minutes
    presigned_url_max_ttl: int = Field(default=3600, env="STORAGE_PRESIGNED_URL_MAX_TTL")  # 1 hour
    
    # Content restrictions
    allowed_mimetypes: List[str] = Field(
        default=[
            "text/plain",
            "text/html",
            "message/rfc822",
            "application/pdf",
            "image/jpeg",
            "image/png",
            "image/gif",
            "application/msword",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/vnd.ms-excel",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        ],
        env="STORAGE_ALLOWED_MIMETYPES"
    )
    max_object_size: int = Field(default=100 * 1024 * 1024, env="STORAGE_MAX_OBJECT_SIZE")  # 100MB
    
    # Health check settings
    health_check_enabled: bool = Field(default=True, env="STORAGE_HEALTH_CHECK_ENABLED")
    canary_test_enabled: bool = Field(default=False, env="STORAGE_CANARY_TEST_ENABLED")
    canary_test_size: int = Field(default=1024, env="STORAGE_CANARY_TEST_SIZE")  # 1KB
    
    @field_validator("provider")
    @classmethod
    def validate_provider(cls, v: str) -> str:
        """Validate storage provider."""
        allowed_providers = ["minio", "s3", "local"]
        if v not in allowed_providers:
            raise ValueError(f"Storage provider must be one of: {allowed_providers}")
        return v
    
    @field_validator("presigned_url_ttl")
    @classmethod
    def validate_presigned_url_ttl(cls, v: int) -> int:
        """Validate presigned URL TTL."""
        if v <= 0 or v > 86400:  # Max 24 hours
            raise ValueError("Presigned URL TTL must be between 1 and 86400 seconds")
        return v
    
    @field_validator("multipart_threshold")
    @classmethod
    def validate_multipart_threshold(cls, v: int) -> int:
        """Validate multipart threshold."""
        if v < 1024 * 1024:  # Min 1MB
            raise ValueError("Multipart threshold must be at least 1MB")
        return v
    
    @field_validator("endpoint_url")
    @classmethod
    def validate_endpoint_url(cls, v: str) -> str:
        """Validate endpoint URL."""
        if not v.startswith(("http://", "https://")):
            raise ValueError("Endpoint URL must start with http:// or https://")
        return v
    
    model_config = {
        "env_prefix": "STORAGE_",
        "case_sensitive": False
    }


def get_storage_settings() -> StorageSettings:
    """Get storage settings instance."""
    return StorageSettings()
