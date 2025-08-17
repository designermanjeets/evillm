"""Application settings configuration."""

import os
from typing import List, Optional
from functools import lru_cache

from pydantic import Field, validator
from pydantic_settings import BaseSettings


class DatabaseSettings(BaseSettings):
    """Database configuration settings."""
    
    host: str = Field(default="localhost", env="DB_HOST")
    port: int = Field(default=5432, env="DB_PORT")
    name: str = Field(default="evillm", env="DB_NAME")
    user: str = Field(default="postgres", env="DB_USER")
    password: str = Field(default="", env="DB_PASSWORD")
    pool_size: int = Field(default=10, env="DB_POOL_SIZE")
    max_overflow: int = Field(default=20, env="DB_MAX_OVERFLOW")
    
    @property
    def url(self) -> str:
        """Get database connection URL."""
        return f"postgresql+asyncpg://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"


class StorageSettings(BaseSettings):
    """Object storage configuration settings."""
    
    provider: str = Field(default="minio", env="STORAGE_PROVIDER")  # minio, s3, local
    endpoint: str = Field(default="localhost:9000", env="STORAGE_ENDPOINT")
    access_key: str = Field(default="", env="STORAGE_ACCESS_KEY")
    secret_key: str = Field(default="", env="STORAGE_SECRET_KEY")
    bucket_name: str = Field(default="evillm", env="STORAGE_BUCKET")
    region: str = Field(default="us-east-1", env="STORAGE_REGION")
    secure: bool = Field(default=False, env="STORAGE_SECURE")
    
    @validator("provider")
    def validate_provider(cls, v: str) -> str:
        """Validate storage provider."""
        if v not in ["minio", "s3", "local"]:
            raise ValueError("Storage provider must be minio, s3, or local")
        return v


class SearchSettings(BaseSettings):
    """Search engine configuration settings."""
    
    # OpenSearch/Elasticsearch
    opensearch_host: str = Field(default="localhost", env="OPENSEARCH_HOST")
    opensearch_port: int = Field(default=9200, env="OPENSEARCH_PORT")
    opensearch_username: str = Field(default="admin", env="OPENSEARCH_USERNAME")
    opensearch_password: str = Field(default="", env="OPENSEARCH_PASSWORD")
    opensearch_use_ssl: bool = Field(default=False, env="OPENSEARCH_USE_SSL")
    
    # Vector store
    vector_store: str = Field(default="qdrant", env="VECTOR_STORE")  # qdrant, pinecone, weaviate
    qdrant_host: str = Field(default="localhost", env="QDRANT_HOST")
    qdrant_port: int = Field(default=6333, env="QDRANT_PORT")
    qdrant_api_key: Optional[str] = Field(default=None, env="QDRANT_API_KEY")
    
    pinecone_api_key: Optional[str] = Field(default=None, env="PINECONE_API_KEY")
    pinecone_environment: str = Field(default="us-east-1-aws", env="PINECONE_ENVIRONMENT")
    pinecone_index: str = Field(default="evillm", env="PINECONE_INDEX")
    
    weaviate_url: str = Field(default="http://localhost:8080", env="WEAVIATE_URL")
    weaviate_api_key: Optional[str] = Field(default=None, env="WEAVIATE_API_KEY")
    
    # Quality assurance thresholds
    quality_thresholds_hit_at_5: float = Field(default=0.65, env="SEARCH_QUALITY_THRESHOLD_HIT_AT_5")
    quality_thresholds_mrr_at_10: float = Field(default=0.55, env="SEARCH_QUALITY_THRESHOLD_MRR_AT_10")
    quality_thresholds_ndcg_at_10: float = Field(default=0.60, env="SEARCH_QUALITY_THRESHOLD_NDCG_AT_10")
    
    @validator("vector_store")
    def validate_vector_store(cls, v: str) -> str:
        """Validate vector store provider."""
        if v not in ["qdrant", "pinecone", "weaviate"]:
            raise ValueError("Vector store must be qdrant, pinecone, or weaviate")
        return v
    
    @validator("quality_thresholds_hit_at_5", "quality_thresholds_mrr_at_10", "quality_thresholds_ndcg_at_10")
    def validate_quality_thresholds(cls, v: float) -> float:
        """Validate quality thresholds are between 0 and 1."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Quality thresholds must be between 0.0 and 1.0")
        return v


class LLMSettings(BaseSettings):
    """Large Language Model configuration settings."""
    
    provider: str = Field(default="openai", env="LLM_PROVIDER")  # openai, anthropic, local
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4", env="OPENAI_MODEL")
    openai_base_url: Optional[str] = Field(default=None, env="OPENAI_BASE_URL")
    
    anthropic_api_key: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    anthropic_model: str = Field(default="claude-3-sonnet-20240229", env="ANTHROPIC_MODEL")
    
    max_tokens: int = Field(default=4000, env="LLM_MAX_TOKENS")
    temperature: float = Field(default=0.1, env="LLM_TEMPERATURE")
    
    @validator("provider")
    def validate_provider(cls, v: str) -> str:
        """Validate LLM provider."""
        if v not in ["openai", "anthropic", "local"]:
            raise ValueError("LLM provider must be openai, anthropic, or local")
        return v


class SecuritySettings(BaseSettings):
    """Security configuration settings."""
    
    secret_key: str = Field(default="", env="SECRET_KEY")
    algorithm: str = Field(default="HS256", env="ALGORITHM")
    access_token_expire_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    
    # PII detection
    pii_detection_enabled: bool = Field(default=True, env="PII_DETECTION_ENABLED")
    pii_redaction_enabled: bool = Field(default=True, env="PII_REDACTION_ENABLED")
    
    # Rate limiting
    rate_limit_requests: int = Field(default=100, env="RATE_LIMIT_REQUESTS")
    rate_limit_window: int = Field(default=3600, env="RATE_LIMIT_WINDOW")  # seconds
    
    @validator("secret_key")
    def validate_secret_key(cls, v: str) -> str:
        """Validate secret key is set."""
        if not v and os.getenv("ENVIRONMENT") == "production":
            raise ValueError("SECRET_KEY must be set in production")
        return v or "dev-secret-key-change-in-production"


class MonitoringSettings(BaseSettings):
    """Monitoring and observability settings."""
    
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(default="json", env="LOG_FORMAT")
    
    # Metrics
    metrics_enabled: bool = Field(default=True, env="METRICS_ENABLED")
    metrics_port: int = Field(default=9090, env="METRICS_PORT")
    
    # Tracing
    tracing_enabled: bool = Field(default=False, env="TRACING_ENABLED")
    jaeger_host: str = Field(default="localhost", env="JAEGER_HOST")
    jaeger_port: int = Field(default=6831, env="JAEGER_PORT")
    
    # LangSmith
    langsmith_enabled: bool = Field(default=False, env="LANGSMITH_ENABLED")
    langsmith_api_key: Optional[str] = Field(default=None, env="LANGSMITH_API_KEY")
    langsmith_project: str = Field(default="evillm", env="LANGSMITH_PROJECT")
    langsmith_endpoint: str = Field(default="https://api.smith.langchain.com", env="LANGSMITH_ENDPOINT")


class Settings(BaseSettings):
    """Main application settings."""
    
    # Application
    environment: str = Field(default="development", env="ENVIRONMENT")
    debug: bool = Field(default=False, env="DEBUG")
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    
    # CORS
    cors_origins: List[str] = Field(default=["*"], env="CORS_ORIGINS")
    
    # Database
    database: DatabaseSettings = DatabaseSettings()
    
    # Storage
    storage: StorageSettings = StorageSettings()
    
    # Search
    search: SearchSettings = SearchSettings()
    
    # LLM
    llm: LLMSettings = LLMSettings()
    
    # Security
    security: SecuritySettings = SecuritySettings()
    
    # Monitoring
    monitoring: MonitoringSettings = MonitoringSettings()
    
    # Performance
    max_concurrent_requests: int = Field(default=50, env="MAX_CONCURRENT_REQUESTS")
    request_timeout: int = Field(default=30, env="REQUEST_TIMEOUT")  # seconds
    
    # Evaluation
    eval_threshold: float = Field(default=0.8, env="EVAL_THRESHOLD")
    eval_enabled: bool = Field(default=True, env="EVAL_ENABLED")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached application settings."""
    return Settings()


# Environment-specific settings
def get_database_url() -> str:
    """Get database URL from environment."""
    return get_settings().database.url


def get_storage_config() -> StorageSettings:
    """Get storage configuration from environment."""
    return get_settings().storage


def get_search_config() -> SearchSettings:
    """Get search configuration from environment."""
    return get_settings().search


def get_llm_config() -> LLMSettings:
    """Get LLM configuration from environment."""
    return get_settings().llm


def get_security_config() -> SecuritySettings:
    """Get security configuration from environment."""
    return get_settings().security


def get_monitoring_config() -> MonitoringSettings:
    """Get monitoring configuration from environment."""
    return get_settings().monitoring
