"""Embedding service with queue management and provider-agnostic interface."""

import asyncio
import time
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, AsyncGenerator
from abc import ABC, abstractmethod
import structlog
import numpy as np

from ..database.models import EmbeddingJob
from ..database.session import get_database_session
from ..config.manager import get_config

logger = structlog.get_logger(__name__)


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""
    
    @abstractmethod
    async def embed_batch(self, texts: List[str], **kwargs) -> List[List[float]]:
        """Embed a batch of texts, return vectors."""
        pass
    
    @abstractmethod
    async def get_embedding_dimensions(self) -> int:
        """Return embedding dimensions for this provider."""
        pass
    
    @abstractmethod
    async def estimate_cost(self, text_count: int) -> float:
        """Estimate cost for embedding operation."""
        pass


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI embedding provider implementation."""
    
    def __init__(self, api_key: str, model: str = "text-embedding-3-large"):
        try:
            from openai import AsyncOpenAI
            if api_key and api_key.strip():
                self.client = AsyncOpenAI(api_key=api_key)
                self.model = model
                self.dimensions = 1536  # text-embedding-3-large dimensions
                self.cost_per_1k_tokens = 0.00013  # OpenAI pricing
            else:
                # Stub mode when no API key provided
                self.client = None
                self.model = "stub"
                self.dimensions = 1536
                self.cost_per_1k_tokens = 0.0
        except ImportError:
            logger.error("OpenAI client not available, using stub provider")
            self.client = None
            self.model = "stub"
            self.dimensions = 1536
            self.cost_per_1k_tokens = 0.0
    
    async def embed_batch(self, texts: List[str], **kwargs) -> List[List[float]]:
        """Embed texts using OpenAI API with batching."""
        if not self.client:
            # Stub implementation for testing
            return [[0.1] * self.dimensions for _ in texts]
        
        try:
            start_time = time.time()
            response = await self.client.embeddings.create(
                model=self.model,
                input=texts
            )
            latency = time.time() - start_time
            
            logger.info("OpenAI embedding completed", 
                       batch_size=len(texts), 
                       latency=latency,
                       model=self.model)
            
            return [data.embedding for data in response.data]
        except Exception as e:
            logger.error("OpenAI embedding failed", 
                        error=str(e), 
                        batch_size=len(texts),
                        model=self.model)
            raise
    
    async def get_embedding_dimensions(self) -> int:
        """Return embedding dimensions for this provider."""
        return self.dimensions
    
    async def estimate_cost(self, text_count: int) -> float:
        """Estimate cost for embedding operation."""
        if self.cost_per_1k_tokens == 0.0:
            return 0.0
        
        # Rough estimate: 1 token ≈ 4 characters
        estimated_tokens = text_count // 4
        return (estimated_tokens / 1000) * self.cost_per_1k_tokens


class EmbeddingBatcher:
    """Intelligent batching for embedding jobs."""
    
    def __init__(self, batch_size: int = 64, max_wait_time: float = 5.0):
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        self.current_batch = []
        self.last_batch_time = time.time()
    
    async def add_job(self, job: EmbeddingJob) -> bool:
        """Add job to current batch, return True if batch is ready."""
        self.current_batch.append(job)
        
        # Check if batch is ready
        if len(self.current_batch) >= self.batch_size:
            return True
        
        # Check if max wait time exceeded
        if time.time() - self.last_batch_time >= self.max_wait_time:
            return True
        
        return False
    
    def get_batch(self) -> List[EmbeddingJob]:
        """Get current batch and reset."""
        batch = self.current_batch.copy()
        self.current_batch = []
        self.last_batch_time = time.time()
        return batch


class EmbeddingCostGuard:
    """Cost control and rate limiting for embedding operations."""
    
    def __init__(self, max_cost_per_day: float = 100.0):
        self.max_cost_per_day = max_cost_per_day
        self.daily_cost = 0.0
        self.last_reset = datetime.now().date()
        self.rate_limit_tokens = 100000  # tokens per hour
        self.last_hour_tokens = 0
        self.last_hour_reset = datetime.now()
    
    def _reset_if_new_day(self):
        """Reset daily cost counter if it's a new day."""
        current_date = datetime.now().date()
        if current_date != self.last_reset:
            self.daily_cost = 0.0
            self.last_reset = current_date
    
    def _reset_if_new_hour(self):
        """Reset hourly token counter if it's a new hour."""
        current_hour = datetime.now().replace(minute=0, second=0, microsecond=0)
        if current_hour != self.last_hour_reset:
            self.last_hour_tokens = 0
            self.last_hour_reset = current_hour
    
    def check_cost_limit(self, estimated_cost: float) -> bool:
        """Check if operation would exceed daily cost limit."""
        self._reset_if_new_day()
        
        if self.daily_cost + estimated_cost > self.max_cost_per_day:
            logger.warning("Daily cost limit exceeded", 
                         daily_cost=self.daily_cost, 
                         estimated_cost=estimated_cost)
            return False
        
        return True
    
    def check_rate_limit(self, estimated_tokens: int) -> bool:
        """Check if operation would exceed hourly rate limit."""
        self._reset_if_new_hour()
        
        if self.last_hour_tokens + estimated_tokens > self.rate_limit_tokens:
            logger.warning("Hourly rate limit exceeded", 
                         current_tokens=self.last_hour_tokens,
                         estimated_tokens=estimated_tokens)
            return False
        
        return True
    
    def record_cost(self, actual_cost: float, actual_tokens: int):
        """Record actual cost and tokens of embedding operation."""
        self.daily_cost += actual_cost
        self.last_hour_tokens += actual_tokens
        logger.info("Cost recorded", 
                   actual_cost=actual_cost, 
                   daily_total=self.daily_cost,
                   hourly_tokens=self.last_hour_tokens)


class EmbeddingWorker:
    """Main embedding worker service."""
    
    def __init__(self, provider: EmbeddingProvider, batcher: EmbeddingBatcher, cost_guard: EmbeddingCostGuard):
        self.provider = provider
        self.batcher = batcher
        self.cost_guard = cost_guard
        self.metrics = EmbeddingMetrics()
    
    async def enqueue_job(self, tenant_id: str, chunk_id: str, email_id: str, 
                         content: str, priority: int = 0) -> str:
        """Enqueue a new embedding job."""
        async with get_database_session() as session:
            # Check if job already exists
            existing_job = await session.get(EmbeddingJob, chunk_id)
            if existing_job:
                logger.info("Embedding job already exists", chunk_id=chunk_id)
                return existing_job.id
            
            # Create new job
            job = EmbeddingJob(
                id=str(uuid.uuid4()),
                tenant_id=tenant_id,
                chunk_id=chunk_id,
                email_id=email_id,
                status="pending",
                priority=priority,
                job_metadata={"content_length": len(content)}
            )
            
            session.add(job)
            await session.commit()
            
            logger.info("Embedding job enqueued", 
                       job_id=job.id, 
                       tenant_id=tenant_id,
                       chunk_id=chunk_id)
            
            return job.id
    
    async def process_batch(self, jobs: List[EmbeddingJob]) -> List[Dict[str, Any]]:
        """Process a batch of embedding jobs."""
        if not jobs:
            return []
        
        tenant_id = jobs[0].tenant_id
        texts = [job.job_metadata.get("content", "") for job in jobs]
        
        try:
            # Check cost and rate limits
            estimated_cost = await self.provider.estimate_cost(len(texts))
            estimated_tokens = sum(len(text) // 4 for text in texts)
            
            if not self.cost_guard.check_cost_limit(estimated_cost):
                raise Exception("Daily cost limit exceeded")
            
            if not self.cost_guard.check_rate_limit(estimated_tokens):
                raise Exception("Hourly rate limit exceeded")
            
            # Generate embeddings
            start_time = time.time()
            vectors = await self.provider.embed_batch(texts)
            latency = time.time() - start_time
            
            # Record metrics
            self.metrics.record_job_completion(True, latency, len(jobs))
            
            # Update job statuses
            results = []
            for job, vector in zip(jobs, vectors):
                job.status = "completed"
                job.embedding_vector = vector
                job.completed_at = datetime.utcnow()
                job.job_metadata["embedding_dimensions"] = len(vector)
                job.job_metadata["processing_latency"] = latency
                
                results.append({
                    "job_id": job.id,
                    "status": "completed",
                    "vector": vector,
                    "metadata": job.job_metadata
                })
            
            # Record actual cost
            actual_cost = await self.provider.estimate_cost(len(texts))
            self.cost_guard.record_cost(actual_cost, estimated_tokens)
            
            logger.info("Batch processing completed", 
                       batch_size=len(jobs), 
                       tenant_id=tenant_id,
                       latency=latency)
            
            return results
            
        except Exception as e:
            # Mark jobs as failed
            for job in jobs:
                job.status = "failed"
                job.error_message = str(e)
                job.retry_count += 1
            
            self.metrics.record_job_completion(False, 0, len(jobs))
            
            logger.error("Batch processing failed", 
                        batch_size=len(jobs), 
                        tenant_id=tenant_id,
                        error=str(e))
            
            raise
    
    async def get_pending_jobs(self, tenant_id: str, limit: int = 100) -> List[EmbeddingJob]:
        """Get pending jobs for a tenant."""
        async with get_database_session() as session:
            # Query pending jobs ordered by priority and creation time
            query = """
                SELECT * FROM embedding_jobs 
                WHERE tenant_id = :tenant_id AND status = 'pending'
                ORDER BY priority DESC, created_at ASC
                LIMIT :limit
            """
            result = await session.execute(query, {"tenant_id": tenant_id, "limit": limit})
            return result.fetchall()
    
    async def retry_failed_jobs(self, tenant_id: str, max_retries: int = 3) -> int:
        """Retry failed jobs that haven't exceeded max retries."""
        async with get_database_session() as session:
            query = """
                UPDATE embedding_jobs 
                SET status = 'pending', updated_at = NOW()
                WHERE tenant_id = :tenant_id 
                AND status = 'failed' 
                AND retry_count < :max_retries
            """
            result = await session.execute(query, {"tenant_id": tenant_id, "max_retries": max_retries})
            await session.commit()
            
            retry_count = result.rowcount
            logger.info("Retried failed jobs", 
                       tenant_id=tenant_id, 
                       retry_count=retry_count)
            
            return retry_count


class EmbeddingMetrics:
    """Metrics collection for embedding operations."""
    
    def __init__(self):
        self.jobs_processed = 0
        self.jobs_failed = 0
        self.embeddings_generated = 0
        self.provider_latency = []
        self.batch_sizes = []
        self.cost_tracking = 0.0
    
    def record_job_completion(self, success: bool, latency: float, batch_size: int):
        """Record job completion metrics."""
        if success:
            self.jobs_processed += 1
            self.embeddings_generated += batch_size
        else:
            self.jobs_failed += 1
        
        self.provider_latency.append(latency)
        self.batch_sizes.append(batch_size)
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status."""
        total_jobs = self.jobs_processed + self.jobs_failed
        success_rate = self.jobs_processed / total_jobs if total_jobs > 0 else 0.0
        
        return {
            "jobs_processed": self.jobs_processed,
            "jobs_failed": self.jobs_failed,
            "success_rate": success_rate,
            "embeddings_generated": self.embeddings_generated,
            "avg_latency_p95": np.percentile(self.provider_latency, 95) if self.provider_latency else 0.0,
            "avg_batch_size": np.mean(self.batch_sizes) if self.batch_sizes else 0.0,
            "daily_cost": self.cost_tracking
        }


# Factory function for creating embedding worker
async def create_embedding_worker() -> EmbeddingWorker:
    """Create and configure embedding worker."""
    config = get_config()
    
    # Get provider configuration
    provider_config = config.get("models", {}).get("embeddings", {})
    provider_name = provider_config.get("provider", "openai")
    api_key = provider_config.get("api_key", "")
    
    # Create provider
    if provider_name == "openai":
        provider = OpenAIEmbeddingProvider(api_key)
    else:
        raise ValueError(f"Unsupported embedding provider: {provider_name}")
    
    # Create batcher and cost guard
    batch_size = provider_config.get("batch_size", 64)
    batcher = EmbeddingBatcher(batch_size=batch_size)
    cost_guard = EmbeddingCostGuard()
    
    return EmbeddingWorker(provider, batcher, cost_guard)
