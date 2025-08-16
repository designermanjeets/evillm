"""Tests for embedding service (EARS-RET-1, EARS-RET-2)."""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta
import numpy as np

from app.services.embeddings import (
    EmbeddingProvider, OpenAIEmbeddingProvider, EmbeddingBatcher,
    EmbeddingCostGuard, EmbeddingWorker, EmbeddingMetrics, create_embedding_worker
)
from app.database.models import EmbeddingJob


class TestEmbeddingProvider:
    """Test embedding provider interface and implementations."""
    
    def test_abstract_provider_interface(self):
        """Test that EmbeddingProvider is abstract and cannot be instantiated."""
        with pytest.raises(TypeError):
            EmbeddingProvider()
    
    @pytest.mark.asyncio
    async def test_openai_provider_stub_mode(self):
        """Test OpenAI provider in stub mode (no API key)."""
        provider = OpenAIEmbeddingProvider("")
        
        # Should work in stub mode
        dimensions = await provider.get_embedding_dimensions()
        assert dimensions == 1536
        
        # Should return stub vectors
        texts = ["Hello world", "Test text"]
        vectors = await provider.embed_batch(texts)
        assert len(vectors) == 2
        assert all(len(vector) == 1536 for vector in vectors)
        assert all(all(val == 0.1 for val in vector) for vector in vectors)
    
    @pytest.mark.asyncio
    async def test_openai_provider_cost_estimation(self):
        """Test cost estimation for OpenAI provider."""
        provider = OpenAIEmbeddingProvider("test_key")
        
        # Test cost estimation
        cost = await provider.estimate_cost(1000)  # 1000 characters ≈ 250 tokens
        expected_cost = (250 / 1000) * 0.00013  # OpenAI pricing
        assert abs(cost - expected_cost) < 0.00001


class TestEmbeddingBatcher:
    """Test embedding batcher functionality."""
    
    def test_batcher_initialization(self):
        """Test batcher initialization with custom parameters."""
        batcher = EmbeddingBatcher(batch_size=32, max_wait_time=10.0)
        assert batcher.batch_size == 32
        assert batcher.max_wait_time == 10.0
        assert len(batcher.current_batch) == 0
    
    def test_batcher_add_job_batch_ready(self):
        """Test that batcher signals when batch is ready."""
        batcher = EmbeddingBatcher(batch_size=2, max_wait_time=5.0)
        
        # Add first job
        job1 = Mock(spec=EmbeddingJob)
        ready = asyncio.run(batcher.add_job(job1))
        assert not ready
        assert len(batcher.current_batch) == 1
        
        # Add second job - batch should be ready
        job2 = Mock(spec=EmbeddingJob)
        ready = asyncio.run(batcher.add_job(job2))
        assert ready
        assert len(batcher.current_batch) == 2
    
    def test_batcher_timeout_trigger(self):
        """Test that batcher triggers on timeout."""
        batcher = EmbeddingBatcher(batch_size=100, max_wait_time=0.1)
        
        # Add one job
        job = Mock(spec=EmbeddingJob)
        ready = asyncio.run(batcher.add_job(job))
        assert not ready
        
        # Wait for timeout
        asyncio.run(asyncio.sleep(0.2))
        
        # Should be ready due to timeout
        ready = asyncio.run(batcher.add_job(job))
        assert ready
    
    def test_batcher_get_batch(self):
        """Test that get_batch returns current batch and resets."""
        batcher = EmbeddingBatcher(batch_size=2)
        
        # Add jobs
        job1, job2 = Mock(spec=EmbeddingJob), Mock(spec=EmbeddingJob)
        asyncio.run(batcher.add_job(job1))
        asyncio.run(batcher.add_job(job2))
        
        # Get batch
        batch = batcher.get_batch()
        assert len(batch) == 2
        assert batch[0] == job1
        assert batch[1] == job2
        
        # Current batch should be empty
        assert len(batcher.current_batch) == 0


class TestEmbeddingCostGuard:
    """Test cost guard functionality."""
    
    def test_cost_guard_initialization(self):
        """Test cost guard initialization."""
        guard = EmbeddingCostGuard(max_cost_per_day=50.0)
        assert guard.max_cost_per_day == 50.0
        assert guard.daily_cost == 0.0
    
    def test_cost_limit_check(self):
        """Test daily cost limit checking."""
        guard = EmbeddingCostGuard(max_cost_per_day=100.0)
        
        # Should allow operation within limit
        assert guard.check_cost_limit(50.0)
        guard.record_cost(50.0, 1000)  # Record the cost
        
        assert guard.check_cost_limit(50.0)
        guard.record_cost(50.0, 1000)  # Record the cost
        
        # Should reject operation that would exceed limit
        assert not guard.check_cost_limit(10.0)
    
    def test_rate_limit_check(self):
        """Test hourly rate limit checking."""
        guard = EmbeddingCostGuard()
        
        # Should allow operation within rate limit
        assert guard.check_rate_limit(50000)  # 50k tokens
        guard.record_cost(0.0, 50000)  # Record the tokens
        
        assert guard.check_rate_limit(50000)  # 50k tokens
        guard.record_cost(0.0, 50000)  # Record the tokens
        
        # Should reject operation that would exceed rate limit
        assert not guard.check_rate_limit(10000)  # Would exceed 100k limit
    
    def test_cost_recording(self):
        """Test cost and token recording."""
        guard = EmbeddingCostGuard()
        
        # Record costs
        guard.record_cost(25.0, 50000)
        assert guard.daily_cost == 25.0
        assert guard.last_hour_tokens == 50000
        
        guard.record_cost(25.0, 50000)
        assert guard.daily_cost == 50.0
        assert guard.last_hour_tokens == 100000


class TestEmbeddingMetrics:
    """Test embedding metrics collection."""
    
    def test_metrics_initialization(self):
        """Test metrics initialization."""
        metrics = EmbeddingMetrics()
        assert metrics.jobs_processed == 0
        assert metrics.jobs_failed == 0
        assert metrics.embeddings_generated == 0
    
    def test_job_completion_recording(self):
        """Test recording job completion metrics."""
        metrics = EmbeddingMetrics()
        
        # Record successful job
        metrics.record_job_completion(True, 1.5, 64)
        assert metrics.jobs_processed == 1
        assert metrics.jobs_failed == 0
        assert metrics.embeddings_generated == 64
        assert len(metrics.provider_latency) == 1
        assert len(metrics.batch_sizes) == 1
        
        # Record failed job
        metrics.record_job_completion(False, 0.0, 32)
        assert metrics.jobs_processed == 1
        assert metrics.jobs_failed == 1
        assert metrics.embeddings_generated == 64  # Unchanged
    
    def test_health_status_calculation(self):
        """Test health status calculation."""
        metrics = EmbeddingMetrics()
        
        # Add some metrics
        metrics.record_job_completion(True, 1.0, 64)
        metrics.record_job_completion(True, 2.0, 64)
        metrics.record_job_completion(False, 0.0, 32)
        
        health = metrics.get_health_status()
        assert health["jobs_processed"] == 2
        assert health["jobs_failed"] == 1
        assert health["success_rate"] == 2/3
        assert health["embeddings_generated"] == 128
        # The average batch size should be (64 + 64 + 32) / 3 = 53.33...
        assert abs(health["avg_batch_size"] - 53.33) < 0.1


class TestEmbeddingWorker:
    """Test embedding worker service."""
    
    @pytest.fixture
    def mock_provider(self):
        """Create mock embedding provider."""
        provider = Mock(spec=EmbeddingProvider)
        provider.get_embedding_dimensions = AsyncMock(return_value=1536)
        provider.estimate_cost = AsyncMock(return_value=0.01)
        provider.embed_batch = AsyncMock(return_value=[[0.1] * 1536] * 2)
        return provider
    
    @pytest.fixture
    def mock_batcher(self):
        """Create mock batcher."""
        return Mock(spec=EmbeddingBatcher)
    
    @pytest.fixture
    def mock_cost_guard(self):
        """Create mock cost guard."""
        guard = Mock(spec=EmbeddingCostGuard)
        guard.check_cost_limit.return_value = True
        guard.check_rate_limit.return_value = True
        return guard
    
    @pytest.fixture
    def worker(self, mock_provider, mock_batcher, mock_cost_guard):
        """Create embedding worker with mocked dependencies."""
        return EmbeddingWorker(mock_provider, mock_batcher, mock_cost_guard)
    
    @pytest.mark.asyncio
    async def test_worker_initialization(self, worker):
        """Test worker initialization."""
        assert worker.provider is not None
        assert worker.batcher is not None
        assert worker.cost_guard is not None
        assert worker.metrics is not None
    
    @pytest.mark.asyncio
    async def test_enqueue_job_success(self, worker):
        """Test successful job enqueueing."""
        with patch('app.services.embeddings.get_database_session') as mock_session:
            mock_session.return_value.__aenter__.return_value.get.return_value = None
            mock_session.return_value.__aenter__.return_value.add = Mock()
            mock_session.return_value.__aenter__.return_value.commit = AsyncMock()
            
            job_id = await worker.enqueue_job(
                tenant_id="tenant-123",
                chunk_id="chunk-456",
                email_id="email-789",
                content="Test content"
            )
            
            assert job_id is not None
            assert isinstance(job_id, str)
    
    @pytest.mark.asyncio
    async def test_enqueue_job_duplicate(self, worker):
        """Test enqueueing duplicate job."""
        with patch('app.services.embeddings.get_database_session') as mock_session:
            existing_job = Mock()
            existing_job.id = "existing-123"
            mock_session.return_value.__aenter__.return_value.get.return_value = existing_job
            
            job_id = await worker.enqueue_job(
                tenant_id="tenant-123",
                chunk_id="chunk-456",
                email_id="email-789",
                content="Test content"
            )
            
            assert job_id == "existing-123"
    
    @pytest.mark.asyncio
    async def test_process_batch_success(self, worker, mock_provider, mock_cost_guard):
        """Test successful batch processing."""
        # Create mock jobs with proper metadata
        jobs = [
            Mock(spec=EmbeddingJob, tenant_id="tenant-123", job_metadata={"content": "Text 1"}),
            Mock(spec=EmbeddingJob, tenant_id="tenant-123", job_metadata={"content": "Text 2"})
        ]
        
        # Configure mocks
        mock_cost_guard.check_cost_limit.return_value = True
        mock_cost_guard.check_rate_limit.return_value = True
        mock_provider.embed_batch.return_value = [[0.1] * 1536, [0.2] * 1536]
        
        # Process batch
        results = await worker.process_batch(jobs)
        
        # Verify results
        assert len(results) == 2
        assert all(result["status"] == "completed" for result in results)
        assert all("vector" in result for result in results)
        
        # Verify job updates
        for job in jobs:
            assert job.status == "completed"
            assert job.embedding_vector is not None
            assert job.completed_at is not None
    
    @pytest.mark.asyncio
    async def test_process_batch_cost_limit_exceeded(self, worker, mock_cost_guard):
        """Test batch processing with cost limit exceeded."""
        # Create mock job with proper attributes
        mock_job = Mock(spec=EmbeddingJob, tenant_id="tenant-123", job_metadata={"content": "Text"})
        mock_job.retry_count = 0  # Set initial retry count
        jobs = [mock_job]
        
        mock_cost_guard.check_cost_limit.return_value = False
        
        with pytest.raises(Exception, match="Daily cost limit exceeded"):
            await worker.process_batch(jobs)
        
        # Verify jobs marked as failed
        for job in jobs:
            assert job.status == "failed"
            assert job.error_message == "Daily cost limit exceeded"
            assert job.retry_count == 1
    
    @pytest.mark.asyncio
    async def test_process_batch_rate_limit_exceeded(self, worker, mock_cost_guard):
        """Test batch processing with rate limit exceeded."""
        # Create mock job with proper attributes
        mock_job = Mock(spec=EmbeddingJob, tenant_id="tenant-123", job_metadata={"content": "Text"})
        mock_job.retry_count = 0  # Set initial retry count
        jobs = [mock_job]
        
        mock_cost_guard.check_cost_limit.return_value = True
        mock_cost_guard.check_rate_limit.return_value = False
        
        with pytest.raises(Exception, match="Hourly rate limit exceeded"):
            await worker.process_batch(jobs)
    
    @pytest.mark.asyncio
    async def test_get_pending_jobs(self, worker):
        """Test retrieving pending jobs."""
        with patch('app.services.embeddings.get_database_session') as mock_session:
            mock_jobs = [Mock(), Mock()]
            mock_result = Mock()
            mock_result.fetchall.return_value = mock_jobs
            mock_session.return_value.__aenter__.return_value.execute = AsyncMock(return_value=mock_result)
            
            jobs = await worker.get_pending_jobs("tenant-123", limit=50)
            
            assert jobs == mock_jobs
    
    @pytest.mark.asyncio
    async def test_retry_failed_jobs(self, worker):
        """Test retrying failed jobs."""
        with patch('app.services.embeddings.get_database_session') as mock_session:
            mock_session.return_value.__aenter__.return_value.execute.return_value.rowcount = 5
            mock_session.return_value.__aenter__.return_value.commit = AsyncMock()
            
            retry_count = await worker.retry_failed_jobs("tenant-123", max_retries=3)
            
            assert retry_count == 5


class TestEmbeddingServiceIntegration:
    """Integration tests for embedding service."""
    
    @pytest.mark.asyncio
    async def test_create_embedding_worker_factory(self):
        """Test embedding worker factory function."""
        with patch('app.services.embeddings.get_config') as mock_get_config:
            mock_get_config.return_value = {
                "models": {
                    "embeddings": {
                        "provider": "openai",
                        "api_key": "test-key",
                        "batch_size": 32
                    }
                }
            }
            
            worker = await create_embedding_worker()
            
            assert worker is not None
            assert isinstance(worker, EmbeddingWorker)
            assert worker.batcher.batch_size == 32
    
    @pytest.mark.asyncio
    async def test_create_embedding_worker_unsupported_provider(self):
        """Test factory with unsupported provider."""
        with patch('app.services.embeddings.get_config') as mock_get_config:
            mock_get_config.return_value = {
                "models": {
                    "embeddings": {
                        "provider": "unsupported",
                        "api_key": "test-key"
                    }
                }
            }
            
            with pytest.raises(ValueError, match="Unsupported embedding provider: unsupported"):
                await create_embedding_worker()


# Test data for parametrized tests
@pytest.mark.parametrize("batch_size,max_wait,expected_ready", [
    (2, 5.0, True),   # Batch size reached
    (100, 0.1, True), # Timeout reached
    (10, 5.0, False), # Neither condition met
])
def test_batcher_ready_conditions(batch_size, max_wait, expected_ready):
    """Test batcher ready conditions with different parameters."""
    batcher = EmbeddingBatcher(batch_size=batch_size, max_wait_time=max_wait)
    
    # Add one job
    job = Mock(spec=EmbeddingJob)
    ready = asyncio.run(batcher.add_job(job))
    
    if batch_size == 2:
        # Add second job to trigger batch size
        job2 = Mock(spec=EmbeddingJob)
        ready = asyncio.run(batcher.add_job(job2))
    
    if max_wait == 0.1:
        # Wait for timeout
        asyncio.run(asyncio.sleep(0.2))
        ready = asyncio.run(batcher.add_job(job))
    
    assert ready == expected_ready


@pytest.mark.parametrize("daily_cost,max_cost,estimated_cost,expected_allowed", [
    (50.0, 100.0, 25.0, True),   # Within limit
    (50.0, 100.0, 60.0, False),  # Would exceed limit
    (0.0, 100.0, 100.0, True),   # Exactly at limit
])
def test_cost_guard_limits(daily_cost, max_cost, estimated_cost, expected_allowed):
    """Test cost guard with different cost scenarios."""
    guard = EmbeddingCostGuard(max_cost_per_day=max_cost)
    guard.daily_cost = daily_cost
    
    allowed = guard.check_cost_limit(estimated_cost)
    assert allowed == expected_allowed
