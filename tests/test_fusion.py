"""Tests for fusion engine (EARS-RET-4, EARS-RET-5)."""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime
import time

from app.services.fusion import (
    ReciprocalRankFusion, Reranker, CrossEncoderReranker, StubReranker,
    HybridSearchEngine, FusionMetrics, create_fusion_engine, create_reranker,
    create_hybrid_search_engine
)


class TestReciprocalRankFusion:
    """Test Reciprocal Rank Fusion algorithm."""
    
    @pytest.fixture
    def fusion_engine(self):
        """Create fusion engine with default parameters."""
        return ReciprocalRankFusion(k=60, weight_bm25=1.0, weight_vector=1.0)
    
    @pytest.fixture
    def bm25_results(self):
        """Create sample BM25 search results."""
        return [
            {
                "score": 0.9,
                "citations": {"chunk_id": "chunk-1", "email_id": "email-1"},
                "subject": "Test Subject 1",
                "content": "Test content 1",
                "highlight": {"content": ["Test <em>content</em> 1"]}
            },
            {
                "score": 0.8,
                "citations": {"chunk_id": "chunk-2", "email_id": "email-2"},
                "subject": "Test Subject 2",
                "content": "Test content 2",
                "highlight": {"content": ["Test <em>content</em> 2"]}
            }
        ]
    
    @pytest.fixture
    def vector_results(self):
        """Create sample vector search results."""
        return [
            {
                "score": 0.95,
                "citations": {"chunk_id": "chunk-1", "email_id": "email-1"},
                "subject": "Test Subject 1",
                "content": "Test content 1",
                "highlight": {"content": ["Test <em>content</em> 1"]}
            },
            {
                "score": 0.7,
                "citations": {"chunk_id": "chunk-3", "email_id": "email-3"},
                "subject": "Test Subject 3",
                "content": "Test content 3",
                "highlight": {"content": ["Test <em>content</em> 3"]}
            }
        ]
    
    def test_fusion_engine_initialization(self, fusion_engine):
        """Test fusion engine initialization."""
        assert fusion_engine.k == 60
        assert fusion_engine.weight_bm25 == 1.0
        assert fusion_engine.weight_vector == 1.0
    
    def test_fuse_results_basic(self, fusion_engine, bm25_results, vector_results):
        """Test basic result fusion."""
        fused_results = fusion_engine.fuse_results(bm25_results, vector_results)
        
        # Should have 3 unique documents
        assert len(fused_results) == 3
        
        # Check that chunk-1 appears first (appears in both with high scores)
        assert fused_results[0]["citations"]["chunk_id"] == "chunk-1"
        assert "fused_score" in fused_results[0]
        assert fused_results[0]["search_sources"] == ["bm25", "vector"]
    
    def test_fuse_results_with_weights(self):
        """Test fusion with different weights."""
        fusion_engine = ReciprocalRankFusion(k=60, weight_bm25=2.0, weight_vector=1.0)
        
        bm25_results = [
            {
                "score": 0.9,
                "citations": {"chunk_id": "chunk-1", "email_id": "email-1"},
                "subject": "Test Subject 1"
            }
        ]
        
        vector_results = [
            {
                "score": 0.95,
                "citations": {"chunk_id": "chunk-1", "email_id": "email-1"},
                "subject": "Test Subject 1"
            }
        ]
        
        fused_results = fusion_engine.fuse_results(bm25_results, vector_results)
        
        # Check that the fusion engine has the correct weights
        assert fusion_engine.weight_bm25 == 2.0
        assert fusion_engine.weight_vector == 1.0
        
        # Check that the fused result has the expected structure
        assert fused_results[0]["citations"]["chunk_id"] == "chunk-1"
        assert "fused_score" in fused_results[0]
        assert fused_results[0]["search_sources"] == ["bm25", "vector"]
    
    def test_fuse_results_bm25_only(self, fusion_engine, bm25_results):
        """Test fusion when only BM25 has results."""
        fused_results = fusion_engine.fuse_results(bm25_results, [])
        
        assert len(fused_results) == 2
        for doc in fused_results:
            assert doc["search_sources"] == ["bm25"]
            assert doc["vector_score"] == 0.0
    
    def test_fuse_results_vector_only(self, fusion_engine, vector_results):
        """Test fusion when only vector has results."""
        fused_results = fusion_engine.fuse_results([], vector_results)
        
        assert len(fused_results) == 2
        for doc in fused_results:
            assert doc["search_sources"] == ["vector"]
            assert doc["bm25_score"] == 0.0
    
    def test_merge_highlights(self, fusion_engine, bm25_results, vector_results):
        """Test highlight merging."""
        fused_results = fusion_engine.fuse_results(bm25_results, vector_results)
        
        # Find chunk-1 which appears in both
        chunk_1 = next(doc for doc in fused_results if doc["citations"]["chunk_id"] == "chunk-1")
        
        # Should have highlights from both sources
        assert "highlight" in chunk_1
        assert "content" in chunk_1["highlight"]
    
    def test_fusion_with_different_k_values(self):
        """Test fusion with different k values."""
        # Test with k=10 (more aggressive ranking)
        fusion_engine = ReciprocalRankFusion(k=10)
        
        bm25_results = [
            {
                "score": 0.9,
                "citations": {"chunk_id": "chunk-1", "email_id": "email-1"},
                "subject": "Test Subject 1"
            }
        ]
        
        vector_results = [
            {
                "score": 0.95,
                "citations": {"chunk_id": "chunk-1", "email_id": "email-1"},
                "subject": "Test Subject 1"
            }
        ]
        
        fused_results = fusion_engine.fuse_results(bm25_results, vector_results)
        
        # With k=10, the RRF scores should be higher
        assert fused_results[0]["fused_score"] > 0.1


class TestReranker:
    """Test reranker interface and implementations."""
    
    def test_reranker_interface(self):
        """Test that Reranker abstract methods raise NotImplementedError."""
        # Create a concrete subclass that doesn't implement the abstract methods
        class ConcreteReranker(Reranker):
            pass
        
        reranker = ConcreteReranker()
        
        # Test that abstract methods raise NotImplementedError
        with pytest.raises(NotImplementedError):
            asyncio.run(reranker.rerank("test", []))
        
        with pytest.raises(NotImplementedError):
            reranker.get_model_info()
    
    def test_stub_reranker(self):
        """Test stub reranker functionality."""
        reranker = StubReranker()
        
        documents = [
            {"id": "doc-1", "content": "Test 1"},
            {"id": "doc-2", "content": "Test 2"}
        ]
        
        # Stub reranker should return documents unchanged
        result = asyncio.run(reranker.rerank("test query", documents))
        assert result == documents
        
        # Test model info
        info = reranker.get_model_info()
        assert info["type"] == "stub"
        assert info["available"] is True
    
    @pytest.mark.asyncio
    async def test_cross_encoder_reranker_stub_mode(self):
        """Test cross-encoder reranker in stub mode."""
        # Mock the import to simulate missing sentence-transformers
        with patch('builtins.__import__', side_effect=ImportError("No module named 'sentence_transformers'")):
            # Also patch the logger to prevent the warning from triggering the import
            with patch('app.services.fusion.logger.warning'):
                reranker = CrossEncoderReranker()
                
                documents = [
                    {"id": "doc-1", "content": "Test 1"},
                    {"id": "doc-2", "content": "Test 2"}
                ]
                
                # Should return documents unchanged in stub mode
                result = await reranker.rerank("test query", documents)
                assert result == documents[:100]  # Limited by max_documents
    
    def test_cross_encoder_model_info(self):
        """Test cross-encoder model information."""
        reranker = CrossEncoderReranker("test-model")
        info = reranker.get_model_info()
        
        assert info["type"] == "cross_encoder"
        assert info["model"] == "test-model"
        assert info["description"] == "Cross-encoder for query-document relevance scoring"


class TestHybridSearchEngine:
    """Test hybrid search engine orchestration."""
    
    @pytest.fixture
    def mock_bm25_service(self):
        """Create mock BM25 service."""
        service = Mock()
        service.search = AsyncMock(return_value={
            "hits": {
                "total": {"value": 2, "relation": "eq"},
                "hits": [
                    {
                        "score": 0.9,
                        "citations": {"chunk_id": "chunk-1", "email_id": "email-1"},
                        "subject": "Test Subject 1"
                    }
                ]
            },
            "performance": {"search_time": 0.1}
        })
        service.get_health_status = AsyncMock(return_value={"status": "healthy"})
        return service
    
    @pytest.fixture
    def mock_vector_service(self):
        """Create mock vector service."""
        service = Mock()
        service.search = AsyncMock(return_value={
            "hits": {
                "total": {"value": 1, "relation": "eq"},
                "hits": [
                    {
                        "score": 0.95,
                        "citations": {"chunk_id": "chunk-1", "email_id": "email-1"},
                        "subject": "Test Subject 1"
                    }
                ]
            },
            "performance": {"search_time": 0.15}
        })
        service.get_health_status = AsyncMock(return_value={"status": "healthy"})
        return service
    
    @pytest.fixture
    def fusion_engine(self):
        """Create fusion engine."""
        return ReciprocalRankFusion(k=60, weight_bm25=1.0, weight_vector=1.0)
    
    @pytest.fixture
    def hybrid_engine(self, mock_bm25_service, mock_vector_service, fusion_engine):
        """Create hybrid search engine."""
        return HybridSearchEngine(mock_bm25_service, mock_vector_service, fusion_engine)
    
    @pytest.mark.asyncio
    async def test_hybrid_search_basic(self, hybrid_engine):
        """Test basic hybrid search."""
        result = await hybrid_engine.hybrid_search(
            tenant_id="tenant-123",
            query="test query",
            size=10
        )
        
        assert result["search_type"] == "hybrid"
        assert "hits" in result
        assert "performance" in result
        assert "search_sources" in result
        
        # Verify performance metrics
        performance = result["performance"]
        assert "total_time" in performance
        assert "bm25_time" in performance
        assert "vector_time" in performance
        assert "fusion_time" in performance
    
    @pytest.mark.asyncio
    async def test_hybrid_search_with_reranking(self, hybrid_engine):
        """Test hybrid search with reranking enabled."""
        # Add reranker to engine
        reranker = StubReranker()
        hybrid_engine.reranker = reranker
        
        result = await hybrid_engine.hybrid_search(
            tenant_id="tenant-123",
            query="test query",
            size=10,
            enable_reranking=True
        )
        
        assert result["performance"]["rerank_time"] >= 0.0
    
    @pytest.mark.asyncio
    async def test_hybrid_search_analytics(self, hybrid_engine):
        """Test search analytics retrieval."""
        analytics = await hybrid_engine.get_search_analytics("tenant-123")
        
        assert analytics["tenant_id"] == "tenant-123"
        assert "bm25_metrics" in analytics
        assert "vector_metrics" in analytics
        assert "fusion_config" in analytics
        assert "performance_stats" in analytics
        assert "health_status" in analytics
    
    @pytest.mark.asyncio
    async def test_hybrid_search_error_handling(self, hybrid_engine):
        """Test error handling in hybrid search."""
        # Make BM25 service fail
        hybrid_engine.bm25_service.search.side_effect = Exception("BM25 search failed")
        
        with pytest.raises(Exception, match="BM25 search failed"):
            await hybrid_engine.hybrid_search(
                tenant_id="tenant-123",
                query="test query"
            )


class TestFusionMetrics:
    """Test fusion metrics collection."""
    
    @pytest.fixture
    def metrics(self):
        """Create metrics instance."""
        return FusionMetrics()
    
    def test_metrics_initialization(self, metrics):
        """Test metrics initialization."""
        assert len(metrics.search_times) == 0
        assert len(metrics.fusion_times) == 0
        assert len(metrics.rerank_times) == 0
        assert len(metrics.result_counts) == 0
    
    def test_record_search(self, metrics):
        """Test recording search metrics."""
        metrics.record_search(
            search_time=0.5,
            bm25_hits=10,
            vector_hits=8,
            fused_hits=12,
            rerank_time=0.1
        )
        
        assert len(metrics.search_times) == 1
        assert metrics.search_times[0] == 0.5
        assert metrics.fusion_times[0] == 0.4
        assert metrics.rerank_times[0] == 0.1
        assert metrics.result_counts[0]["bm25"] == 10
        assert metrics.result_counts[0]["vector"] == 8
        assert metrics.result_counts[0]["fused"] == 12
    
    def test_performance_stats_empty(self, metrics):
        """Test performance stats when no data."""
        stats = metrics.get_performance_stats()
        assert stats == {}
    
    def test_performance_stats_with_data(self, metrics):
        """Test performance stats with recorded data."""
        # Record multiple searches
        for i in range(5):
            metrics.record_search(
                search_time=0.1 + i * 0.1,
                bm25_hits=10 + i,
                vector_hits=8 + i,
                fused_hits=12 + i,
                rerank_time=0.01
            )
        
        stats = metrics.get_performance_stats()
        
        assert stats["total_searches"] == 5
        assert "avg_search_time" in stats
        assert "p95_search_time" in stats
        assert "avg_fusion_time" in stats
        assert "avg_rerank_time" in stats
        assert "avg_result_counts" in stats
    
    def test_health_status_empty(self, metrics):
        """Test health status when no data."""
        health = metrics.get_health_status()
        assert health["status"] == "no_data"
        assert "message" in health
    
    def test_health_status_excellent(self, metrics):
        """Test health status for excellent performance."""
        metrics.record_search(0.05, 10, 8, 12, 0.01)
        
        health = metrics.get_health_status()
        assert health["status"] == "excellent"
        assert health["avg_search_time"] == 0.05
    
    def test_health_status_poor(self, metrics):
        """Test health status for poor performance."""
        metrics.record_search(1.5, 10, 8, 12, 0.01)
        
        health = metrics.get_health_status()
        assert health["status"] == "poor"
        assert health["avg_search_time"] == 1.5


class TestFusionFactoryFunctions:
    """Test factory functions for creating fusion components."""
    
    def test_create_fusion_engine(self):
        """Test fusion engine factory."""
        engine = create_fusion_engine(k=100, weight_bm25=2.0, weight_vector=1.5)
        
        assert isinstance(engine, ReciprocalRankFusion)
        assert engine.k == 100
        assert engine.weight_bm25 == 2.0
        assert engine.weight_vector == 1.5
    
    def test_create_reranker_stub(self):
        """Test reranker factory with stub type."""
        reranker = create_reranker("stub")
        assert isinstance(reranker, StubReranker)
    
    def test_create_reranker_cross_encoder(self):
        """Test reranker factory with cross-encoder type."""
        reranker = create_reranker("cross_encoder", "test-model")
        assert isinstance(reranker, CrossEncoderReranker)
        assert reranker.model_name == "test-model"
    
    def test_create_hybrid_search_engine(self):
        """Test hybrid search engine factory."""
        mock_bm25 = Mock()
        mock_vector = Mock()
        
        engine = create_hybrid_search_engine(
            mock_bm25, mock_vector,
            k=80, weight_bm25=1.5, weight_vector=1.0,
            reranker_type="stub"
        )
        
        assert isinstance(engine, HybridSearchEngine)
        assert engine.bm25_service == mock_bm25
        assert engine.vector_service == mock_vector
        assert engine.fusion_engine.k == 80
        assert engine.fusion_engine.weight_bm25 == 1.5
        assert engine.fusion_engine.weight_vector == 1.0
        assert isinstance(engine.reranker, StubReranker)


class TestFusionIntegration:
    """Integration tests for fusion engine."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_fusion_workflow(self):
        """Test complete fusion workflow."""
        # Create fusion engine
        fusion_engine = ReciprocalRankFusion(k=60, weight_bm25=1.0, weight_vector=1.0)
        
        # Create mock search results
        bm25_results = [
            {
                "score": 0.9,
                "citations": {"chunk_id": "chunk-1", "email_id": "email-1"},
                "subject": "Logistics Query",
                "content": "This is about logistics and shipping",
                "highlight": {"content": ["This is about <em>logistics</em>"]}
            },
            {
                "score": 0.7,
                "citations": {"chunk_id": "chunk-2", "email_id": "email-2"},
                "subject": "General Query",
                "content": "This is a general email",
                "highlight": {"content": ["This is a <em>general</em> email"]}
            }
        ]
        
        vector_results = [
            {
                "score": 0.95,
                "citations": {"chunk_id": "chunk-1", "email_id": "email-1"},
                "subject": "Logistics Query",
                "content": "This is about logistics and shipping",
                "highlight": {"content": ["This is about <em>logistics</em>"]}
            },
            {
                "score": 0.6,
                "citations": {"chunk_id": "chunk-3", "email_id": "email-3"},
                "subject": "Another Query",
                "content": "This is another email",
                "highlight": {"content": ["This is <em>another</em> email"]}
            }
        ]
        
        # Fuse results
        fused_results = fusion_engine.fuse_results(bm25_results, vector_results)
        
        # Verify fusion results
        assert len(fused_results) == 3  # 3 unique documents
        
        # chunk-1 should be first (appears in both with high scores)
        assert fused_results[0]["citations"]["chunk_id"] == "chunk-1"
        assert fused_results[0]["search_sources"] == ["bm25", "vector"]
        assert "fused_score" in fused_results[0]
        
        # Verify all documents have proper metadata
        for doc in fused_results:
            assert "fused_score" in doc
            assert "search_sources" in doc
            assert "citations" in doc
            assert "chunk_id" in doc["citations"]


# Test data for parametrized tests
@pytest.mark.parametrize("k,weight_bm25,weight_vector", [
    (60, 1.0, 1.0),
    (100, 2.0, 1.0),
    (30, 1.0, 2.0),
    (80, 1.5, 1.5),
])
def test_fusion_engine_parameters(k, weight_bm25, weight_vector):
    """Test fusion engine with different parameters."""
    engine = ReciprocalRankFusion(k=k, weight_bm25=weight_bm25, weight_vector=weight_vector)
    
    assert engine.k == k
    assert engine.weight_bm25 == weight_bm25
    assert engine.weight_vector == weight_vector


@pytest.mark.parametrize("reranker_type,expected_class", [
    ("stub", StubReranker),
    ("cross_encoder", CrossEncoderReranker),
])
def test_reranker_factory_types(reranker_type, expected_class):
    """Test reranker factory with different types."""
    reranker = create_reranker(reranker_type)
    assert isinstance(reranker, expected_class)
