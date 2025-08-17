"""Tests for search metrics functionality."""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime, timedelta
from fastapi.testclient import TestClient

from app.services.search_metrics import (
    SearchMetrics, 
    SearchQualityMetrics, 
    SearchPerformanceSummary,
    SearchMetricsCollector,
    get_search_metrics_collector
)
from app.main import create_app


class TestSearchMetrics:
    """Test search metrics data structures and calculations."""
    
    def test_search_metrics_creation(self):
        """Test SearchMetrics dataclass creation."""
        metrics = SearchMetrics(
            query_id="test-123",
            tenant_id="tenant-123",
            query="test query",
            timestamp=datetime.now(),
            total_time_ms=100.0,
            bm25_time_ms=50.0,
            vector_time_ms=30.0,
            fusion_time_ms=20.0,
            bm25_results=5,
            vector_results=5,
            fused_results=10,
            final_results=10,
            k_requested=10,
            search_type="hybrid",
            success=True
        )
        
        assert metrics.query_id == "test-123"
        assert metrics.tenant_id == "tenant-123"
        assert metrics.total_time_ms == 100.0
        assert metrics.search_type == "hybrid"
        assert metrics.success is True
    
    def test_search_quality_metrics_creation(self):
        """Test SearchQualityMetrics dataclass creation."""
        quality_metrics = SearchQualityMetrics(
            query_id="test-123",
            tenant_id="tenant-123",
            timestamp=datetime.now(),
            hit_at_k={1: 1.0, 3: 0.67, 5: 0.8, 10: 0.7},
            mean_reciprocal_rank=0.5,
            normalized_discounted_cumulative_gain=0.8,
            precision_at_k={1: 1.0, 3: 0.67, 5: 0.8, 10: 0.7},
            recall_at_k={1: 0.5, 3: 1.0, 5: 1.0, 10: 1.0},
            f1_at_k={1: 0.67, 3: 0.8, 5: 0.89, 10: 0.82},
            relevance_scores=[0.9, 0.8, 0.7, 0.6, 0.5],
            citation_coverage=1.0
        )
        
        assert quality_metrics.query_id == "test-123"
        assert quality_metrics.hit_at_k[1] == 1.0
        assert quality_metrics.mean_reciprocal_rank == 0.5
        assert quality_metrics.citation_coverage == 1.0


class TestSearchMetricsCollector:
    """Test SearchMetricsCollector functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.collector = SearchMetricsCollector(max_metrics_history=100)
    
    def test_record_search_metrics(self):
        """Test recording search metrics."""
        metrics = SearchMetrics(
            query_id="test-123",
            tenant_id="tenant-123",
            query="test query",
            timestamp=datetime.now(),
            total_time_ms=100.0,
            bm25_time_ms=50.0,
            vector_time_ms=30.0,
            fusion_time_ms=20.0,
            bm25_results=5,
            vector_results=5,
            fused_results=10,
            final_results=10,
            k_requested=10,
            search_type="hybrid",
            success=True
        )
        
        self.collector.record_search_metrics(metrics)
        
        assert len(self.collector.search_metrics) == 1
        assert len(self.collector.tenant_metrics["tenant-123"]["searches"]) == 1
    
    def test_record_quality_metrics(self):
        """Test recording quality metrics."""
        quality_metrics = SearchQualityMetrics(
            query_id="test-123",
            tenant_id="tenant-123",
            timestamp=datetime.now(),
            hit_at_k={1: 1.0, 3: 0.67, 5: 0.8, 10: 0.7},
            mean_reciprocal_rank=0.5,
            normalized_discounted_cumulative_gain=0.8,
            precision_at_k={1: 1.0, 3: 0.67, 5: 0.8, 10: 0.7},
            recall_at_k={1: 0.5, 3: 1.0, 5: 1.0, 10: 1.0},
            f1_at_k={1: 0.67, 3: 0.8, 5: 0.89, 10: 0.82},
            relevance_scores=[0.9, 0.8, 0.7, 0.6, 0.5],
            citation_coverage=1.0
        )
        
        self.collector.record_quality_metrics(quality_metrics)
        
        assert len(self.collector.quality_metrics) == 1
        assert len(self.collector.tenant_metrics["tenant-123"]["quality"]) == 1
    
    def test_calculate_hit_at_k(self):
        """Test hit@k calculation."""
        relevance_scores = [0.9, 0.8, 0.7, 0.6, 0.5]
        
        hit_at_1 = self.collector.calculate_hit_at_k(relevance_scores, 1)
        hit_at_3 = self.collector.calculate_hit_at_k(relevance_scores, 3)
        hit_at_5 = self.collector.calculate_hit_at_k(relevance_scores, 5)
        
        assert hit_at_1 == 1.0  # First result has score 0.9 > 0.5
        assert hit_at_3 == 1.0  # All first 3 results have scores > 0.5
        assert hit_at_5 == 0.8  # 4 out of 5 results have scores > 0.5 (0.9, 0.8, 0.7, 0.6)
    
    def test_calculate_mrr(self):
        """Test Mean Reciprocal Rank calculation."""
        relevance_scores = [0.3, 0.9, 0.4, 0.8, 0.6]
        
        mrr = self.collector.calculate_mrr(relevance_scores)
        
        # First relevant result (score > 0.5) is at index 1, so MRR = 1/(1+1) = 0.5
        assert mrr == 0.5
    
    def test_calculate_ndcg(self):
        """Test NDCG calculation."""
        relevance_scores = [0.9, 0.8, 0.7, 0.6, 0.5]
        
        ndcg = self.collector.calculate_ndcg(relevance_scores, 5)
        
        # Should be close to 1.0 since scores are already in descending order
        assert 0.9 <= ndcg <= 1.0
    
    def test_calculate_precision_recall_f1(self):
        """Test precision, recall, and F1 calculation."""
        relevance_scores = [0.9, 0.8, 0.3, 0.7, 0.4]
        
        precision, recall, f1 = self.collector.calculate_precision_recall_f1(relevance_scores, 3)
        
        # First 3 results: 2 relevant (0.9, 0.8), 1 not relevant (0.3)
        # Precision = 2/3 = 0.67
        # Total relevant in all results: 3 (0.9, 0.8, 0.7) - 0.4 is not > 0.5
        # Recall = 2/3 = 0.67
        # F1 = 2 * (0.67 * 0.67) / (0.67 + 0.67) = 0.67
        
        assert abs(precision - 0.67) < 0.01
        assert abs(recall - 0.67) < 0.01
        assert abs(f1 - 0.67) < 0.01
    
    def test_generate_quality_metrics(self):
        """Test quality metrics generation."""
        relevance_scores = [0.9, 0.8, 0.7, 0.6, 0.5]
        citation_coverage = 0.8
        
        quality_metrics = self.collector.generate_quality_metrics(
            "test-123", "tenant-123", relevance_scores, citation_coverage
        )
        
        assert quality_metrics.query_id == "test-123"
        assert quality_metrics.tenant_id == "tenant-123"
        assert quality_metrics.hit_at_k[1] == 1.0
        assert quality_metrics.hit_at_k[3] == 1.0
        assert quality_metrics.hit_at_k[5] == 0.8  # 4 out of 5 results > 0.5
        assert quality_metrics.citation_coverage == 0.8
    
    def test_get_performance_summary_empty(self):
        """Test performance summary when no metrics are available."""
        summary = self.collector.get_performance_summary("tenant-123", "24h")
        
        assert summary.tenant_id == "tenant-123"
        assert summary.time_window == "24h"
        assert summary.total_searches == 0
        assert summary.successful_searches == 0
        assert summary.failed_searches == 0
    
    def test_get_performance_summary_with_data(self):
        """Test performance summary with actual metrics data."""
        # Add some test metrics
        for i in range(5):
            metrics = SearchMetrics(
                query_id=f"test-{i}",
                tenant_id="tenant-123",
                query=f"test query {i}",
                timestamp=datetime.now(),
                total_time_ms=100.0 + i,
                bm25_time_ms=50.0,
                vector_time_ms=30.0,
                fusion_time_ms=20.0,
                bm25_results=5,
                vector_results=5,
                fused_results=10,
                final_results=10,
                k_requested=10,
                search_type="hybrid",
                success=True
            )
            self.collector.record_search_metrics(metrics)
        
        summary = self.collector.get_performance_summary("tenant-123", "24h")
        
        assert summary.tenant_id == "tenant-123"
        assert summary.total_searches == 5
        assert summary.successful_searches == 5
        assert summary.failed_searches == 0
        assert summary.avg_total_time_ms > 0
    
    def test_get_global_metrics(self):
        """Test global metrics aggregation."""
        # Add metrics for multiple tenants
        for tenant_id in ["tenant-1", "tenant-2"]:
            for i in range(3):
                metrics = SearchMetrics(
                    query_id=f"test-{tenant_id}-{i}",
                    tenant_id=tenant_id,
                    query=f"test query {i}",
                    timestamp=datetime.now(),
                    total_time_ms=100.0,
                    bm25_time_ms=50.0,
                    vector_time_ms=30.0,
                    fusion_time_ms=20.0,
                    bm25_results=5,
                    vector_results=5,
                    fused_results=10,
                    final_results=10,
                    k_requested=10,
                    search_type="hybrid",
                    success=True
                )
                self.collector.record_search_metrics(metrics)
        
        global_metrics = self.collector.get_global_metrics("24h")
        
        assert global_metrics["total_tenants"] == 2
        assert global_metrics["total_searches"] == 6
        assert global_metrics["successful_searches"] == 6
    
    def test_export_metrics(self):
        """Test metrics export functionality."""
        # Add some test metrics
        metrics = SearchMetrics(
            query_id="test-123",
            tenant_id="tenant-123",
            query="test query",
            timestamp=datetime.now(),
            total_time_ms=100.0,
            bm25_time_ms=50.0,
            vector_time_ms=30.0,
            fusion_time_ms=20.0,
            bm25_results=5,
            vector_results=5,
            fused_results=10,
            final_results=10,
            k_requested=10,
            search_type="hybrid",
            success=True
        )
        self.collector.record_search_metrics(metrics)
        
        # Export tenant-specific metrics
        exported = self.collector.export_metrics("tenant-123", "24h")
        
        assert exported["tenant_id"] == "tenant-123"
        assert exported["time_window"] == "24h"
        assert "summary" in exported
        assert "recent_searches" in exported
        
        # Export global metrics
        global_exported = self.collector.export_metrics(time_window="24h")
        
        assert global_exported["time_window"] == "24h"
        assert "global_summary" in global_exported
        assert "tenant_summaries" in global_exported
    
    def test_reset_metrics(self):
        """Test metrics reset functionality."""
        # Add some test metrics
        metrics = SearchMetrics(
            query_id="test-123",
            tenant_id="tenant-123",
            query="test query",
            timestamp=datetime.now(),
            total_time_ms=100.0,
            bm25_time_ms=50.0,
            vector_time_ms=30.0,
            fusion_time_ms=20.0,
            bm25_results=5,
            vector_results=5,
            fused_results=10,
            final_results=10,
            k_requested=10,
            search_type="hybrid",
            success=True
        )
        self.collector.record_search_metrics(metrics)
        
        # Verify metrics were added
        assert len(self.collector.search_metrics) == 1
        
        # Reset specific tenant
        self.collector.reset_metrics("tenant-123")
        
        # Verify tenant metrics were reset
        assert len(self.collector.tenant_metrics["tenant-123"]["searches"]) == 0
        
        # Global metrics should still exist
        assert len(self.collector.search_metrics) == 1
        
        # Reset all metrics
        self.collector.reset_metrics()
        
        # Verify all metrics were reset
        assert len(self.collector.search_metrics) == 0
        assert len(self.collector.tenant_metrics) == 0


class TestSearchMetricsAPI:
    """Test search metrics API endpoints."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.app = create_app()
        self.client = TestClient(self.app)
    
    @patch('app.routers.search_metrics.get_tenant_id')
    def test_health_check(self, mock_get_tenant_id):
        """Test search metrics health check endpoint."""
        response = self.client.get("/search-metrics/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "search-metrics"
    
    @patch('app.routers.search_metrics.get_tenant_id')
    def test_get_current_tenant_summary(self, mock_get_tenant_id):
        """Test getting current tenant summary."""
        mock_get_tenant_id.return_value = "tenant-123"
        
        response = self.client.get("/search-metrics/summary")
        
        assert response.status_code == 200
        data = response.json()
        assert data["tenant_id"] == "tenant-123"
        assert "summary" in data
    
    @patch('app.routers.search_metrics.get_tenant_id')
    def test_get_tenant_summary(self, mock_get_tenant_id):
        """Test getting specific tenant summary."""
        mock_get_tenant_id.return_value = "tenant-123"
        
        response = self.client.get("/search-metrics/summary/tenant-123")
        
        assert response.status_code == 200
        data = response.json()
        assert data["tenant_id"] == "tenant-123"
        assert "summary" in data
    
    @patch('app.routers.search_metrics.get_tenant_id')
    def test_get_tenant_summary_access_denied(self, mock_get_tenant_id):
        """Test access denied when requesting different tenant summary."""
        mock_get_tenant_id.return_value = "tenant-456"
        
        response = self.client.get("/search-metrics/summary/tenant-123")
        
        assert response.status_code == 403
        data = response.json()
        assert "Access denied" in data["detail"]
    
    @patch('app.routers.search_metrics.get_tenant_id')
    def test_get_global_metrics(self, mock_get_tenant_id):
        """Test getting global metrics."""
        mock_get_tenant_id.return_value = "tenant-123"
        
        response = self.client.get("/search-metrics/global")
        
        assert response.status_code == 200
        data = response.json()
        assert "global_metrics" in data
    
    @patch('app.routers.search_metrics.get_tenant_id')
    def test_export_current_tenant_metrics(self, mock_get_tenant_id):
        """Test exporting current tenant metrics."""
        mock_get_tenant_id.return_value = "tenant-123"
        
        response = self.client.get("/search-metrics/export")
        
        assert response.status_code == 200
        assert "Content-Disposition" in response.headers
    
    @patch('app.routers.search_metrics.get_tenant_id')
    def test_reset_current_tenant_metrics(self, mock_get_tenant_id):
        """Test resetting current tenant metrics."""
        mock_get_tenant_id.return_value = "tenant-123"
        
        response = self.client.delete("/search-metrics/reset")
        
        assert response.status_code == 200
        data = response.json()
        assert data["tenant_id"] == "tenant-123"
        assert "reset" in data["message"]
    
    @patch('app.routers.search_metrics.get_tenant_id')
    def test_get_recent_searches(self, mock_get_tenant_id):
        """Test getting recent searches."""
        mock_get_tenant_id.return_value = "tenant-123"
        
        response = self.client.get("/search-metrics/recent-searches?limit=10")
        
        assert response.status_code == 200
        data = response.json()
        assert data["tenant_id"] == "tenant-123"
        assert data["limit"] == 10
        assert "recent_searches" in data
    
    @patch('app.routers.search_metrics.get_tenant_id')
    def test_get_quality_metrics(self, mock_get_tenant_id):
        """Test getting quality metrics."""
        mock_get_tenant_id.return_value = "tenant-123"
        
        response = self.client.get("/search-metrics/quality")
        
        assert response.status_code == 200
        data = response.json()
        assert data["tenant_id"] == "tenant-123"
        assert "quality_metrics" in data


class TestSearchMetricsIntegration:
    """Test search metrics integration with RetrieverAdapter."""
    
    def test_metrics_collector_singleton(self):
        """Test that metrics collector is a singleton."""
        collector1 = get_search_metrics_collector()
        collector2 = get_search_metrics_collector()
        
        assert collector1 is collector2
    
    def test_metrics_collector_configuration(self):
        """Test metrics collector configuration."""
        collector = SearchMetricsCollector(max_metrics_history=500)
        
        assert collector.max_metrics_history == 500
        assert len(collector.search_metrics) == 0
        assert len(collector.quality_metrics) == 0
