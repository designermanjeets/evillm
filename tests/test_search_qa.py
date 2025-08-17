"""Tests for search quality assurance functionality."""

import pytest
import json
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime
from pathlib import Path
from fastapi.testclient import TestClient

from app.services.search_qa import (
    SearchQualityAssurance, 
    QualityThresholds, 
    SearchQualityResult,
    BenchmarkResult,
    get_search_qa
)
from app.main import create_app


class TestQualityThresholds:
    """Test quality thresholds configuration."""
    
    def test_default_thresholds(self):
        """Test default threshold values."""
        thresholds = QualityThresholds()
        
        assert thresholds.hit_at_5 == 0.65
        assert thresholds.mrr_at_10 == 0.55
        assert thresholds.ndcg_at_10 == 0.60
    
    def test_custom_thresholds(self):
        """Test custom threshold values."""
        thresholds = QualityThresholds(
            hit_at_5=0.80,
            mrr_at_10=0.70,
            ndcg_at_10=0.75
        )
        
        assert thresholds.hit_at_5 == 0.80
        assert thresholds.mrr_at_10 == 0.70
        assert thresholds.ndcg_at_10 == 0.75


class TestSearchQualityAssurance:
    """Test search quality assurance functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            "quality_thresholds_hit_at_5": 0.65,
            "quality_thresholds_mrr_at_10": 0.55,
            "quality_thresholds_ndcg_at_10": 0.60
        }
        self.qa_service = SearchQualityAssurance(self.config)
    
    def test_validate_query_empty(self):
        """Test validation of empty queries."""
        is_valid, warnings = self.qa_service.validate_query("", "tenant-123")
        
        assert not is_valid
        assert "Query cannot be empty" in warnings
    
    def test_validate_query_whitespace(self):
        """Test validation of whitespace-only queries."""
        is_valid, warnings = self.qa_service.validate_query("   ", "tenant-123")
        
        assert not is_valid
        assert "Query cannot be empty" in warnings
    
    def test_validate_query_wildcard_only(self):
        """Test validation of wildcard-only queries."""
        for wildcard in ["*", "?", ".*", ".*.*"]:
            is_valid, warnings = self.qa_service.validate_query(wildcard, "tenant-123")
            
            assert not is_valid
            assert "Wildcard-only queries are not allowed" in warnings
    
    def test_validate_query_dangerous_patterns(self):
        """Test validation of dangerous query patterns."""
        dangerous_queries = [
            "javascript:alert('xss')",
            "data:text/html,<script>alert('xss')</script>",
            "vbscript:msgbox('xss')",
            "<script>alert('xss')</script>",
            "eval(malicious_code)",
            "union select * from users",
            "drop table users",
            "delete from users"
        ]
        
        for query in dangerous_queries:
            is_valid, warnings = self.qa_service.validate_query(query, "tenant-123")
            
            assert not is_valid
            assert any("dangerous pattern" in warning.lower() for warning in warnings)
    
    def test_validate_query_too_long(self):
        """Test validation of overly long queries."""
        long_query = "a" * 501
        is_valid, warnings = self.qa_service.validate_query(long_query, "tenant-123")
        
        assert is_valid  # Should still be valid but with warning
        assert any("exceeds recommended limit" in warning for warning in warnings)
    
    def test_validate_query_very_short(self):
        """Test validation of very short queries."""
        short_query = "a"
        is_valid, warnings = self.qa_service.validate_query(short_query, "tenant-123")
        
        assert is_valid  # Should be valid but with warning
        assert any("very short queries" in warning.lower() for warning in warnings)
    
    def test_validate_query_generic_terms(self):
        """Test validation of queries with only generic terms."""
        generic_query = "the and or but in on at to for"
        is_valid, warnings = self.qa_service.validate_query(generic_query, "tenant-123")
        
        assert is_valid  # Should be valid but with warning
        assert any("generic terms" in warning.lower() for warning in warnings)
    
    def test_validate_query_good(self):
        """Test validation of good queries."""
        good_queries = [
            "project timeline update",
            "email attachment processing",
            "customer support request",
            "invoice payment status"
        ]
        
        for query in good_queries:
            is_valid, warnings = self.qa_service.validate_query(query, "tenant-123")
            
            assert is_valid
            assert len(warnings) == 0
    
    def test_assess_search_quality_pass(self):
        """Test quality assessment that passes all thresholds."""
        quality_metrics = {
            "hit_at_k": {5: 0.8},
            "mean_reciprocal_rank": 0.7,
            "normalized_discounted_cumulative_gain": 0.75
        }
        
        result = self.qa_service.assess_search_quality(
            "test query", "tenant-123", quality_metrics
        )
        
        assert result.overall_pass
        assert result.thresholds_met["hit_at_5"]
        assert result.thresholds_met["mrr_at_10"]
        assert result.thresholds_met["ndcg_at_10"]
        assert "no action needed" in result.recommendations[0]
    
    def test_assess_search_quality_fail_grounding(self):
        """Test quality assessment that fails grounding threshold."""
        quality_metrics = {
            "hit_at_k": {5: 0.5},  # Below 0.65 threshold
            "mean_reciprocal_rank": 0.7,
            "normalized_discounted_cumulative_gain": 0.75
        }
        
        result = self.qa_service.assess_search_quality(
            "test query", "tenant-123", quality_metrics
        )
        
        assert not result.overall_pass
        assert not result.thresholds_met["hit_at_5"]
        assert result.thresholds_met["mrr_at_10"]
        assert result.thresholds_met["ndcg_at_10"]
        assert "below threshold" in result.warnings[0]
        assert "refining query" in result.recommendations[0]
    
    def test_assess_search_quality_fail_mrr(self):
        """Test quality assessment that fails MRR threshold."""
        quality_metrics = {
            "hit_at_k": {5: 0.8},
            "mean_reciprocal_rank": 0.4,  # Below 0.55 threshold
            "normalized_discounted_cumulative_gain": 0.75
        }
        
        result = self.qa_service.assess_search_quality(
            "test query", "tenant-123", quality_metrics
        )
        
        assert not result.overall_pass
        assert result.thresholds_met["hit_at_5"]
        assert not result.thresholds_met["mrr_at_10"]
        assert result.thresholds_met["ndcg_at_10"]
        assert "ranking algorithms" in result.recommendations[0] or "document ranking" in result.recommendations[0]
    
    def test_assess_search_quality_fail_ndcg(self):
        """Test quality assessment that fails NDCG threshold."""
        quality_metrics = {
            "hit_at_k": {5: 0.8},
            "mean_reciprocal_rank": 0.7,
            "normalized_discounted_cumulative_gain": 0.4  # Below 0.60 threshold
        }
        
        result = self.qa_service.assess_search_quality(
            "test query", "tenant-123", quality_metrics
        )
        
        assert not result.overall_pass
        assert result.thresholds_met["hit_at_5"]
        assert result.thresholds_met["mrr_at_10"]
        assert not result.thresholds_met["ndcg_at_10"]
        assert "document quality" in result.recommendations[0]
    
    def test_assess_search_quality_missing_metrics(self):
        """Test quality assessment with missing metrics."""
        quality_metrics = {}
        
        result = self.qa_service.assess_search_quality(
            "test query", "tenant-123", quality_metrics
        )
        
        assert not result.overall_pass
        assert not any(result.thresholds_met.values())
        assert "assessment failed" in result.warnings[0] or "below threshold" in result.warnings[0]
    
    def test_get_quality_summary(self):
        """Test quality summary generation."""
        summary = self.qa_service.get_quality_summary("tenant-123", "24h")
        
        assert summary["tenant_id"] == "tenant-123"
        assert summary["time_window"] == "24h"
        assert "thresholds" in summary
        assert "compliance" in summary
        assert "status" in summary
    
    def test_get_recent_benchmarks_empty(self):
        """Test getting recent benchmarks when none exist."""
        # Reset benchmarks to ensure clean state
        self.qa_service.reset_benchmarks("tenant-123")
        
        benchmarks = self.qa_service.get_recent_benchmarks("tenant-123", 10)
        
        assert isinstance(benchmarks, list)
        assert len(benchmarks) == 0
    
    def test_reset_benchmarks_tenant(self):
        """Test resetting benchmarks for a specific tenant."""
        # This test verifies the method doesn't crash
        self.qa_service.reset_benchmarks("tenant-123")
        # No assertion needed - just checking it doesn't raise an exception


class TestSearchQABenchmark:
    """Test search QA benchmarking functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            "quality_thresholds_hit_at_5": 0.65,
            "quality_thresholds_mrr_at_10": 0.55,
            "quality_thresholds_ndcg_at_10": 0.60
        }
        self.qa_service = SearchQualityAssurance(self.config)
    
    @pytest.mark.asyncio
    async def test_run_benchmark_success(self):
        """Test successful benchmark execution."""
        # Mock retriever
        mock_retriever = Mock()
        mock_retriever.retrieve = AsyncMock(return_value=[
            Mock(score=0.9),
            Mock(score=0.8),
            Mock(score=0.7)
        ])
        
        # Mock metrics collector
        with patch.object(self.qa_service, 'metrics_collector') as mock_collector:
            mock_metrics = Mock()
            mock_metrics.hit_at_k = {5: 0.8}
            mock_metrics.mean_reciprocal_rank = 0.7
            mock_metrics.normalized_discounted_cumulative_gain = 0.75
            mock_collector.generate_quality_metrics.return_value = mock_metrics
            
            queries = ["query 1", "query 2", "query 3"]
            result = await self.qa_service.run_benchmark("tenant-123", queries, mock_retriever)
            
            assert result.benchmark_id.startswith("benchmark_tenant-123_")
            assert result.tenant_id == "tenant-123"
            assert result.total_queries == 3
            assert result.passed_queries == 3
            assert result.failed_queries == 0
            assert "All quality thresholds met" in result.recommendations[0]
    
    @pytest.mark.asyncio
    async def test_run_benchmark_with_failures(self):
        """Test benchmark execution with some query failures."""
        # Mock retriever that fails on some queries
        mock_retriever = Mock()
        mock_retriever.retrieve = AsyncMock(side_effect=[
            [Mock(score=0.9)],  # Success
            Exception("Search failed"),  # Failure
            [Mock(score=0.4)]  # Below threshold
        ])
        
        # Mock metrics collector
        with patch.object(self.qa_service, 'metrics_collector') as mock_collector:
            mock_metrics = Mock()
            mock_metrics.hit_at_k = {5: 0.4}  # Below threshold
            mock_metrics.mean_reciprocal_rank = 0.3
            mock_metrics.normalized_discounted_cumulative_gain = 0.4
            mock_collector.generate_quality_metrics.return_value = mock_metrics
            
            queries = ["query 1", "query 2", "query 3"]
            result = await self.qa_service.run_benchmark("tenant-123", queries, mock_retriever)
            
            assert result.total_queries == 3
            assert result.passed_queries == 0
            assert result.failed_queries == 1  # Only the exception counts as failed
            assert "below threshold" in result.recommendations[0]
    
    @pytest.mark.asyncio
    async def test_run_benchmark_empty_queries(self):
        """Test benchmark execution with empty query list."""
        mock_retriever = Mock()
        
        result = await self.qa_service.run_benchmark("tenant-123", [], mock_retriever)
        
        assert result.total_queries == 0
        assert result.passed_queries == 0
        assert result.failed_queries == 0
        assert "All quality thresholds met" in result.recommendations[0] or "below threshold" in result.recommendations[0]


class TestSearchQAAPI:
    """Test search QA API endpoints."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.app = create_app()
        self.client = TestClient(self.app)
    
    @patch('app.routers.search_qa.get_tenant_id')
    def test_health_check(self, mock_get_tenant_id):
        """Test search QA health check endpoint."""
        response = self.client.get("/search-qa/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "search-qa"
        assert "thresholds" in data
    
    @patch('app.routers.search_qa.get_tenant_id')
    def test_get_quality_summary(self, mock_get_tenant_id):
        """Test getting quality summary."""
        mock_get_tenant_id.return_value = "tenant-123"
        
        response = self.client.get("/search-qa/quality/summary", headers={"X-Tenant-ID": "123e4567-e89b-12d3-a456-426614174000"})
        
        assert response.status_code == 200
        data = response.json()
        assert data["tenant_id"] == "tenant-123"
        assert "summary" in data
    
    @patch('app.routers.search_qa.get_tenant_id')
    def test_validate_query(self, mock_get_tenant_id):
        """Test query validation endpoint."""
        mock_get_tenant_id.return_value = "tenant-123"
        
        query_data = {
            "query": "project timeline update",
            "tenant_id": "tenant-123"
        }
        
        response = self.client.post("/search-qa/validate-query", json=query_data, headers={"X-Tenant-ID": "123e4567-e89b-12d3-a456-426614174000"})
        
        assert response.status_code == 200
        data = response.json()
        assert data["is_valid"]
        assert data["tenant_id"] == "tenant-123"
    
    @patch('app.routers.search_qa.get_tenant_id')
    def test_validate_query_dangerous(self, mock_get_tenant_id):
        """Test query validation with dangerous patterns."""
        mock_get_tenant_id.return_value = "tenant-123"
        
        query_data = {
            "query": "javascript:alert('xss')",
            "tenant_id": "tenant-123"
        }
        
        response = self.client.post("/search-qa/validate-query", json=query_data, headers={"X-Tenant-ID": "123e4567-e89b-12d3-a456-426614174000"})
        
        assert response.status_code == 200
        data = response.json()
        assert not data["is_valid"]
        assert any("dangerous pattern" in warning.lower() for warning in data["warnings"])
    
    @patch('app.routers.search_qa.get_tenant_id')
    def test_get_quality_thresholds(self, mock_get_tenant_id):
        """Test getting quality thresholds."""
        mock_get_tenant_id.return_value = "tenant-123"
        
        response = self.client.get("/search-qa/thresholds", headers={"X-Tenant-ID": "123e4567-e89b-12d3-a456-426614174000"})
        
        assert response.status_code == 200
        data = response.json()
        assert data["tenant_id"] == "tenant-123"
        assert "thresholds" in data
        assert "description" in data
    
    @patch('app.routers.search_qa.get_tenant_id')
    def test_get_qa_service_status(self, mock_get_tenant_id):
        """Test getting QA service status."""
        mock_get_tenant_id.return_value = "tenant-123"
        
        response = self.client.get("/search-qa/status", headers={"X-Tenant-ID": "123e4567-e89b-12d3-a456-426614174000"})
        
        assert response.status_code == 200
        data = response.json()
        assert data["tenant_id"] == "tenant-123"
        assert "overall_health" in data
        assert "quality_status" in data
        assert "thresholds" in data


class TestSearchQAIntegration:
    """Test search QA integration with other services."""
    
    def test_qa_service_singleton(self):
        """Test that QA service is a singleton."""
        qa1 = get_search_qa()
        qa2 = get_search_qa()
        
        assert qa1 is qa2
    
    def test_qa_service_configuration(self):
        """Test QA service configuration."""
        config = {
            "quality_thresholds_hit_at_5": 0.80,
            "quality_thresholds_mrr_at_10": 0.70,
            "quality_thresholds_ndcg_at_10": 0.75
        }
        
        qa_service = SearchQualityAssurance(config)
        
        assert qa_service.thresholds.hit_at_5 == 0.80
        assert qa_service.thresholds.mrr_at_10 == 0.70
        assert qa_service.thresholds.ndcg_at_10 == 0.75
    
    def test_qa_service_default_config(self):
        """Test QA service with default configuration."""
        qa_service = SearchQualityAssurance({})
        
        assert qa_service.thresholds.hit_at_5 == 0.65
        assert qa_service.thresholds.mrr_at_10 == 0.55
        assert qa_service.thresholds.ndcg_at_10 == 0.60
