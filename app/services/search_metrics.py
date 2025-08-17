"""Search performance metrics and quality monitoring service."""

import time
import structlog
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import statistics
from collections import defaultdict, deque

logger = structlog.get_logger(__name__)


@dataclass
class SearchMetrics:
    """Individual search operation metrics."""
    query_id: str
    tenant_id: str
    query: str
    timestamp: datetime
    total_time_ms: float
    bm25_time_ms: float
    vector_time_ms: float
    fusion_time_ms: float
    bm25_results: int
    vector_results: int
    fused_results: int
    final_results: int
    k_requested: int
    search_type: str  # 'hybrid', 'bm25_only', 'vector_only', 'fallback'
    success: bool
    error_message: Optional[str] = None


@dataclass
class SearchQualityMetrics:
    """Search quality and relevance metrics."""
    query_id: str
    tenant_id: str
    timestamp: datetime
    hit_at_k: Dict[int, float]  # hit@1, hit@3, hit@5, hit@10
    mean_reciprocal_rank: float
    normalized_discounted_cumulative_gain: float
    precision_at_k: Dict[int, float]
    recall_at_k: Dict[int, float]
    f1_at_k: Dict[int, float]
    relevance_scores: List[float]
    citation_coverage: float  # percentage of results with citations


@dataclass
class SearchPerformanceSummary:
    """Aggregated search performance metrics."""
    tenant_id: str
    time_window: str  # '1h', '24h', '7d', '30d'
    total_searches: int
    successful_searches: int
    failed_searches: int
    avg_total_time_ms: float
    p95_total_time_ms: float
    p99_total_time_ms: float
    avg_bm25_time_ms: float
    avg_vector_time_ms: float
    avg_fusion_time_ms: float
    avg_hit_at_1: float
    avg_hit_at_3: float
    avg_hit_at_5: float
    avg_hit_at_10: float
    avg_mrr: float
    avg_ndcg: float
    search_type_distribution: Dict[str, int]
    error_distribution: Dict[str, int]


class SearchMetricsCollector:
    """Collects and aggregates search performance metrics."""
    
    def __init__(self, max_metrics_history: int = 10000):
        self.max_metrics_history = max_metrics_history
        self.search_metrics: deque = deque(maxlen=max_metrics_history)
        self.quality_metrics: deque = deque(maxlen=max_metrics_history)
        self.tenant_metrics: Dict[str, Dict[str, deque]] = defaultdict(
            lambda: {
                'searches': deque(maxlen=max_metrics_history),
                'quality': deque(maxlen=max_metrics_history)
            }
        )
    
    def record_search_metrics(self, metrics: SearchMetrics):
        """Record search operation metrics."""
        try:
            # Store in global history
            self.search_metrics.append(metrics)
            
            # Store in tenant-specific history
            tenant_data = self.tenant_metrics[metrics.tenant_id]
            tenant_data['searches'].append(metrics)
            
            logger.info("Search metrics recorded", 
                       query_id=metrics.query_id,
                       tenant_id=metrics.tenant_id,
                       total_time_ms=metrics.total_time_ms,
                       search_type=metrics.search_type)
            
        except Exception as e:
            logger.error("Failed to record search metrics", error=str(e))
    
    def record_quality_metrics(self, metrics: SearchQualityMetrics):
        """Record search quality metrics."""
        try:
            # Store in global history
            self.quality_metrics.append(metrics)
            
            # Store in tenant-specific history
            tenant_data = self.tenant_metrics[metrics.tenant_id]
            tenant_data['quality'].append(metrics)
            
            logger.info("Quality metrics recorded", 
                       query_id=metrics.query_id,
                       tenant_id=metrics.tenant_id,
                       hit_at_1=metrics.hit_at_k.get(1, 0.0),
                       mrr=metrics.mean_reciprocal_rank)
            
        except Exception as e:
            logger.error("Failed to record quality metrics", error=str(e))
    
    def calculate_hit_at_k(self, relevance_scores: List[float], k: int) -> float:
        """Calculate hit@k metric."""
        if not relevance_scores or k <= 0:
            return 0.0
        
        # Consider a result relevant if score > 0.5
        relevant_count = sum(1 for score in relevance_scores[:k] if score > 0.5)
        return relevant_count / min(k, len(relevance_scores))
    
    def calculate_mrr(self, relevance_scores: List[float]) -> float:
        """Calculate Mean Reciprocal Rank."""
        if not relevance_scores:
            return 0.0
        
        for i, score in enumerate(relevance_scores):
            if score > 0.5:  # Consider relevant
                return 1.0 / (i + 1)
        
        return 0.0
    
    def calculate_ndcg(self, relevance_scores: List[float], k: int) -> float:
        """Calculate Normalized Discounted Cumulative Gain."""
        if not relevance_scores or k <= 0:
            return 0.0
        
        # DCG calculation
        dcg = 0.0
        for i, score in enumerate(relevance_scores[:k]):
            dcg += score / (1 + i)  # Discount factor
        
        # IDCG calculation (ideal ordering)
        ideal_scores = sorted(relevance_scores[:k], reverse=True)
        idcg = 0.0
        for i, score in enumerate(ideal_scores):
            idcg += score / (1 + i)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def calculate_precision_recall_f1(self, relevance_scores: List[float], k: int) -> Tuple[float, float, float]:
        """Calculate precision, recall, and F1 at k."""
        if not relevance_scores or k <= 0:
            return 0.0, 0.0, 0.0
        
        relevant_count = sum(1 for score in relevance_scores[:k] if score > 0.5)
        total_relevant = sum(1 for score in relevance_scores if score > 0.5)
        
        precision = relevant_count / k if k > 0 else 0.0
        recall = relevant_count / total_relevant if total_relevant > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return precision, recall, f1
    
    def generate_quality_metrics(self, 
                                query_id: str,
                                tenant_id: str,
                                relevance_scores: List[float],
                                citation_coverage: float) -> SearchQualityMetrics:
        """Generate quality metrics from search results."""
        try:
            # Calculate hit@k for common k values
            hit_at_k = {
                1: self.calculate_hit_at_k(relevance_scores, 1),
                3: self.calculate_hit_at_k(relevance_scores, 3),
                5: self.calculate_hit_at_k(relevance_scores, 5),
                10: self.calculate_hit_at_k(relevance_scores, 10)
            }
            
            # Calculate other metrics
            mrr = self.calculate_mrr(relevance_scores)
            ndcg = self.calculate_ndcg(relevance_scores, min(10, len(relevance_scores)))
            
            # Calculate precision/recall/F1 for common k values
            precision_at_k = {}
            recall_at_k = {}
            f1_at_k = {}
            
            for k in [1, 3, 5, 10]:
                if k <= len(relevance_scores):
                    precision, recall, f1 = self.calculate_precision_recall_f1(relevance_scores, k)
                    precision_at_k[k] = precision
                    recall_at_k[k] = recall
                    f1_at_k[k] = f1
            
            return SearchQualityMetrics(
                query_id=query_id,
                tenant_id=tenant_id,
                timestamp=datetime.now(),
                hit_at_k=hit_at_k,
                mean_reciprocal_rank=mrr,
                normalized_discounted_cumulative_gain=ndcg,
                precision_at_k=precision_at_k,
                recall_at_k=recall_at_k,
                f1_at_k=f1_at_k,
                relevance_scores=relevance_scores,
                citation_coverage=citation_coverage
            )
            
        except Exception as e:
            logger.error("Failed to generate quality metrics", error=str(e))
            # Return default metrics on error
            return SearchQualityMetrics(
                query_id=query_id,
                tenant_id=tenant_id,
                timestamp=datetime.now(),
                hit_at_k={1: 0.0, 3: 0.0, 5: 0.0, 10: 0.0},
                mean_reciprocal_rank=0.0,
                normalized_discounted_cumulative_gain=0.0,
                precision_at_k={},
                recall_at_k={},
                f1_at_k={},
                relevance_scores=[],
                citation_coverage=0.0
            )
    
    def get_performance_summary(self, 
                               tenant_id: str, 
                               time_window: str = '24h') -> SearchPerformanceSummary:
        """Get aggregated performance summary for a tenant."""
        try:
            # Calculate time threshold
            now = datetime.now()
            if time_window == '1h':
                threshold = now - timedelta(hours=1)
            elif time_window == '24h':
                threshold = now - timedelta(days=1)
            elif time_window == '7d':
                threshold = now - timedelta(days=7)
            elif time_window == '30d':
                threshold = now - timedelta(days=30)
            else:
                threshold = now - timedelta(days=1)  # Default to 24h
            
            # Filter metrics by time
            tenant_data = self.tenant_metrics.get(tenant_id, {})
            searches = [m for m in tenant_data.get('searches', []) if m.timestamp >= threshold]
            quality_metrics = [m for m in tenant_data.get('quality', []) if m.timestamp >= threshold]
            
            if not searches:
                return self._empty_performance_summary(tenant_id, time_window)
            
            # Calculate basic metrics
            total_searches = len(searches)
            successful_searches = sum(1 for m in searches if m.success)
            failed_searches = total_searches - successful_searches
            
            # Calculate timing metrics
            total_times = [m.total_time_ms for m in searches if m.success]
            bm25_times = [m.bm25_time_ms for m in searches if m.success and m.bm25_time_ms > 0]
            vector_times = [m.vector_time_ms for m in searches if m.success and m.vector_time_ms > 0]
            fusion_times = [m.fusion_time_ms for m in searches if m.success and m.fusion_time_ms > 0]
            
            avg_total_time = statistics.mean(total_times) if total_times else 0.0
            p95_total_time = statistics.quantiles(total_times, n=20)[18] if len(total_times) >= 20 else 0.0
            p99_total_time = statistics.quantiles(total_times, n=100)[98] if len(total_times) >= 100 else 0.0
            
            avg_bm25_time = statistics.mean(bm25_times) if bm25_times else 0.0
            avg_vector_time = statistics.mean(vector_times) if vector_times else 0.0
            avg_fusion_time = statistics.mean(fusion_times) if fusion_times else 0.0
            
            # Calculate quality metrics
            if quality_metrics:
                avg_hit_at_1 = statistics.mean([m.hit_at_k.get(1, 0.0) for m in quality_metrics])
                avg_hit_at_3 = statistics.mean([m.hit_at_k.get(3, 0.0) for m in quality_metrics])
                avg_hit_at_5 = statistics.mean([m.hit_at_k.get(5, 0.0) for m in quality_metrics])
                avg_hit_at_10 = statistics.mean([m.hit_at_k.get(10, 0.0) for m in quality_metrics])
                avg_mrr = statistics.mean([m.mean_reciprocal_rank for m in quality_metrics])
                avg_ndcg = statistics.mean([m.normalized_discounted_cumulative_gain for m in quality_metrics])
            else:
                avg_hit_at_1 = avg_hit_at_3 = avg_hit_at_5 = avg_hit_at_10 = 0.0
                avg_mrr = avg_ndcg = 0.0
            
            # Calculate distributions
            search_type_distribution = defaultdict(int)
            error_distribution = defaultdict(int)
            
            for search in searches:
                search_type_distribution[search.search_type] += 1
                if not search.success and search.error_message:
                    error_distribution[search.error_message] += 1
            
            return SearchPerformanceSummary(
                tenant_id=tenant_id,
                time_window=time_window,
                total_searches=total_searches,
                successful_searches=successful_searches,
                failed_searches=failed_searches,
                avg_total_time_ms=round(avg_total_time, 2),
                p95_total_time_ms=round(p95_total_time, 2),
                p99_total_time_ms=round(p99_total_time, 2),
                avg_bm25_time_ms=round(avg_bm25_time, 2),
                avg_vector_time_ms=round(avg_vector_time, 2),
                avg_fusion_time_ms=round(avg_fusion_time, 2),
                avg_hit_at_1=round(avg_hit_at_1, 3),
                avg_hit_at_3=round(avg_hit_at_3, 3),
                avg_hit_at_5=round(avg_hit_at_5, 3),
                avg_hit_at_10=round(avg_hit_at_10, 3),
                avg_mrr=round(avg_mrr, 3),
                avg_ndcg=round(avg_ndcg, 3),
                search_type_distribution=dict(search_type_distribution),
                error_distribution=dict(error_distribution)
            )
            
        except Exception as e:
            logger.error("Failed to generate performance summary", error=str(e))
            return self._empty_performance_summary(tenant_id, time_window)
    
    def _empty_performance_summary(self, tenant_id: str, time_window: str) -> SearchPerformanceSummary:
        """Return empty performance summary when no data is available."""
        return SearchPerformanceSummary(
            tenant_id=tenant_id,
            time_window=time_window,
            total_searches=0,
            successful_searches=0,
            failed_searches=0,
            avg_total_time_ms=0.0,
            p95_total_time_ms=0.0,
            p99_total_time_ms=0.0,
            avg_bm25_time_ms=0.0,
            avg_vector_time_ms=0.0,
            avg_fusion_time_ms=0.0,
            avg_hit_at_1=0.0,
            avg_hit_at_3=0.0,
            avg_hit_at_5=0.0,
            avg_hit_at_10=0.0,
            avg_mrr=0.0,
            avg_ndcg=0.0,
            search_type_distribution={},
            error_distribution={}
        )
    
    def get_global_metrics(self, time_window: str = '24h') -> Dict[str, Any]:
        """Get global metrics across all tenants."""
        try:
            all_tenants = list(self.tenant_metrics.keys())
            if not all_tenants:
                return {}
            
            # Aggregate metrics across all tenants
            global_summary = {
                'time_window': time_window,
                'total_tenants': len(all_tenants),
                'total_searches': 0,
                'successful_searches': 0,
                'failed_searches': 0,
                'avg_total_time_ms': 0.0,
                'avg_hit_at_1': 0.0,
                'avg_hit_at_3': 0.0,
                'avg_hit_at_5': 0.0,
                'avg_hit_at_10': 0.0,
                'avg_mrr': 0.0,
                'avg_ndcg': 0.0,
                'search_type_distribution': defaultdict(int),
                'error_distribution': defaultdict(int)
            }
            
            tenant_summaries = []
            for tenant_id in all_tenants:
                summary = self.get_performance_summary(tenant_id, time_window)
                tenant_summaries.append(summary)
                
                # Aggregate metrics
                global_summary['total_searches'] += summary.total_searches
                global_summary['successful_searches'] += summary.successful_searches
                global_summary['failed_searches'] += summary.failed_searches
                
                # Aggregate search type distribution
                for search_type, count in summary.search_type_distribution.items():
                    global_summary['search_type_distribution'][search_type] += count
                
                # Aggregate error distribution
                for error, count in summary.error_distribution.items():
                    global_summary['error_distribution'][error] += count
            
            # Calculate averages
            if tenant_summaries:
                global_summary['avg_total_time_ms'] = round(
                    statistics.mean([s.avg_total_time_ms for s in tenant_summaries]), 2
                )
                global_summary['avg_hit_at_1'] = round(
                    statistics.mean([s.avg_hit_at_1 for s in tenant_summaries]), 3
                )
                global_summary['avg_hit_at_3'] = round(
                    statistics.mean([s.avg_hit_at_3 for s in tenant_summaries]), 3
                )
                global_summary['avg_hit_at_5'] = round(
                    statistics.mean([s.avg_hit_at_5 for s in tenant_summaries]), 3
                )
                global_summary['avg_hit_at_10'] = round(
                    statistics.mean([s.avg_hit_at_10 for s in tenant_summaries]), 3
                )
                global_summary['avg_mrr'] = round(
                    statistics.mean([s.avg_mrr for s in tenant_summaries]), 3
                )
                global_summary['avg_ndcg'] = round(
                    statistics.mean([s.avg_ndcg for s in tenant_summaries]), 3
                )
            
            # Convert defaultdict to regular dict for JSON serialization
            global_summary['search_type_distribution'] = dict(global_summary['search_type_distribution'])
            global_summary['error_distribution'] = dict(global_summary['error_distribution'])
            
            return global_summary
            
        except Exception as e:
            logger.error("Failed to generate global metrics", error=str(e))
            return {}
    
    def export_metrics(self, tenant_id: Optional[str] = None, 
                       time_window: str = '24h') -> Dict[str, Any]:
        """Export metrics for external systems."""
        try:
            if tenant_id:
                # Export tenant-specific metrics
                summary = self.get_performance_summary(tenant_id, time_window)
                return {
                    'tenant_id': tenant_id,
                    'time_window': time_window,
                    'timestamp': datetime.now().isoformat(),
                    'summary': summary.__dict__,
                    'recent_searches': [
                        {
                            'query_id': m.query_id,
                            'query': m.query[:100] if len(m.query) > 100 else m.query,  # Truncate long queries
                            'timestamp': m.timestamp.isoformat(),
                            'total_time_ms': m.total_time_ms,
                            'search_type': m.search_type,
                            'success': m.success
                        }
                        for m in list(self.tenant_metrics.get(tenant_id, {}).get('searches', []))[-100:]
                    ]
                }
            else:
                # Export global metrics
                return {
                    'time_window': time_window,
                    'timestamp': datetime.now().isoformat(),
                    'global_summary': self.get_global_metrics(time_window),
                    'tenant_summaries': {
                        tid: self.get_performance_summary(tid, time_window).__dict__
                        for tid in self.tenant_metrics.keys()
                    }
                }
                
        except Exception as e:
            logger.error("Failed to export metrics", error=str(e))
            return {'error': str(e)}
    
    def reset_metrics(self, tenant_id: Optional[str] = None):
        """Reset metrics for a specific tenant or all tenants."""
        try:
            if tenant_id:
                # Reset specific tenant
                if tenant_id in self.tenant_metrics:
                    self.tenant_metrics[tenant_id] = {
                        'searches': deque(maxlen=self.max_metrics_history),
                        'quality': deque(maxlen=self.max_metrics_history)
                    }
                    logger.info("Metrics reset for tenant", tenant_id=tenant_id)
            else:
                # Reset all metrics
                self.search_metrics.clear()
                self.quality_metrics.clear()
                self.tenant_metrics.clear()
                logger.info("All metrics reset")
                
        except Exception as e:
            logger.error("Failed to reset metrics", error=str(e))


# Global metrics collector instance
search_metrics_collector = SearchMetricsCollector()


def get_search_metrics_collector() -> SearchMetricsCollector:
    """Get the global search metrics collector instance."""
    return search_metrics_collector
