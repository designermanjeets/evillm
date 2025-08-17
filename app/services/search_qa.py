"""Search Quality Assurance service with threshold enforcement and benchmarking."""

import json
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import structlog
from pathlib import Path

from app.services.search_metrics import get_search_metrics_collector
from app.services.retriever import RetrieverAdapter

logger = structlog.get_logger(__name__)


@dataclass
class QualityThresholds:
    """Quality thresholds for search performance."""
    hit_at_5: float = 0.65
    mrr_at_10: float = 0.55
    ndcg_at_10: float = 0.60


@dataclass
class SearchQualityResult:
    """Search quality assessment result."""
    query: str
    tenant_id: str
    timestamp: datetime
    quality_metrics: Dict[str, float]
    thresholds_met: Dict[str, bool]
    overall_pass: bool
    warnings: List[str]
    recommendations: List[str]


@dataclass
class BenchmarkResult:
    """Search benchmark result."""
    benchmark_id: str
    tenant_id: str
    timestamp: datetime
    total_queries: int
    passed_queries: int
    failed_queries: int
    quality_summary: Dict[str, float]
    threshold_compliance: Dict[str, float]
    recommendations: List[str]


class SearchQualityAssurance:
    """Search quality assurance with threshold enforcement and benchmarking."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.thresholds = QualityThresholds(
            hit_at_5=config.get("quality_thresholds_hit_at_5", 0.65),
            mrr_at_10=config.get("quality_thresholds_mrr_at_10", 0.55),
            ndcg_at_10=config.get("quality_thresholds_ndcg_at_10", 0.60)
        )
        self.metrics_collector = get_search_metrics_collector()
        self.benchmark_dir = Path("/tmp/search_benchmarks")
        self.benchmark_dir.mkdir(exist_ok=True)
    
    def validate_query(self, query: str, tenant_id: str) -> Tuple[bool, List[str]]:
        """Validate search query for quality and security."""
        warnings = []
        
        # Security: Check for empty or malicious queries
        if not query or not query.strip():
            return False, ["Query cannot be empty"]
        
        query = query.strip()
        
        # Security: Check query length
        if len(query) > 500:
            warnings.append(f"Query length {len(query)} exceeds recommended limit of 500 characters")
        
        # Security: Check for wildcard-only queries
        if query.strip() in ["*", "?", ".*", ".*.*"]:
            return False, ["Wildcard-only queries are not allowed"]
        
        # Security: Check for potentially dangerous patterns
        dangerous_patterns = [
            "javascript:", "data:", "vbscript:", "<script>", "eval(",
            "union select", "drop table", "delete from"
        ]
        
        for pattern in dangerous_patterns:
            if pattern.lower() in query.lower():
                return False, [f"Query contains potentially dangerous pattern: {pattern}"]
        
        # Quality: Check for very short queries
        if len(query.split()) < 2:
            warnings.append("Very short queries may return low-quality results")
        
        # Quality: Check for overly generic terms
        generic_terms = ["the", "and", "or", "but", "in", "on", "at", "to", "for"]
        if all(word.lower() in generic_terms for word in query.split()):
            warnings.append("Query contains only generic terms - consider more specific language")
        
        return True, warnings
    
    def assess_search_quality(
        self, 
        query: str, 
        tenant_id: str, 
        quality_metrics: Dict[str, float]
    ) -> SearchQualityResult:
        """Assess search quality against thresholds."""
        try:
            # Extract metrics
            hit_at_5 = quality_metrics.get("hit_at_k", {}).get(5, 0.0)
            mrr = quality_metrics.get("mean_reciprocal_rank", 0.0)
            ndcg = quality_metrics.get("normalized_discounted_cumulative_gain", 0.0)
            
            # Check thresholds
            thresholds_met = {
                "hit_at_5": hit_at_5 >= self.thresholds.hit_at_5,
                "mrr_at_10": mrr >= self.thresholds.mrr_at_10,
                "ndcg_at_10": ndcg >= self.thresholds.ndcg_at_10
            }
            
            overall_pass = all(thresholds_met.values())
            
            # Generate warnings and recommendations
            warnings = []
            recommendations = []
            
            if not thresholds_met["hit_at_5"]:
                warnings.append(f"Hit@5 score {hit_at_5:.3f} below threshold {self.thresholds.hit_at_5}")
                recommendations.append("Consider refining query with more specific terms")
                recommendations.append("Check if relevant documents exist in the index")
            
            if not thresholds_met["mrr_at_10"]:
                warnings.append(f"MRR score {mrr:.3f} below threshold {self.thresholds.mrr_at_10}")
                recommendations.append("Review document ranking and relevance scoring")
                recommendations.append("Consider adjusting search algorithm weights")
            
            if not thresholds_met["ndcg_at_10"]:
                warnings.append(f"NDCG score {ndcg:.3f} below threshold {self.thresholds.ndcg_at_10}")
                recommendations.append("Evaluate document quality and relevance")
                recommendations.append("Consider improving document preprocessing")
            
            if overall_pass:
                recommendations.append("Search quality meets all thresholds - no action needed")
            
            return SearchQualityResult(
                query=query,
                tenant_id=tenant_id,
                timestamp=datetime.now(),
                quality_metrics={
                    "hit_at_5": hit_at_5,
                    "mrr": mrr,
                    "ndcg": ndcg
                },
                thresholds_met=thresholds_met,
                overall_pass=overall_pass,
                warnings=warnings,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error("Failed to assess search quality", error=str(e), tenant_id=tenant_id)
            return SearchQualityResult(
                query=query,
                tenant_id=tenant_id,
                timestamp=datetime.now(),
                quality_metrics={},
                thresholds_met={},
                overall_pass=False,
                warnings=[f"Quality assessment failed: {str(e)}"],
                recommendations=["Contact system administrator for assistance"]
            )
    
    async def run_benchmark(
        self, 
        tenant_id: str, 
        queries: List[str],
        retriever: RetrieverAdapter
    ) -> BenchmarkResult:
        """Run search quality benchmark on a set of queries."""
        try:
            benchmark_id = f"benchmark_{tenant_id}_{int(time.time())}"
            start_time = datetime.now()
            
            total_queries = len(queries)
            passed_queries = 0
            failed_queries = 0
            quality_scores = []
            
            logger.info("Starting search quality benchmark", 
                       benchmark_id=benchmark_id,
                       tenant_id=tenant_id,
                       total_queries=total_queries)
            
            for i, query in enumerate(queries):
                try:
                    # Validate query
                    is_valid, warnings = self.validate_query(query, tenant_id)
                    if not is_valid:
                        failed_queries += 1
                        logger.warning("Query validation failed", 
                                     query=query[:100], 
                                     warnings=warnings)
                        continue
                    
                    # Execute search
                    citations = await retriever.retrieve(tenant_id, query, k=10)
                    
                    # Get quality metrics
                    if citations:
                        relevance_scores = [c.score for c in citations]
                        citation_coverage = 1.0
                        
                        quality_metrics = self.metrics_collector.generate_quality_metrics(
                            f"benchmark_{i}", tenant_id, relevance_scores, citation_coverage
                        )
                        
                        # Convert metrics to dict format for assessment
                        metrics_dict = {}
                        if hasattr(quality_metrics, '__dict__'):
                            metrics_dict = quality_metrics.__dict__
                        else:
                            # Handle case where metrics is already a dict or has attributes
                            metrics_dict = {
                                'hit_at_k': getattr(quality_metrics, 'hit_at_k', {}),
                                'mean_reciprocal_rank': getattr(quality_metrics, 'mean_reciprocal_rank', 0.0),
                                'normalized_discounted_cumulative_gain': getattr(quality_metrics, 'normalized_discounted_cumulative_gain', 0.0)
                            }
                        
                        # Assess quality
                        quality_result = self.assess_search_quality(
                            query, tenant_id, metrics_dict
                        )
                        
                        if quality_result.overall_pass:
                            passed_queries += 1
                        
                        quality_scores.append(quality_result.quality_metrics)
                        
                        # Record metrics
                        self.metrics_collector.record_quality_metrics(quality_metrics)
                        
                    else:
                        failed_queries += 1
                        logger.warning("Search returned no results", query=query[:100])
                    
                except Exception as e:
                    failed_queries += 1
                    logger.error("Benchmark query failed", 
                               query=query[:100], 
                               error=str(e))
            
            # Calculate quality summary
            if quality_scores:
                avg_hit_at_5 = sum(s.get("hit_at_5", 0.0) for s in quality_scores) / len(quality_scores)
                avg_mrr = sum(s.get("mrr", 0.0) for s in quality_scores) / len(quality_scores)
                avg_ndcg = sum(s.get("ndcg", 0.0) for s in quality_scores) / len(quality_scores)
            else:
                avg_hit_at_5 = avg_mrr = avg_ndcg = 0.0
            
            quality_summary = {
                "avg_hit_at_5": round(avg_hit_at_5, 3),
                "avg_mrr": round(avg_mrr, 3),
                "avg_ndcg": round(avg_ndcg, 3)
            }
            
            # Calculate threshold compliance
            threshold_compliance = {
                "hit_at_5": round(avg_hit_at_5 / self.thresholds.hit_at_5, 3),
                "mrr_at_10": round(avg_mrr / self.thresholds.mrr_at_10, 3),
                "ndcg_at_10": round(avg_ndcg / self.thresholds.ndcg_at_10, 3)
            }
            
            # Generate recommendations
            recommendations = []
            if avg_hit_at_5 < self.thresholds.hit_at_5:
                recommendations.append("Hit@5 below threshold - review query optimization")
            if avg_mrr < self.thresholds.mrr_at_10:
                recommendations.append("MRR below threshold - improve ranking algorithms")
            if avg_ndcg < self.thresholds.ndcg_at_10:
                recommendations.append("NDCG below threshold - enhance document quality")
            
            if not recommendations:
                recommendations.append("All quality thresholds met - search system performing well")
            
            benchmark_result = BenchmarkResult(
                benchmark_id=benchmark_id,
                tenant_id=tenant_id,
                timestamp=start_time,
                total_queries=total_queries,
                passed_queries=passed_queries,
                failed_queries=failed_queries,
                quality_summary=quality_summary,
                threshold_compliance=threshold_compliance,
                recommendations=recommendations
            )
            
            # Save benchmark result
            self._save_benchmark_result(benchmark_result)
            
            logger.info("Search quality benchmark completed", 
                       benchmark_id=benchmark_id,
                       tenant_id=tenant_id,
                       passed_queries=passed_queries,
                       failed_queries=failed_queries,
                       overall_pass_rate=passed_queries/total_queries if total_queries > 0 else 0)
            
            return benchmark_result
            
        except Exception as e:
            logger.error("Benchmark execution failed", error=str(e), tenant_id=tenant_id)
            return BenchmarkResult(
                benchmark_id="failed",
                tenant_id=tenant_id,
                timestamp=datetime.now(),
                total_queries=0,
                passed_queries=0,
                failed_queries=0,
                quality_summary={},
                threshold_compliance={},
                recommendations=[f"Benchmark failed: {str(e)}"]
            )
    
    def _save_benchmark_result(self, result: BenchmarkResult) -> None:
        """Save benchmark result to file."""
        try:
            filename = f"{result.benchmark_id}.json"
            filepath = self.benchmark_dir / filename
            
            # Convert to dict for JSON serialization
            result_dict = {
                "benchmark_id": result.benchmark_id,
                "tenant_id": result.tenant_id,
                "timestamp": result.timestamp.isoformat(),
                "total_queries": result.total_queries,
                "passed_queries": result.passed_queries,
                "failed_queries": result.failed_queries,
                "quality_summary": result.quality_summary,
                "threshold_compliance": result.threshold_compliance,
                "recommendations": result.recommendations
            }
            
            with open(filepath, 'w') as f:
                json.dump(result_dict, f, indent=2)
            
            logger.info("Benchmark result saved", 
                       filepath=str(filepath),
                       benchmark_id=result.benchmark_id)
            
        except Exception as e:
            logger.error("Failed to save benchmark result", 
                        error=str(e),
                        benchmark_id=result.benchmark_id)
    
    def get_quality_summary(self, tenant_id: str, time_window: str = "24h") -> Dict[str, Any]:
        """Get search quality summary for a tenant."""
        try:
            # Get performance summary
            performance_summary = self.metrics_collector.get_performance_summary(tenant_id, time_window)
            
            # Extract quality metrics
            quality_summary = {
                "tenant_id": tenant_id,
                "time_window": time_window,
                "timestamp": datetime.now().isoformat(),
                "total_searches": performance_summary.total_searches,
                "quality_metrics": {
                    "avg_hit_at_1": performance_summary.avg_hit_at_1,
                    "avg_hit_at_3": performance_summary.avg_hit_at_3,
                    "avg_hit_at_5": performance_summary.avg_hit_at_5,
                    "avg_hit_at_10": performance_summary.avg_hit_at_10,
                    "avg_mrr": performance_summary.avg_mrr,
                    "avg_ndcg": performance_summary.avg_ndcg
                },
                "thresholds": {
                    "hit_at_5": self.thresholds.hit_at_5,
                    "mrr_at_10": self.thresholds.mrr_at_10,
                    "ndcg_at_10": self.thresholds.ndcg_at_10
                },
                "compliance": {
                    "hit_at_5": performance_summary.avg_hit_at_5 >= self.thresholds.hit_at_5,
                    "mrr_at_10": performance_summary.avg_mrr >= self.thresholds.mrr_at_10,
                    "ndcg_at_10": performance_summary.avg_ndcg >= self.thresholds.ndcg_at_10
                }
            }
            
            # Calculate overall compliance
            compliance_rates = list(quality_summary["compliance"].values())
            overall_compliance = sum(compliance_rates) / len(compliance_rates) if compliance_rates else 0.0
            
            quality_summary["overall_compliance"] = round(overall_compliance, 3)
            quality_summary["status"] = "healthy" if overall_compliance >= 0.8 else "warning" if overall_compliance >= 0.6 else "critical"
            
            return quality_summary
            
        except Exception as e:
            logger.error("Failed to get quality summary", error=str(e), tenant_id=tenant_id)
            return {
                "tenant_id": tenant_id,
                "time_window": time_window,
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "status": "error"
            }
    
    def get_recent_benchmarks(self, tenant_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent benchmark results for a tenant."""
        try:
            benchmark_files = list(self.benchmark_dir.glob(f"*_{tenant_id}_*.json"))
            benchmark_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            recent_benchmarks = []
            for filepath in benchmark_files[:limit]:
                try:
                    with open(filepath, 'r') as f:
                        benchmark_data = json.load(f)
                        recent_benchmarks.append(benchmark_data)
                except Exception as e:
                    logger.warning("Failed to read benchmark file", 
                                 filepath=str(filepath), 
                                 error=str(e))
            
            return recent_benchmarks
            
        except Exception as e:
            logger.error("Failed to get recent benchmarks", error=str(e), tenant_id=tenant_id)
            return []
    
    def reset_benchmarks(self, tenant_id: Optional[str] = None) -> None:
        """Reset benchmark results for a tenant or all tenants."""
        try:
            if tenant_id:
                # Remove tenant-specific benchmarks
                pattern = f"*_{tenant_id}_*.json"
                for filepath in self.benchmark_dir.glob(pattern):
                    filepath.unlink()
                logger.info("Benchmarks reset for tenant", tenant_id=tenant_id)
            else:
                # Remove all benchmarks
                for filepath in self.benchmark_dir.glob("*.json"):
                    filepath.unlink()
                logger.info("All benchmarks reset")
                
        except Exception as e:
            logger.error("Failed to reset benchmarks", error=str(e))


# Global search QA instance
search_qa = None


def get_search_qa() -> SearchQualityAssurance:
    """Get the global search QA instance."""
    global search_qa
    if search_qa is None:
        # Initialize with default config - will be updated when settings are loaded
        search_qa = SearchQualityAssurance({})
    return search_qa


def initialize_search_qa(config: Dict[str, Any]) -> None:
    """Initialize the global search QA instance with configuration."""
    global search_qa
    search_qa = SearchQualityAssurance(config)
