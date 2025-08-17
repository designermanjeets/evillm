"""Search Quality Assurance API router."""

from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, Body
from fastapi.responses import JSONResponse
import structlog
from datetime import datetime

from app.services.search_qa import get_search_qa, SearchQualityAssurance
from app.services.retriever import RetrieverAdapter
from app.middleware.tenant_isolation import get_tenant_id

logger = structlog.get_logger(__name__)

router = APIRouter(tags=["search-qa"])


@router.get("/health")
async def health_check():
    """Health check for search QA service."""
    try:
        qa_service = get_search_qa()
        return {
            "status": "healthy",
            "service": "search-qa",
            "thresholds": {
                "hit_at_5": qa_service.thresholds.hit_at_5,
                "mrr_at_10": qa_service.thresholds.mrr_at_10,
                "ndcg_at_10": qa_service.thresholds.ndcg_at_10
            }
        }
    except Exception as e:
        logger.error("Search QA health check failed", error=str(e))
        raise HTTPException(status_code=500, detail="Search QA service unhealthy")


@router.get("/quality/summary")
async def get_quality_summary(
    time_window: str = Query("24h", description="Time window: 1h, 24h, 7d, 30d"),
    current_tenant: str = Depends(get_tenant_id)
):
    """Get search quality summary for the current tenant."""
    try:
        qa_service = get_search_qa()
        summary = qa_service.get_quality_summary(current_tenant, time_window)
        
        return {
            "tenant_id": current_tenant,
            "time_window": time_window,
            "summary": summary
        }
        
    except Exception as e:
        logger.error("Failed to get quality summary", error=str(e), tenant_id=current_tenant)
        raise HTTPException(status_code=500, detail="Failed to retrieve quality summary")


@router.get("/quality/summary/{tenant_id}")
async def get_tenant_quality_summary(
    tenant_id: str,
    time_window: str = Query("24h", description="Time window: 1h, 24h, 7d, 30d"),
    current_tenant: str = Depends(get_tenant_id)
):
    """Get search quality summary for a specific tenant."""
    try:
        # Validate tenant access
        if current_tenant != tenant_id:
            raise HTTPException(status_code=403, detail="Access denied to tenant quality data")
        
        qa_service = get_search_qa()
        summary = qa_service.get_quality_summary(tenant_id, time_window)
        
        return {
            "tenant_id": tenant_id,
            "time_window": time_window,
            "summary": summary
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get tenant quality summary", error=str(e), tenant_id=tenant_id)
        raise HTTPException(status_code=500, detail="Failed to retrieve quality summary")


@router.post("/validate-query")
async def validate_query(
    query_data: Dict[str, str] = Body(...),
    current_tenant: str = Depends(get_tenant_id)
):
    """Validate a search query for quality and security."""
    try:
        query = query_data.get("query", "")
        tenant_id = query_data.get("tenant_id", current_tenant)
        
        # Validate tenant access
        if current_tenant != tenant_id:
            raise HTTPException(status_code=403, detail="Access denied to tenant")
        
        qa_service = get_search_qa()
        is_valid, warnings = qa_service.validate_query(query, tenant_id)
        
        return {
            "query": query,
            "tenant_id": tenant_id,
            "is_valid": is_valid,
            "warnings": warnings,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Query validation failed", error=str(e), tenant_id=current_tenant)
        raise HTTPException(status_code=500, detail="Query validation failed")


@router.post("/assess-quality")
async def assess_search_quality(
    assessment_data: Dict[str, Any] = Body(...),
    current_tenant: str = Depends(get_tenant_id)
):
    """Assess search quality for a specific query and results."""
    try:
        query = assessment_data.get("query", "")
        tenant_id = assessment_data.get("tenant_id", current_tenant)
        quality_metrics = assessment_data.get("quality_metrics", {})
        
        # Validate tenant access
        if current_tenant != tenant_id:
            raise HTTPException(status_code=403, detail="Access denied to tenant")
        
        qa_service = get_search_qa()
        assessment = qa_service.assess_search_quality(query, tenant_id, quality_metrics)
        
        return {
            "query": query,
            "tenant_id": tenant_id,
            "assessment": {
                "overall_pass": assessment.overall_pass,
                "quality_metrics": assessment.quality_metrics,
                "thresholds_met": assessment.thresholds_met,
                "warnings": assessment.warnings,
                "recommendations": assessment.recommendations
            },
            "timestamp": assessment.timestamp.isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Quality assessment failed", error=str(e), tenant_id=current_tenant)
        raise HTTPException(status_code=500, detail="Quality assessment failed")


@router.post("/benchmark")
async def run_search_benchmark(
    benchmark_data: Dict[str, Any] = Body(...),
    current_tenant: str = Depends(get_tenant_id)
):
    """Run a search quality benchmark on a set of queries."""
    try:
        tenant_id = benchmark_data.get("tenant_id", current_tenant)
        queries = benchmark_data.get("queries", [])
        
        # Validate tenant access
        if current_tenant != tenant_id:
            raise HTTPException(status_code=403, detail="Access denied to tenant")
        
        # Validate queries
        if not queries or not isinstance(queries, list):
            raise HTTPException(status_code=400, detail="Queries must be a non-empty list")
        
        if len(queries) > 100:
            raise HTTPException(status_code=400, detail="Maximum 100 queries allowed per benchmark")
        
        # Create retriever instance
        from app.config.manager import ConfigManager
        config = ConfigManager()
        retriever = RetrieverAdapter(config)
        
        # Run benchmark
        qa_service = get_search_qa()
        benchmark_result = await qa_service.run_benchmark(tenant_id, queries, retriever)
        
        return {
            "benchmark_id": benchmark_result.benchmark_id,
            "tenant_id": tenant_id,
            "timestamp": benchmark_result.timestamp.isoformat(),
            "total_queries": benchmark_result.total_queries,
            "passed_queries": benchmark_result.passed_queries,
            "failed_queries": benchmark_result.failed_queries,
            "quality_summary": benchmark_result.quality_summary,
            "threshold_compliance": benchmark_result.threshold_compliance,
            "recommendations": benchmark_result.recommendations
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Benchmark execution failed", error=str(e), tenant_id=current_tenant)
        raise HTTPException(status_code=500, detail="Benchmark execution failed")


@router.get("/benchmarks")
async def get_recent_benchmarks(
    limit: int = Query(10, ge=1, le=50, description="Number of recent benchmarks to return"),
    current_tenant: str = Depends(get_tenant_id)
):
    """Get recent benchmark results for the current tenant."""
    try:
        qa_service = get_search_qa()
        benchmarks = qa_service.get_recent_benchmarks(current_tenant, limit)
        
        return {
            "tenant_id": current_tenant,
            "limit": limit,
            "total_benchmarks": len(benchmarks),
            "benchmarks": benchmarks
        }
        
    except Exception as e:
        logger.error("Failed to get recent benchmarks", error=str(e), tenant_id=current_tenant)
        raise HTTPException(status_code=500, detail="Failed to retrieve benchmarks")


@router.get("/benchmarks/{tenant_id}")
async def get_tenant_benchmarks(
    tenant_id: str,
    limit: int = Query(10, ge=1, le=50, description="Number of recent benchmarks to return"),
    current_tenant: str = Depends(get_tenant_id)
):
    """Get recent benchmark results for a specific tenant."""
    try:
        # Validate tenant access
        if current_tenant != tenant_id:
            raise HTTPException(status_code=403, detail="Access denied to tenant benchmark data")
        
        qa_service = get_search_qa()
        benchmarks = qa_service.get_recent_benchmarks(tenant_id, limit)
        
        return {
            "tenant_id": tenant_id,
            "limit": limit,
            "total_benchmarks": len(benchmarks),
            "benchmarks": benchmarks
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get tenant benchmarks", error=str(e), tenant_id=tenant_id)
        raise HTTPException(status_code=500, detail="Failed to retrieve benchmarks")


@router.delete("/benchmarks/reset")
async def reset_benchmarks(
    current_tenant: str = Depends(get_tenant_id)
):
    """Reset benchmark results for the current tenant."""
    try:
        qa_service = get_search_qa()
        qa_service.reset_benchmarks(current_tenant)
        
        return {
            "message": f"Benchmarks reset for tenant {current_tenant}",
            "tenant_id": current_tenant
        }
        
    except Exception as e:
        logger.error("Failed to reset benchmarks", error=str(e), tenant_id=current_tenant)
        raise HTTPException(status_code=500, detail="Failed to reset benchmarks")


@router.delete("/benchmarks/reset/{tenant_id}")
async def reset_tenant_benchmarks(
    tenant_id: str,
    current_tenant: str = Depends(get_tenant_id)
):
    """Reset benchmark results for a specific tenant."""
    try:
        # Validate tenant access
        if current_tenant != tenant_id:
            raise HTTPException(status_code=403, detail="Access denied to tenant")
        
        qa_service = get_search_qa()
        qa_service.reset_benchmarks(tenant_id)
        
        return {
            "message": f"Benchmarks reset for tenant {tenant_id}",
            "tenant_id": tenant_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to reset tenant benchmarks", error=str(e), tenant_id=tenant_id)
        raise HTTPException(status_code=500, detail="Failed to reset benchmarks")


@router.get("/thresholds")
async def get_quality_thresholds(
    current_tenant: str = Depends(get_tenant_id)
):
    """Get current quality thresholds configuration."""
    try:
        qa_service = get_search_qa()
        
        return {
            "tenant_id": current_tenant,
            "thresholds": {
                "hit_at_5": qa_service.thresholds.hit_at_5,
                "mrr_at_10": qa_service.thresholds.mrr_at_10,
                "ndcg_at_10": qa_service.thresholds.ndcg_at_10
            },
            "description": {
                "hit_at_5": "Hit@5 threshold - percentage of relevant results in top 5",
                "mrr_at_10": "Mean Reciprocal Rank threshold for top 10 results",
                "ndcg_at_10": "Normalized Discounted Cumulative Gain threshold for top 10 results"
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error("Failed to get quality thresholds", error=str(e), tenant_id=current_tenant)
        raise HTTPException(status_code=500, detail="Failed to retrieve quality thresholds")


@router.get("/status")
async def get_qa_service_status(
    current_tenant: str = Depends(get_tenant_id)
):
    """Get overall QA service status and health."""
    try:
        qa_service = get_search_qa()
        
        # Get quality summary
        summary = qa_service.get_quality_summary(current_tenant, "24h")
        
        # Get recent benchmarks
        recent_benchmarks = qa_service.get_recent_benchmarks(current_tenant, 5)
        
        # Calculate overall health
        overall_health = "healthy"
        if summary.get("status") == "critical":
            overall_health = "critical"
        elif summary.get("status") == "warning":
            overall_health = "warning"
        
        return {
            "tenant_id": current_tenant,
            "overall_health": overall_health,
            "quality_status": summary.get("status", "unknown"),
            "recent_benchmarks": len(recent_benchmarks),
            "thresholds": {
                "hit_at_5": qa_service.thresholds.hit_at_5,
                "mrr_at_10": qa_service.thresholds.mrr_at_10,
                "ndcg_at_10": qa_service.thresholds.ndcg_at_10
            },
            "last_updated": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error("Failed to get QA service status", error=str(e), tenant_id=current_tenant)
        raise HTTPException(status_code=500, detail="Failed to retrieve service status")
