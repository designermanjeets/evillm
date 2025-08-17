"""Search metrics API router for monitoring and analytics."""

from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse
import structlog

from app.services.search_metrics import get_search_metrics_collector
from app.middleware.tenant_isolation import get_tenant_id

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/search-metrics", tags=["search-metrics"])


@router.get("/health")
async def health_check():
    """Health check for search metrics service."""
    try:
        collector = get_search_metrics_collector()
        return {
            "status": "healthy",
            "service": "search-metrics",
            "metrics_collected": len(collector.search_metrics)
        }
    except Exception as e:
        logger.error("Search metrics health check failed", error=str(e))
        raise HTTPException(status_code=500, detail="Search metrics service unhealthy")


@router.get("/summary/{tenant_id}")
async def get_tenant_summary(
    tenant_id: str,
    time_window: str = Query("24h", description="Time window: 1h, 24h, 7d, 30d"),
    current_tenant: str = Depends(get_tenant_id)
):
    """Get search performance summary for a specific tenant."""
    try:
        # Validate tenant access
        if current_tenant != tenant_id:
            raise HTTPException(status_code=403, detail="Access denied to tenant metrics")
        
        collector = get_search_metrics_collector()
        summary = collector.get_performance_summary(tenant_id, time_window)
        
        return {
            "tenant_id": tenant_id,
            "time_window": time_window,
            "summary": summary.__dict__
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get tenant summary", error=str(e), tenant_id=tenant_id)
        raise HTTPException(status_code=500, detail="Failed to retrieve tenant summary")


@router.get("/summary")
async def get_current_tenant_summary(
    time_window: str = Query("24h", description="Time window: 1h, 24h, 7d, 30d"),
    current_tenant: str = Depends(get_tenant_id)
):
    """Get search performance summary for the current tenant."""
    try:
        collector = get_search_metrics_collector()
        summary = collector.get_performance_summary(current_tenant, time_window)
        
        return {
            "tenant_id": current_tenant,
            "time_window": time_window,
            "summary": summary.__dict__
        }
        
    except Exception as e:
        logger.error("Failed to get current tenant summary", error=str(e), tenant_id=current_tenant)
        raise HTTPException(status_code=500, detail="Failed to retrieve tenant summary")


@router.get("/global")
async def get_global_metrics(
    time_window: str = Query("24h", description="Time window: 1h, 24h, 7d, 30d"),
    current_tenant: str = Depends(get_tenant_id)
):
    """Get global search metrics across all tenants (admin only)."""
    try:
        # TODO: Add admin role check here
        # For now, allow any authenticated user to access global metrics
        
        collector = get_search_metrics_collector()
        global_metrics = collector.get_global_metrics(time_window)
        
        return {
            "time_window": time_window,
            "global_metrics": global_metrics
        }
        
    except Exception as e:
        logger.error("Failed to get global metrics", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve global metrics")


@router.get("/export/{tenant_id}")
async def export_tenant_metrics(
    tenant_id: str,
    time_window: str = Query("24h", description="Time window: 1h, 24h, 7d, 30d"),
    current_tenant: str = Depends(get_tenant_id)
):
    """Export search metrics for a specific tenant."""
    try:
        # Validate tenant access
        if current_tenant != tenant_id:
            raise HTTPException(status_code=403, detail="Access denied to tenant metrics")
        
        collector = get_search_metrics_collector()
        exported_data = collector.export_metrics(tenant_id, time_window)
        
        return JSONResponse(
            content=exported_data,
            headers={
                "Content-Disposition": f"attachment; filename=search_metrics_{tenant_id}_{time_window}.json"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to export tenant metrics", error=str(e), tenant_id=tenant_id)
        raise HTTPException(status_code=500, detail="Failed to export tenant metrics")


@router.get("/export")
async def export_current_tenant_metrics(
    time_window: str = Query("24h", description="Time window: 1h, 24h, 7d, 30d"),
    current_tenant: str = Depends(get_tenant_id)
):
    """Export search metrics for the current tenant."""
    try:
        collector = get_search_metrics_collector()
        exported_data = collector.export_metrics(current_tenant, time_window)
        
        return JSONResponse(
            content=exported_data,
            headers={
                "Content-Disposition": f"attachment; filename=search_metrics_{current_tenant}_{time_window}.json"
            }
        )
        
    except Exception as e:
        logger.error("Failed to export current tenant metrics", error=str(e), tenant_id=current_tenant)
        raise HTTPException(status_code=500, detail="Failed to export tenant metrics")


@router.delete("/reset/{tenant_id}")
async def reset_tenant_metrics(
    tenant_id: str,
    current_tenant: str = Depends(get_tenant_id)
):
    """Reset search metrics for a specific tenant."""
    try:
        # Validate tenant access
        if current_tenant != tenant_id:
            raise HTTPException(status_code=403, detail="Access denied to tenant metrics")
        
        collector = get_search_metrics_collector()
        collector.reset_metrics(tenant_id)
        
        return {
            "message": f"Metrics reset for tenant {tenant_id}",
            "tenant_id": tenant_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to reset tenant metrics", error=str(e), tenant_id=tenant_id)
        raise HTTPException(status_code=500, detail="Failed to reset tenant metrics")


@router.delete("/reset")
async def reset_current_tenant_metrics(
    current_tenant: str = Depends(get_tenant_id)
):
    """Reset search metrics for the current tenant."""
    try:
        collector = get_search_metrics_collector()
        collector.reset_metrics(current_tenant)
        
        return {
            "message": f"Metrics reset for tenant {current_tenant}",
            "tenant_id": current_tenant
        }
        
    except Exception as e:
        logger.error("Failed to reset current tenant metrics", error=str(e), tenant_id=current_tenant)
        raise HTTPException(status_code=500, detail="Failed to reset tenant metrics")


@router.get("/recent-searches/{tenant_id}")
async def get_recent_searches(
    tenant_id: str,
    limit: int = Query(100, ge=1, le=1000, description="Number of recent searches to return"),
    current_tenant: str = Depends(get_tenant_id)
):
    """Get recent search operations for a specific tenant."""
    try:
        # Validate tenant access
        if current_tenant != tenant_id:
            raise HTTPException(status_code=403, detail="Access denied to tenant metrics")
        
        collector = get_search_metrics_collector()
        tenant_data = collector.tenant_metrics.get(tenant_id, {})
        searches = list(tenant_data.get('searches', []))[-limit:]
        
        recent_searches = [
            {
                'query_id': m.query_id,
                'query': m.query[:100] if len(m.query) > 100 else m.query,
                'timestamp': m.timestamp.isoformat(),
                'total_time_ms': m.total_time_ms,
                'search_type': m.search_type,
                'success': m.success,
                'final_results': m.final_results,
                'k_requested': m.k_requested
            }
            for m in searches
        ]
        
        return {
            "tenant_id": tenant_id,
            "limit": limit,
            "total_searches": len(recent_searches),
            "recent_searches": recent_searches
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get recent searches", error=str(e), tenant_id=tenant_id)
        raise HTTPException(status_code=500, detail="Failed to retrieve recent searches")


@router.get("/recent-searches")
async def get_current_tenant_recent_searches(
    limit: int = Query(100, ge=1, le=1000, description="Number of recent searches to return"),
    current_tenant: str = Depends(get_tenant_id)
):
    """Get recent search operations for the current tenant."""
    try:
        collector = get_search_metrics_collector()
        tenant_data = collector.tenant_metrics.get(current_tenant, {})
        searches = list(tenant_data.get('searches', []))[-limit:]
        
        recent_searches = [
            {
                'query_id': m.query_id,
                'query': m.query[:100] if len(m.query) > 100 else m.query,
                'timestamp': m.timestamp.isoformat(),
                'total_time_ms': m.total_time_ms,
                'search_type': m.search_type,
                'success': m.success,
                'final_results': m.final_results,
                'k_requested': m.k_requested
            }
            for m in searches
        ]
        
        return {
            "tenant_id": current_tenant,
            "limit": limit,
            "total_searches": len(recent_searches),
            "recent_searches": recent_searches
        }
        
    except Exception as e:
        logger.error("Failed to get current tenant recent searches", error=str(e), tenant_id=current_tenant)
        raise HTTPException(status_code=500, detail="Failed to retrieve recent searches")


@router.get("/quality/{tenant_id}")
async def get_quality_metrics(
    tenant_id: str,
    time_window: str = Query("24h", description="Time window: 1h, 24h, 7d, 30d"),
    current_tenant: str = Depends(get_tenant_id)
):
    """Get search quality metrics for a specific tenant."""
    try:
        # Validate tenant access
        if current_tenant != tenant_id:
            raise HTTPException(status_code=403, detail="Access denied to tenant metrics")
        
        collector = get_search_metrics_collector()
        tenant_data = collector.tenant_metrics.get(tenant_id, {})
        quality_metrics = list(tenant_data.get('quality', []))
        
        if not quality_metrics:
            return {
                "tenant_id": tenant_id,
                "time_window": time_window,
                "message": "No quality metrics available",
                "quality_metrics": []
            }
        
        # Filter by time window if needed
        # TODO: Implement time filtering for quality metrics
        
        quality_summary = {
            "tenant_id": tenant_id,
            "time_window": time_window,
            "total_quality_metrics": len(quality_metrics),
            "avg_hit_at_1": sum(m.hit_at_k.get(1, 0.0) for m in quality_metrics) / len(quality_metrics),
            "avg_hit_at_3": sum(m.hit_at_k.get(3, 0.0) for m in quality_metrics) / len(quality_metrics),
            "avg_hit_at_5": sum(m.hit_at_k.get(5, 0.0) for m in quality_metrics) / len(quality_metrics),
            "avg_hit_at_10": sum(m.hit_at_k.get(10, 0.0) for m in quality_metrics) / len(quality_metrics),
            "avg_mrr": sum(m.mean_reciprocal_rank for m in quality_metrics) / len(quality_metrics),
            "avg_ndcg": sum(m.normalized_discounted_cumulative_gain for m in quality_metrics) / len(quality_metrics),
            "recent_quality_metrics": [
                {
                    'query_id': m.query_id,
                    'timestamp': m.timestamp.isoformat(),
                    'hit_at_k': m.hit_at_k,
                    'mean_reciprocal_rank': m.mean_reciprocal_rank,
                    'normalized_discounted_cumulative_gain': m.normalized_discounted_cumulative_gain,
                    'citation_coverage': m.citation_coverage
                }
                for m in quality_metrics[-10:]  # Last 10 quality metrics
            ]
        }
        
        return quality_summary
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get quality metrics", error=str(e), tenant_id=tenant_id)
        raise HTTPException(status_code=500, detail="Failed to retrieve quality metrics")


@router.get("/quality")
async def get_current_tenant_quality_metrics(
    time_window: str = Query("24h", description="Time window: 1h, 24h, 7d, 30d"),
    current_tenant: str = Depends(get_tenant_id)
):
    """Get search quality metrics for the current tenant."""
    try:
        collector = get_search_metrics_collector()
        tenant_data = collector.tenant_metrics.get(current_tenant, {})
        quality_metrics = list(tenant_data.get('quality', []))
        
        if not quality_metrics:
            return {
                "tenant_id": current_tenant,
                "time_window": time_window,
                "message": "No quality metrics available",
                "quality_metrics": []
            }
        
        quality_summary = {
            "tenant_id": current_tenant,
            "time_window": time_window,
            "total_quality_metrics": len(quality_metrics),
            "avg_hit_at_1": sum(m.hit_at_k.get(1, 0.0) for m in quality_metrics) / len(quality_metrics),
            "avg_hit_at_3": sum(m.hit_at_k.get(3, 0.0) for m in quality_metrics) / len(quality_metrics),
            "avg_hit_at_5": sum(m.hit_at_k.get(5, 0.0) for m in quality_metrics) / len(quality_metrics),
            "avg_hit_at_10": sum(m.hit_at_k.get(10, 0.0) for m in quality_metrics) / len(quality_metrics),
            "avg_mrr": sum(m.mean_reciprocal_rank for m in quality_metrics) / len(quality_metrics),
            "avg_ndcg": sum(m.normalized_discounted_cumulative_gain for m in quality_metrics) / len(quality_metrics),
            "recent_quality_metrics": [
                {
                    'query_id': m.query_id,
                    'timestamp': m.timestamp.isoformat(),
                    'hit_at_k': m.hit_at_k,
                    'mean_reciprocal_rank': m.mean_reciprocal_rank,
                    'normalized_discounted_cumulative_gain': m.normalized_discounted_cumulative_gain,
                    'citation_coverage': m.citation_coverage
                }
                for m in quality_metrics[-10:]  # Last 10 quality metrics
            ]
        }
        
        return quality_summary
        
    except Exception as e:
        logger.error("Failed to get current tenant quality metrics", error=str(e), tenant_id=current_tenant)
        raise HTTPException(status_code=500, detail="Failed to retrieve quality metrics")

