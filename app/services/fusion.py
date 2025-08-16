"""Fusion engine for combining BM25 and vector search results."""

import asyncio
import time
from typing import Dict, List, Any, Optional, Union
import structlog

logger = structlog.get_logger(__name__)


class ReciprocalRankFusion:
    """Implements Reciprocal Rank Fusion algorithm for combining search results."""
    
    def __init__(self, k: int = 60, weight_bm25: float = 1.0, weight_vector: float = 1.0):
        self.k = k
        self.weight_bm25 = weight_bm25
        self.weight_vector = weight_vector
    
    def fuse_results(self, bm25_results: List[Dict], vector_results: List[Dict]) -> List[Dict]:
        """Fuse BM25 and vector search results using RRF."""
        # Create document ID to rank mapping
        bm25_ranks = {doc["citations"]["chunk_id"]: i for i, doc in enumerate(bm25_results)}
        vector_ranks = {doc["citations"]["chunk_id"]: i for i, doc in enumerate(vector_results)}
        
        # Calculate RRF scores
        fused_scores = {}
        all_docs = set(bm25_ranks.keys()) | set(vector_ranks.keys())
        
        for doc_id in all_docs:
            bm25_rank = bm25_ranks.get(doc_id, self.k)
            vector_rank = vector_ranks.get(doc_id, self.k)
            
            # RRF formula: 1 / (k + rank)
            rrf_score = (
                (self.weight_bm25 / (self.k + bm25_rank)) +
                (self.weight_vector / (self.k + vector_rank))
            )
            
            fused_scores[doc_id] = rrf_score
        
        # Sort by fused scores and reconstruct results
        sorted_docs = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Merge document information
        fused_results = []
        for doc_id, score in sorted_docs:
            # Find document in both result sets
            bm25_doc = next((doc for doc in bm25_results if doc["citations"]["chunk_id"] == doc_id), None)
            vector_doc = next((doc for doc in vector_results if doc["citations"]["chunk_id"] == doc_id), None)
            
            # Merge document data
            merged_doc = self._merge_document_data(bm25_doc, vector_doc, score)
            fused_results.append(merged_doc)
        
        return fused_results
    
    def _merge_document_data(self, bm25_doc: Optional[Dict], vector_doc: Optional[Dict], 
                            fused_score: float) -> Dict:
        """Merge document data from both search sources."""
        if bm25_doc and vector_doc:
            # Both sources have the document
            merged = bm25_doc.copy()
            merged["fused_score"] = fused_score
            merged["search_sources"] = ["bm25", "vector"]
            merged["bm25_score"] = bm25_doc.get("score", 0.0)
            merged["vector_score"] = vector_doc.get("score", 0.0)
            merged["highlight"] = self._merge_highlights(
                bm25_doc.get("highlight", {}),
                vector_doc.get("highlight", {})
            )
        elif bm25_doc:
            # Only BM25 has the document
            merged = bm25_doc.copy()
            merged["fused_score"] = fused_score
            merged["search_sources"] = ["bm25"]
            merged["bm25_score"] = bm25_doc.get("score", 0.0)
            merged["vector_score"] = 0.0
        else:
            # Only vector has the document
            merged = vector_doc.copy()
            merged["fused_score"] = fused_score
            merged["search_sources"] = ["vector"]
            merged["bm25_score"] = 0.0
            merged["vector_score"] = vector_doc.get("score", 0.0)
        
        return merged
    
    def _merge_highlights(self, bm25_highlights: Dict, vector_highlights: Dict) -> Dict:
        """Merge highlights from both search sources."""
        merged = {}
        all_fields = set(bm25_highlights.keys()) | set(vector_highlights.keys())
        
        for field in all_fields:
            bm25_fragments = bm25_highlights.get(field, [])
            vector_fragments = vector_highlights.get(field, [])
            
            # Combine and deduplicate fragments
            all_fragments = bm25_fragments + vector_fragments
            # Simple deduplication (in production, use more sophisticated text similarity)
            unique_fragments = list(dict.fromkeys(all_fragments))
            merged[field] = unique_fragments[:3]  # Limit to 3 fragments
        
        return merged


class Reranker:
    """Abstract interface for document reranking."""
    
    async def rerank(self, query: str, documents: List[Dict], 
                    max_documents: int = 100) -> List[Dict]:
        """Rerank documents based on query relevance."""
        raise NotImplementedError
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the reranker model."""
        raise NotImplementedError


class CrossEncoderReranker(Reranker):
    """Cross-encoder reranker using sentence-transformers."""
    
    def __init__(self, model_name: str = "ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the cross-encoder model."""
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(self.model_name)
            logger.info("Cross-encoder reranker loaded", model=self.model_name)
        except ImportError:
            logger.warning("sentence-transformers not available, using stub mode")
            self.model = None
    
    async def rerank(self, query: str, documents: List[Dict], 
                    max_documents: int = 100) -> List[Dict]:
        """Rerank documents using cross-encoder."""
        if not self.model:
            logger.warning("Cross-encoder not available, returning original order")
            return documents[:max_documents]
        
        try:
            # Prepare query-document pairs
            pairs = []
            for doc in documents[:max_documents]:
                # Create text representation for reranking
                text = self._create_document_text(doc)
                pairs.append([query, text])
            
            # Get reranking scores
            scores = self.model.predict(pairs)
            
            # Sort documents by reranking scores
            scored_docs = list(zip(documents[:max_documents], scores))
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            
            # Update documents with reranking scores
            reranked_docs = []
            for doc, score in scored_docs:
                doc_copy = doc.copy()
                doc_copy["rerank_score"] = float(score)
                doc_copy["final_score"] = doc_copy.get("fused_score", 0.0) * 0.7 + float(score) * 0.3
                reranked_docs.append(doc_copy)
            
            return reranked_docs
            
        except Exception as e:
            logger.error("Reranking failed", error=str(e))
            return documents[:max_documents]
    
    def _create_document_text(self, document: Dict) -> str:
        """Create text representation for reranking."""
        parts = []
        
        if document.get("subject"):
            parts.append(f"Subject: {document['subject']}")
        
        if document.get("content"):
            # Truncate content for reranking
            content = document["content"][:500]  # Limit to 500 chars
            parts.append(f"Content: {content}")
        
        if document.get("from_addr"):
            parts.append(f"From: {document['from_addr']}")
        
        return " | ".join(parts)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the reranker model."""
        return {
            "type": "cross_encoder",
            "model": self.model_name,
            "available": self.model is not None,
            "description": "Cross-encoder for query-document relevance scoring"
        }


class StubReranker(Reranker):
    """Stub reranker for testing and when no reranker is available."""
    
    async def rerank(self, query: str, documents: List[Dict], 
                    max_documents: int = 100) -> List[Dict]:
        """Return documents in original order (no reranking)."""
        return documents[:max_documents]
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get stub reranker information."""
        return {
            "type": "stub",
            "model": "none",
            "available": True,
            "description": "Stub reranker - no actual reranking performed"
        }


class HybridSearchEngine:
    """Orchestrates hybrid search combining BM25 and vector search."""
    
    def __init__(self, bm25_service, vector_service,
                 fusion_engine: ReciprocalRankFusion, reranker: Optional[Reranker] = None):
        self.bm25_service = bm25_service
        self.vector_service = vector_service
        self.fusion_engine = fusion_engine
        self.reranker = reranker
        self.metrics = FusionMetrics()
    
    async def hybrid_search(self, tenant_id: str, query: str, filters: Dict[str, Any] = None,
                           size: int = 20, from_: int = 0, enable_reranking: bool = False) -> Dict[str, Any]:
        """Execute hybrid search with optional reranking."""
        start_time = time.time()
        
        try:
            # Execute searches in parallel
            bm25_task = self.bm25_service.search(tenant_id, query, filters, size * 2, from_)
            vector_task = self.vector_service.search(tenant_id, query, filters, size * 2, from_)
            
            bm25_results, vector_results = await asyncio.gather(bm25_task, vector_task)
            
            # Fuse results
            fused_results = self.fusion_engine.fuse_results(
                bm25_results["hits"]["hits"],
                vector_results["hits"]["hits"]
            )
            
            # Apply reranking if enabled and available
            rerank_time = 0.0
            if enable_reranking and self.reranker:
                rerank_start = time.time()
                fused_results = await self.reranker.rerank(query, fused_results)
                rerank_time = time.time() - rerank_start
            
            # Limit results to requested size
            final_results = fused_results[:size]
            
            # Calculate performance metrics
            search_time = time.time() - start_time
            
            # Record metrics
            self.metrics.record_search(
                search_time=search_time,
                bm25_hits=bm25_results["hits"]["total"]["value"],
                vector_hits=vector_results["hits"]["total"]["value"],
                fused_hits=len(fused_results),
                rerank_time=rerank_time
            )
            
            return {
                "hits": {
                    "total": {"value": len(fused_results), "relation": "eq"},
                    "hits": final_results
                },
                "search_type": "hybrid",
                "performance": {
                    "total_time": search_time,
                    "bm25_time": bm25_results.get("performance", {}).get("search_time", 0),
                    "vector_time": vector_results.get("performance", {}).get("search_time", 0),
                    "fusion_time": search_time - max(
                        bm25_results.get("performance", {}).get("search_time", 0),
                        vector_results.get("performance", {}).get("search_time", 0)
                    ),
                    "rerank_time": rerank_time
                },
                "search_sources": {
                    "bm25": {
                        "total_hits": bm25_results["hits"]["total"]["value"],
                        "returned_hits": len(bm25_results["hits"]["hits"])
                    },
                    "vector": {
                        "total_hits": vector_results["hits"]["total"]["value"],
                        "returned_hits": len(vector_results["hits"]["hits"])
                    }
                }
            }
            
        except Exception as e:
            logger.error("Hybrid search failed", 
                        tenant_id=tenant_id, 
                        query=query, 
                        error=str(e))
            raise
    
    async def get_search_analytics(self, tenant_id: str, time_range: str = "24h") -> Dict[str, Any]:
        """Get search performance analytics."""
        try:
            # Get performance metrics from both services
            bm25_metrics = await self.bm25_service.get_health_status(tenant_id)
            vector_metrics = await self.vector_service.get_health_status(tenant_id)
            
            return {
                "tenant_id": tenant_id,
                "time_range": time_range,
                "bm25_metrics": bm25_metrics,
                "vector_metrics": vector_metrics,
                "fusion_config": {
                    "k": self.fusion_engine.k,
                    "weight_bm25": self.fusion_engine.weight_bm25,
                    "weight_vector": self.fusion_engine.weight_vector,
                    "reranker_enabled": self.reranker is not None
                },
                "performance_stats": self.metrics.get_performance_stats(),
                "health_status": self.metrics.get_health_status()
            }
            
        except Exception as e:
            logger.error("Failed to get search analytics", 
                        tenant_id=tenant_id, 
                        error=str(e))
            raise


class FusionMetrics:
    """Metrics collection for fusion engine performance."""
    
    def __init__(self):
        self.search_times = []
        self.fusion_times = []
        self.rerank_times = []
        self.result_counts = []
    
    def record_search(self, search_time: float, bm25_hits: int, vector_hits: int, 
                     fused_hits: int, rerank_time: float = 0.0):
        """Record search performance metrics."""
        self.search_times.append(search_time)
        self.fusion_times.append(search_time - rerank_time)
        self.rerank_times.append(rerank_time)
        self.result_counts.append({
            "bm25": bm25_hits,
            "vector": vector_hits,
            "fused": fused_hits
        })
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        if not self.search_times:
            return {}
        
        return {
            "total_searches": len(self.search_times),
            "avg_search_time": sum(self.search_times) / len(self.search_times),
            "p95_search_time": sorted(self.search_times)[int(len(self.search_times) * 0.95)],
            "avg_fusion_time": sum(self.fusion_times) / len(self.fusion_times),
            "avg_rerank_time": sum(self.rerank_times) / len(self.rerank_times),
            "avg_result_counts": {
                "bm25": sum(r["bm25"] for r in self.result_counts) / len(self.result_counts),
                "vector": sum(r["vector"] for r in self.result_counts) / len(self.result_counts),
                "fused": sum(r["fused"] for r in self.result_counts) / len(self.result_counts)
            }
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status based on performance metrics."""
        if not self.search_times:
            return {"status": "no_data", "message": "No search data available"}
        
        avg_time = sum(self.search_times) / len(self.search_times)
        
        if avg_time < 0.1:
            status = "excellent"
        elif avg_time < 0.5:
            status = "good"
        elif avg_time < 1.0:
            status = "fair"
        else:
            status = "poor"
        
        return {
            "status": status,
            "avg_search_time": avg_time,
            "total_searches": len(self.search_times),
            "last_search": max(self.search_times) if self.search_times else None
        }


# Factory functions for creating fusion components
def create_fusion_engine(k: int = 60, weight_bm25: float = 1.0, weight_vector: float = 1.0) -> ReciprocalRankFusion:
    """Create a Reciprocal Rank Fusion engine."""
    return ReciprocalRankFusion(k=k, weight_bm25=weight_bm25, weight_vector=weight_vector)


def create_reranker(reranker_type: str = "stub", model_name: str = None) -> Reranker:
    """Create a reranker based on configuration."""
    if reranker_type == "cross_encoder":
        return CrossEncoderReranker(model_name or "ms-marco-MiniLM-L-6-v2")
    else:
        return StubReranker()


def create_hybrid_search_engine(bm25_service, vector_service, 
                               k: int = 60, weight_bm25: float = 1.0, weight_vector: float = 1.0,
                               reranker_type: str = "stub", reranker_model: str = None) -> HybridSearchEngine:
    """Create a hybrid search engine with all components."""
    fusion_engine = create_fusion_engine(k, weight_bm25, weight_vector)
    reranker = create_reranker(reranker_type, reranker_model)
    
    return HybridSearchEngine(bm25_service, vector_service, fusion_engine, reranker)
