"""Hybrid retrieval service with fallback to SQL text search."""

import asyncio
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import structlog
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.session import get_database_session
from app.config.manager import ConfigManager
from app.services.bm25_search import BM25SearchService
from app.services.embeddings import VectorSearchService
from app.services.fusion import ReciprocalRankFusion

logger = structlog.get_logger(__name__)


@dataclass
class CitationItem:
    """Citation item with metadata and relevance score."""
    email_id: str
    chunk_uid: str
    object_key: str
    score: float
    content_preview: str
    chunk_id: Optional[str] = None
    attachment_id: Optional[str] = None


class RetrieverAdapter:
    """Hybrid retrieval adapter with fallback to SQL text search."""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.fallback_enabled = config.get("retriever.fallback_enabled", True)
        self.max_results = config.get("retriever.max_results", 10)
        
        # Initialize search services
        self.bm25_service = BM25SearchService(
            host=config.get("search.bm25.host", "localhost"),
            index_prefix=config.get("search.bm25.index_prefix", "emails")
        )
        self.vector_service = VectorSearchService(config)
        self.fusion_engine = ReciprocalRankFusion(
            k=config.get("search.fusion.k", 60),
            weight_bm25=config.get("search.fusion.weight_bm25", 1.0),
            weight_vector=config.get("search.fusion.weight_vector", 1.0)
        )
        
        # Services will be initialized lazily on first use
        self._services_initialized = False
    
    async def _ensure_services_initialized(self):
        """Ensure search services are initialized."""
        if not self._services_initialized:
            try:
                await self.bm25_service.initialize()
                self._services_initialized = True
                logger.info("Search services initialized")
            except Exception as e:
                logger.error("Failed to initialize search services", error=str(e))
    
    async def retrieve(
        self, 
        tenant_id: str, 
        query: str, 
        k: int = 10, 
        filters: Optional[Dict[str, Any]] = None
    ) -> List[CitationItem]:
        """Retrieve top-k results with citations using hybrid search."""
        try:
            # Ensure services are initialized
            await self._ensure_services_initialized()
            
            # Try hybrid search first if available
            if self._has_hybrid_search():
                results = await self._hybrid_search(tenant_id, query, k, filters)
                if results:
                    return results
            
            # Fallback to SQL text search
            if self.fallback_enabled:
                return await self._sql_text_search(tenant_id, query, k, filters)
            
            # No results available
            logger.warning("No retrieval services available", tenant_id=tenant_id)
            return []
            
        except Exception as e:
            logger.error("Retrieval failed", error=str(e), tenant_id=tenant_id)
            return []
    
    def _has_hybrid_search(self) -> bool:
        """Check if hybrid search services are available."""
        return (self.bm25_service.client is not None or 
                self.vector_service._client["type"] != "stub")
    
    async def _hybrid_search(
        self, 
        tenant_id: str, 
        query: str, 
        k: int, 
        filters: Optional[Dict[str, Any]] = None
    ) -> List[CitationItem]:
        """Perform hybrid search using BM25 + vector search with performance tracking."""
        start_time = time.time()
        search_metrics = {
            "bm25_results": 0,
            "vector_results": 0,
            "fused_results": 0,
            "total_time_ms": 0
        }
        
        try:
            # Get BM25 results
            bm25_results = []
            bm25_start = time.time()
            if self.bm25_service.client:
                bm25_response = await self.bm25_service.search(
                    tenant_id, query, filters, k * 2, 0
                )
                bm25_results = self._convert_bm25_results(bm25_response)
                search_metrics["bm25_results"] = len(bm25_results)
                logger.info("BM25 search completed", 
                           tenant_id=tenant_id, 
                           results=len(bm25_results),
                           time_ms=round((time.time() - bm25_start) * 1000, 2))
            
            # Get vector search results
            vector_results = []
            vector_start = time.time()
            if self.vector_service._client["type"] != "stub":
                # Generate query embedding
                embedding_provider = self._get_embedding_provider()
                if embedding_provider:
                    query_vector = await embedding_provider.embed_batch([query])
                    if query_vector:
                        vector_response = await self.vector_service.search_similar(
                            tenant_id, query_vector[0], k * 2, 0.7
                        )
                        vector_results = self._convert_vector_results(vector_response)
                        search_metrics["vector_results"] = len(vector_results)
                        logger.info("Vector search completed", 
                                   tenant_id=tenant_id, 
                                   results=len(vector_results),
                                   time_ms=round((time.time() - vector_start) * 1000, 2))
            
            # If we have both types of results, fuse them
            if bm25_results and vector_results:
                fusion_start = time.time()
                fused_results = self.fusion_engine.fuse_results(bm25_results, vector_results)
                search_metrics["fused_results"] = len(fused_results)
                logger.info("Result fusion completed", 
                           tenant_id=tenant_id, 
                           fused_count=len(fused_results),
                           time_ms=round((time.time() - fusion_start) * 1000, 2))
                
                final_results = self._convert_to_citations(fused_results[:k])
            else:
                # If we only have one type, return it
                if bm25_results:
                    final_results = self._convert_to_citations(bm25_results[:k])
                elif vector_results:
                    final_results = self._convert_to_citations(vector_results[:k])
                else:
                    final_results = []
            
            # Calculate total time
            search_metrics["total_time_ms"] = round((time.time() - start_time) * 1000, 2)
            
            # Log search performance
            logger.info("Hybrid search completed", 
                       tenant_id=tenant_id,
                       query=query[:100],
                       k=k,
                       final_results=len(final_results),
                       metrics=search_metrics)
            
            return final_results
            
        except Exception as e:
            search_metrics["total_time_ms"] = round((time.time() - start_time) * 1000, 2)
            logger.error("Hybrid search failed", 
                        error=str(e), 
                        tenant_id=tenant_id,
                        metrics=search_metrics)
            return []
    
    def _get_embedding_provider(self):
        """Get embedding provider for vector search."""
        try:
            from app.services.embeddings import OpenAIEmbeddingProvider
            config = self.config
            api_key = config.get("models.embeddings.openai.api_key", "")
            return OpenAIEmbeddingProvider(api_key)
        except Exception as e:
            logger.warning("Failed to get embedding provider", error=str(e))
            return None
    
    def _convert_bm25_results(self, bm25_response: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert BM25 search results to standard format."""
        results = []
        hits = bm25_response.get("hits", {}).get("hits", [])
        
        for hit in hits:
            result = {
                "id": hit.get("_id"),
                "score": hit.get("_score", 0.0),
                "source": "bm25",
                "citations": {
                    "chunk_id": hit.get("_source", {}).get("chunk_id"),
                    "email_id": hit.get("_source", {}).get("email_id"),
                    "content": hit.get("_source", {}).get("content", ""),
                    "subject": hit.get("_source", {}).get("subject", ""),
                    "created_at": hit.get("_source", {}).get("created_at"),
                    "tenant_id": hit.get("_source", {}).get("tenant_id")
                }
            }
            results.append(result)
        
        return results
    
    def _convert_vector_results(self, vector_response: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert vector search results to standard format."""
        results = []
        
        for item in vector_response:
            result = {
                "id": item.get("id"),
                "score": item.get("score", 0.0),
                "citations": {
                    "chunk_id": item.get("payload", {}).get("chunk_id"),
                    "email_id": item.get("payload", {}).get("email_id"),
                    "content": item.get("payload", {}).get("content", ""),
                    "subject": ""  # Vector search doesn't have subject field
                }
            }
            results.append(result)
        
        return results
    
    def _convert_to_citations(self, search_results: List[Dict[str, Any]]) -> List[CitationItem]:
        """Convert search results to CitationItem objects."""
        citations = []
        
        for result in search_results:
            citations.append(CitationItem(
                email_id=result["citations"]["email_id"],
                chunk_uid=result["citations"]["chunk_id"],
                object_key=f"search_result_{result['id']}",
                score=result["score"],
                content_preview=result["citations"]["content"][:200] + "..." if len(result["citations"]["content"]) > 200 else result["citations"]["content"]
            ))
        
        return citations
    
    async def _sql_text_search(
        self, 
        tenant_id: str, 
        query: str, 
        k: int, 
        filters: Optional[Dict[str, Any]] = None
    ) -> List[CitationItem]:
        """Fallback to SQL text search over chunks content."""
        try:
            async with get_database_session(tenant_id) as db_session:
                # Build SQL query with tenant isolation
                sql = """
                SELECT 
                    c.chunk_id,
                    c.chunk_uid,
                    c.email_id,
                    c.attachment_id,
                    c.content,
                    c.object_key,
                    c.tenant_id
                FROM chunks c
                WHERE c.tenant_id = :tenant_id
                AND c.content ILIKE :query_pattern
                ORDER BY 
                    CASE 
                        WHEN c.content ILIKE :exact_pattern THEN 1.0
                        WHEN c.content ILIKE :start_pattern THEN 0.8
                        ELSE 0.5
                    END DESC,
                    c.created_at DESC
                LIMIT :limit
                """
                
                # Create search patterns
                query_pattern = f"%{query}%"
                exact_pattern = f"%{query}%"
                start_pattern = f"{query}%"
                
                # Execute query
                result = await db_session.execute(
                    text(sql),
                    {
                        "tenant_id": tenant_id,
                        "query_pattern": query_pattern,
                        "exact_pattern": exact_pattern,
                        "start_pattern": start_pattern,
                        "limit": min(k, self.max_results)
                    }
                )
                
                # Convert to CitationItem objects
                citations = []
                for row in result.fetchall():
                    # Calculate simple relevance score
                    content_lower = row.content.lower()
                    query_lower = query.lower()
                    
                    # Score based on pattern matching
                    if query_lower in content_lower:
                        if content_lower.startswith(query_lower):
                            score = 0.9
                        else:
                            score = 0.7
                    else:
                        score = 0.5
                    
                    # Create citation item
                    citation = CitationItem(
                        email_id=row.email_id,
                        chunk_uid=row.chunk_uid,
                        object_key=row.object_key,
                        score=score,
                        content_preview=row.content[:200] + "..." if len(row.content) > 200 else row.content,
                        chunk_id=row.chunk_id,
                        attachment_id=row.attachment_id
                    )
                    citations.append(citation)
                
                # Sort by score and return top-k
                citations.sort(key=lambda x: x.score, reverse=True)
                return citations[:k]
                
        except Exception as e:
            logger.error("SQL text search failed", error=str(e), tenant_id=tenant_id)
            return []
    
    async def get_citation_by_uid(self, tenant_id: str, chunk_uid: str) -> Optional[CitationItem]:
        """Get citation details by chunk UID."""
        try:
            async with get_database_session(tenant_id) as db_session:
                sql = """
                SELECT 
                    c.chunk_id,
                    c.chunk_uid,
                    c.email_id,
                    c.attachment_id,
                    c.content,
                    c.object_key,
                    c.tenant_id
                FROM chunks c
                WHERE c.tenant_id = :tenant_id AND c.chunk_uid = :chunk_uid
                """
                
                result = await db_session.execute(
                    text(sql),
                    {"tenant_id": tenant_id, "chunk_uid": chunk_uid}
                )
                
                row = result.fetchone()
                if row:
                    return CitationItem(
                        email_id=row.email_id,
                        chunk_uid=row.chunk_uid,
                        object_key=row.object_key,
                        score=1.0,  # Direct lookup gets full score
                        content_preview=row.content[:200] + "..." if len(row.content) > 200 else row.content,
                        chunk_id=row.chunk_id,
                        attachment_id=row.attachment_id
                    )
                return None
                
        except Exception as e:
            logger.error("Citation lookup failed", error=str(e), tenant_id=tenant_id, chunk_uid=chunk_uid)
            return None
