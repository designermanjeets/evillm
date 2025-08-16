"""Hybrid retrieval service with fallback to SQL text search."""

import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import structlog
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.session import get_database_session
from app.config.manager import ConfigManager

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
        
    async def retrieve(
        self, 
        tenant_id: str, 
        query: str, 
        k: int = 10, 
        filters: Optional[Dict[str, Any]] = None
    ) -> List[CitationItem]:
        """Retrieve top-k results with citations."""
        try:
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
        # TODO: Check if BM25/vector services are configured and healthy
        return False
    
    async def _hybrid_search(
        self, 
        tenant_id: str, 
        query: str, 
        k: int, 
        filters: Optional[Dict[str, Any]] = None
    ) -> List[CitationItem]:
        """Perform hybrid search using BM25 + vector search."""
        # TODO: Implement actual hybrid search
        # For now, return empty to trigger fallback
        return []
    
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
