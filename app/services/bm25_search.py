"""BM25 search service with OpenSearch integration."""

import asyncio
import json
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import structlog

logger = structlog.get_logger(__name__)


class BM25SearchService:
    """BM25 search service using OpenSearch/Elasticsearch."""
    
    def __init__(self, host: str, index_prefix: str = "emails"):
        self.host = host
        self.index_prefix = index_prefix
        self.client = None
        self.optimizer = BM25SearchOptimizer()
        self.index_manager = BM25IndexManager(host, index_prefix)
    
    async def initialize(self):
        """Initialize OpenSearch client."""
        try:
            from opensearchpy import AsyncOpenSearch
            self.client = AsyncOpenSearch(
                hosts=[self.host],
                http_auth=None,  # Add auth if needed
                use_ssl=False,    # Configure SSL as needed
                verify_certs=False,
                ssl_show_warn=False
            )
            
            # Test connection
            await self.client.info()
            logger.info("BM25 search service initialized", host=self.host)
            
        except ImportError:
            logger.warning("OpenSearch client not available, using stub mode")
            self.client = None
        except Exception as e:
            logger.error("Failed to initialize OpenSearch client", error=str(e))
            self.client = None
    
    def _get_index_name(self, tenant_id: str) -> str:
        """Generate tenant-scoped index name."""
        return f"{self.index_prefix}_{tenant_id}"
    
    def _build_tenant_filter(self, tenant_id: str) -> Dict[str, Any]:
        """Build mandatory tenant filter for all queries."""
        return {
            "term": {
                "tenant_id": tenant_id
            }
        }
    
    async def search(self, tenant_id: str, query: str, filters: Dict[str, Any] = None, 
                    size: int = 20, from_: int = 0) -> Dict[str, Any]:
        """Perform BM25 search with tenant isolation."""
        if not self.client:
            return self._stub_search(tenant_id, query, filters, size, from_)
        
        try:
            # Always include tenant filter
            must_clauses = [self._build_tenant_filter(tenant_id)]
            
            # Add user-provided filters
            if filters:
                for field, value in filters.items():
                    if field in ["thread_id", "has_attachments"]:
                        must_clauses.append({"term": {field: value}})
                    elif field == "sent_at_range":
                        must_clauses.append({
                            "range": {
                                "sent_at": {
                                    "gte": value.get("from"),
                                    "lte": value.get("to")
                                }
                            }
                        })
                    elif field == "attachment_types":
                        if isinstance(value, list):
                            must_clauses.append({
                                "terms": {field: value}
                            })
                        else:
                            must_clauses.append({
                                "term": {field: value}
                            })
            
            # Build search query
            search_body = {
                "query": {
                    "bool": {
                        "must": [
                            {
                                "multi_match": {
                                    "query": query,
                                    "fields": ["subject^2", "content", "from_addr"],
                                    "type": "best_fields",
                                    "fuzziness": "AUTO"
                                }
                            }
                        ],
                        "filter": must_clauses
                    }
                },
                "highlight": {
                    "fields": {
                        "subject": {},
                        "content": {
                            "fragment_size": 150,
                            "number_of_fragments": 3
                        }
                    }
                },
                "sort": [
                    {"_score": {"order": "desc"}},
                    {"sent_at": {"order": "desc"}}
                ],
                "size": size,
                "from": from_
            }
            
            # Apply optimizations
            search_body = self.optimizer.optimize_query(query, filters)
            
            return await self._execute_search(tenant_id, search_body)
            
        except Exception as e:
            logger.error("BM25 search failed", 
                        tenant_id=tenant_id, 
                        query=query, 
                        error=str(e))
            return {
                "hits": {"total": {"value": 0}, "hits": []},
                "error": str(e)
            }
    
    async def _execute_search(self, tenant_id: str, search_body: Dict[str, Any]) -> Dict[str, Any]:
        """Execute search against OpenSearch."""
        index_name = self._get_index_name(tenant_id)
        
        try:
            response = await self.client.search(
                index=index_name,
                body=search_body
            )
            
            # Transform response to include citations
            transformed_response = self._transform_search_response(response, tenant_id)
            
            logger.info("BM25 search completed", 
                       tenant_id=tenant_id, 
                       total_hits=transformed_response["hits"]["total"]["value"])
            
            return transformed_response
            
        except Exception as e:
            logger.error("Failed to execute search", 
                        index_name=index_name, 
                        error=str(e))
            raise
    
    def _transform_search_response(self, response: Dict[str, Any], tenant_id: str) -> Dict[str, Any]:
        """Transform OpenSearch response to include citations."""
        transformed_hits = []
        
        for hit in response.get("hits", {}).get("hits", []):
            source = hit["_source"]
            transformed_hit = {
                "score": hit["_score"],
                "chunk_id": source.get("chunk_id"),
                "email_id": source.get("email_id"),
                "thread_id": source.get("thread_id"),
                "subject": source.get("subject"),
                "content": source.get("content"),
                "from_addr": source.get("from_addr"),
                "sent_at": source.get("sent_at"),
                "has_attachments": source.get("has_attachments"),
                "highlight": hit.get("highlight", {}),
                "citations": {
                    "chunk_id": source.get("chunk_id"),
                    "email_id": source.get("email_id"),
                    "attachment_id": source.get("attachment_id"),
                    "tenant_id": tenant_id
                }
            }
            transformed_hits.append(transformed_hit)
        
        return {
            "hits": {
                "total": response["hits"]["total"],
                "hits": transformed_hits
            },
            "aggregations": response.get("aggregations", {}),
            "search_type": "bm25"
        }
    
    def _stub_search(self, tenant_id: str, query: str, filters: Dict[str, Any] = None, 
                     size: int = 20, from_: int = 0) -> Dict[str, Any]:
        """Stub search implementation for testing."""
        logger.info("Using stub search", tenant_id=tenant_id, query=query)
        
        # Return mock results
        mock_hits = []
        for i in range(min(size, 5)):
            mock_hits.append({
                "score": 1.0 - (i * 0.1),
                "chunk_id": f"chunk-{i}",
                "email_id": f"email-{i}",
                "thread_id": f"thread-{i}",
                "subject": f"Mock subject {i}",
                "content": f"Mock content for query: {query}",
                "from_addr": f"user{i}@example.com",
                "sent_at": datetime.now().isoformat(),
                "has_attachments": i % 2 == 0,
                "highlight": {
                    "content": [f"Mock content for query: <em>{query}</em>"]
                },
                "citations": {
                    "chunk_id": f"chunk-{i}",
                    "email_id": f"email-{i}",
                    "attachment_id": None,
                    "tenant_id": tenant_id
                }
            })
        
        return {
            "hits": {
                "total": {"value": len(mock_hits), "relation": "eq"},
                "hits": mock_hits
            },
            "aggregations": {},
            "search_type": "bm25_stub"
        }
    
    async def index_document(self, tenant_id: str, chunk_id: str, document: Dict[str, Any]) -> bool:
        """Index a document in the BM25 index."""
        if not self.client:
            logger.info("Stub mode: document would be indexed", chunk_id=chunk_id)
            return True
        
        try:
            index_name = self._get_index_name(tenant_id)
            
            # Ensure index exists
            await self.index_manager.create_index(tenant_id)
            
            # Index document
            await self.client.index(
                index=index_name,
                id=chunk_id,
                body=document
            )
            
            logger.info("Document indexed successfully", 
                       chunk_id=chunk_id, 
                       tenant_id=tenant_id)
            return True
            
        except Exception as e:
            logger.error("Failed to index document", 
                        chunk_id=chunk_id, 
                        tenant_id=tenant_id, 
                        error=str(e))
            return False
    
    async def delete_document(self, tenant_id: str, chunk_id: str) -> bool:
        """Delete a document from the BM25 index."""
        if not self.client:
            logger.info("Stub mode: document would be deleted", chunk_id=chunk_id)
            return True
        
        try:
            index_name = self._get_index_name(tenant_id)
            
            await self.client.delete(
                index=index_name,
                id=chunk_id
            )
            
            logger.info("Document deleted successfully", 
                       chunk_id=chunk_id, 
                       tenant_id=tenant_id)
            return True
            
        except Exception as e:
            logger.error("Failed to delete document", 
                        chunk_id=chunk_id, 
                        tenant_id=tenant_id, 
                        error=str(e))
            return False
    
    async def get_health_status(self, tenant_id: str) -> Dict[str, Any]:
        """Get search service health status."""
        if not self.client:
            return {
                "status": "stub_mode",
                "tenant_id": tenant_id,
                "message": "OpenSearch client not available"
            }
        
        try:
            # Get cluster health
            cluster_health = await self.client.cluster.health()
            
            # Get index health
            index_health = await self.index_manager.get_health_status(tenant_id)
            
            return {
                "cluster_status": cluster_health.get("status"),
                "cluster_nodes": cluster_health.get("number_of_nodes"),
                "index_health": index_health,
                "service_status": "healthy"
            }
            
        except Exception as e:
            logger.error("Failed to get health status", error=str(e))
            return {
                "status": "error",
                "tenant_id": tenant_id,
                "error": str(e)
            }


class BM25SearchOptimizer:
    """Optimizes search performance and relevance."""
    
    def __init__(self):
        self.boost_weights = {
            "subject": 2.0,
            "content": 1.0,
            "from_addr": 1.5,
            "thread_id": 0.8
        }
    
    def optimize_query(self, query: str, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize search query for better relevance."""
        # Apply field boosting
        boosted_fields = []
        for field, boost in self.boost_weights.items():
            boosted_fields.append(f"{field}^{boost}")
        
        # Build optimized search body
        search_body = {
            "query": {
                "bool": {
                    "should": [
                        {
                            "multi_match": {
                                "query": query,
                                "fields": boosted_fields,
                                "type": "best_fields",
                                "fuzziness": "AUTO",
                                "minimum_should_match": "75%"
                            }
                        },
                        {
                            "multi_match": {
                                "query": query,
                                "fields": boosted_fields,
                                "type": "phrase_prefix",
                                "boost": 0.5
                            }
                        }
                    ]
                }
            },
            "aggs": {
                "thread_groups": {
                    "terms": {
                        "field": "thread_id",
                        "size": 10
                    }
                },
                "date_ranges": {
                    "date_range": {
                        "field": "sent_at",
                        "ranges": [
                            {"from": "now-1d", "to": "now"},
                            {"from": "now-7d", "to": "now-1d"},
                            {"from": "now-30d", "to": "now-7d"}
                        ]
                    }
                }
            }
        }
        
        return search_body


class BM25IndexManager:
    """Manages OpenSearch indices and health monitoring."""
    
    def __init__(self, host: str, index_prefix: str = "emails"):
        self.host = host
        self.index_prefix = index_prefix
        self.client = None
    
    async def initialize(self):
        """Initialize OpenSearch client."""
        try:
            from opensearchpy import AsyncOpenSearch
            self.client = AsyncOpenSearch(
                hosts=[self.host],
                http_auth=None,
                use_ssl=False,
                verify_certs=False,
                ssl_show_warn=False
            )
        except ImportError:
            logger.warning("OpenSearch client not available")
            self.client = None
    
    def _get_index_name(self, tenant_id: str) -> str:
        """Generate tenant-scoped index name."""
        return f"{self.index_prefix}_{tenant_id}"
    
    def _get_index_mapping(self) -> Dict[str, Any]:
        """Get index mapping configuration."""
        return {
            "mappings": {
                "properties": {
                    "tenant_id": {
                        "type": "keyword",
                        "index": True
                    },
                    "email_id": {
                        "type": "keyword",
                        "index": True
                    },
                    "chunk_id": {
                        "type": "keyword",
                        "index": True
                    },
                    "thread_id": {
                        "type": "keyword",
                        "index": True
                    },
                    "subject": {
                        "type": "text",
                        "analyzer": "standard",
                        "search_analyzer": "standard"
                    },
                    "content": {
                        "type": "text",
                        "analyzer": "standard",
                        "search_analyzer": "standard"
                    },
                    "from_addr": {
                        "type": "keyword",
                        "index": True
                    },
                    "to_addrs": {
                        "type": "keyword",
                        "index": True
                    },
                    "sent_at": {
                        "type": "date",
                        "format": "strict_date_optional_time||epoch_millis"
                    },
                    "has_attachments": {
                        "type": "boolean"
                    },
                    "attachment_types": {
                        "type": "keyword",
                        "index": True
                    },
                    "chunk_uid": {
                        "type": "keyword",
                        "index": True
                    },
                    "token_count": {
                        "type": "integer"
                    },
                    "created_at": {
                        "type": "date",
                        "format": "strict_date_optional_time||epoch_millis"
                    }
                }
            },
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 1,
                "analysis": {
                    "analyzer": {
                        "standard": {
                            "type": "standard",
                            "stopwords": "_english_"
                        }
                    }
                }
            }
        }
    
    async def create_index(self, tenant_id: str) -> bool:
        """Create tenant-specific index with proper mapping."""
        if not self.client:
            logger.info("Stub mode: index would be created", tenant_id=tenant_id)
            return True
        
        index_name = self._get_index_name(tenant_id)
        
        try:
            # Check if index exists
            if await self.client.indices.exists(index=index_name):
                logger.info("Index already exists", index_name=index_name)
                return True
            
            # Create index with mapping
            await self.client.indices.create(
                index=index_name,
                body=self._get_index_mapping()
            )
            
            logger.info("Index created successfully", index_name=index_name)
            return True
            
        except Exception as e:
            logger.error("Failed to create index", 
                        index_name=index_name, 
                        error=str(e))
            return False
    
    async def update_aliases(self, tenant_id: str, operation: str = "add") -> bool:
        """Update index aliases for zero-downtime operations."""
        if not self.client:
            logger.info("Stub mode: alias would be updated", tenant_id=tenant_id)
            return True
        
        index_name = self._get_index_name(tenant_id)
        alias_name = f"{self.index_prefix}_current_{tenant_id}"
        
        try:
            if operation == "add":
                await self.client.indices.put_alias(
                    index=index_name,
                    name=alias_name
                )
            elif operation == "remove":
                await self.client.indices.delete_alias(
                    index=index_name,
                    name=alias_name
                )
            
            logger.info("Alias updated", 
                       operation=operation, 
                       index_name=index_name, 
                       alias_name=alias_name)
            return True
            
        except Exception as e:
            logger.error("Failed to update alias", 
                        operation=operation, 
                        index_name=index_name, 
                        error=str(e))
            return False
    
    async def get_health_status(self, tenant_id: str) -> Dict[str, Any]:
        """Get index health status for monitoring."""
        if not self.client:
            return {
                "index_name": self._get_index_name(tenant_id),
                "tenant_id": tenant_id,
                "status": "stub_mode"
            }
        
        index_name = self._get_index_name(tenant_id)
        
        try:
            # Get index stats
            stats = await self.client.indices.stats(index=index_name)
            index_stats = stats["indices"].get(index_name, {})
            
            # Get index settings
            settings = await self.client.indices.get_settings(index=index_name)
            index_settings = settings["indices"].get(index_name, {})
            
            return {
                "index_name": index_name,
                "tenant_id": tenant_id,
                "document_count": index_stats.get("total", {}).get("docs", {}).get("count", 0),
                "storage_size": index_stats.get("total", {}).get("store", {}).get("size_in_bytes", 0),
                "shard_count": len(index_stats.get("shards", {})),
                "status": "green" if index_stats.get("health") == "green" else "yellow",
                "created_at": index_settings.get("settings", {}).get("index", {}).get("creation_date")
            }
            
        except Exception as e:
            logger.error("Failed to get health status", 
                        index_name=index_name, 
                        error=str(e))
            return {
                "index_name": index_name,
                "tenant_id": tenant_id,
                "status": "error",
                "error": str(e)
            }


class BM25IngestionHook:
    """Hooks into ingestion pipeline to index new content."""
    
    def __init__(self, search_service: BM25SearchService):
        self.search_service = search_service
    
    async def on_chunk_created(self, chunk, email) -> bool:
        """Index new chunk when created during ingestion."""
        try:
            # Prepare document for indexing
            doc = {
                "tenant_id": chunk.tenant_id,
                "email_id": chunk.email_id,
                "chunk_id": chunk.chunk_id,
                "thread_id": email.thread_id,
                "subject": email.subject or "",
                "content": chunk.content,
                "from_addr": email.from_addr or "",
                "to_addrs": email.to_addrs or [],
                "sent_at": email.sent_at.isoformat() if email.sent_at else None,
                "has_attachments": email.has_attachments,
                "attachment_types": self._extract_attachment_types(email),
                "chunk_uid": chunk.chunk_uid,
                "token_count": chunk.token_count,
                "created_at": chunk.created_at.isoformat()
            }
            
            # Index document
            success = await self.search_service.index_document(
                chunk.tenant_id, 
                chunk.chunk_id, 
                doc
            )
            
            if success:
                logger.info("Chunk indexed successfully", 
                           chunk_id=chunk.chunk_id, 
                           tenant_id=chunk.tenant_id)
            else:
                logger.error("Failed to index chunk", 
                           chunk_id=chunk.chunk_id, 
                           tenant_id=chunk.tenant_id)
            
            return success
            
        except Exception as e:
            logger.error("Error indexing chunk", 
                        chunk_id=chunk.chunk_id, 
                        error=str(e))
            return False
    
    def _extract_attachment_types(self, email) -> List[str]:
        """Extract attachment types for filtering."""
        if not hasattr(email, 'attachments') or not email.attachments:
            return []
        
        types = []
        for attachment in email.attachments:
            if hasattr(attachment, 'mimetype') and attachment.mimetype:
                main_type = attachment.mimetype.split('/')[0]
                if main_type not in types:
                    types.append(main_type)
        
        return types


# Factory function for creating BM25 search service
async def create_bm25_search_service(host: str = None, index_prefix: str = "emails") -> BM25SearchService:
    """Create and configure BM25 search service."""
    if not host:
        from ..config.manager import get_config
        config = get_config()
        host = config.get("search", {}).get("bm25", {}).get("host", "http://localhost:9200")
    
    service = BM25SearchService(host, index_prefix)
    await service.initialize()
    
    return service
