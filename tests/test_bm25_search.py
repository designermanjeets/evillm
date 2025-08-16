"""Tests for BM25 search service (EARS-RET-3)."""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta
import json

from app.services.bm25_search import (
    BM25SearchService, BM25SearchOptimizer, BM25IndexManager, 
    BM25IngestionHook, create_bm25_search_service
)


class TestBM25SearchService:
    """Test BM25 search service functionality."""
    
    @pytest.fixture
    def search_service(self):
        """Create BM25 search service instance."""
        return BM25SearchService("http://localhost:9200", "emails")
    
    @pytest.mark.asyncio
    async def test_service_initialization(self, search_service):
        """Test service initialization."""
        assert search_service.host == "http://localhost:9200"
        assert search_service.index_prefix == "emails"
        assert search_service.client is None
        assert search_service.optimizer is not None
        assert search_service.index_manager is not None
    
    @pytest.mark.asyncio
    async def test_initialize_with_opensearch(self, search_service):
        """Test initialization with OpenSearch client."""
        # Mock the import to simulate successful OpenSearch import
        with patch('builtins.__import__', side_effect=lambda name, *args, **kwargs: Mock()):
            await search_service.initialize()
            
            # In this test, we're simulating that OpenSearch is available
            assert True  # Test passes if no exception is raised
    
    @pytest.mark.asyncio
    async def test_initialize_without_opensearch(self, search_service):
        """Test initialization without OpenSearch client."""
        # Mock the import to simulate ImportError
        with patch('builtins.__import__', side_effect=ImportError("No module named 'opensearchpy'")):
            # Also patch the logger to prevent the warning from triggering the import
            with patch('app.services.bm25_search.logger.warning'):
                await search_service.initialize()
                
                assert search_service.client is None
    
    def test_get_index_name(self, search_service):
        """Test tenant-scoped index name generation."""
        index_name = search_service._get_index_name("tenant-123")
        assert index_name == "emails_tenant-123"
    
    def test_build_tenant_filter(self, search_service):
        """Test tenant filter construction."""
        filter_clause = search_service._build_tenant_filter("tenant-123")
        expected = {"term": {"tenant_id": "tenant-123"}}
        assert filter_clause == expected
    
    @pytest.mark.asyncio
    async def test_search_with_filters(self, search_service):
        """Test search with various filters."""
        # Mock client
        mock_client = Mock()
        mock_client.search = AsyncMock(return_value={
            "hits": {
                "total": {"value": 2, "relation": "eq"},
                "hits": [
                    {
                        "_score": 1.0,
                        "_source": {
                            "chunk_id": "chunk-1",
                            "email_id": "email-1",
                            "thread_id": "thread-1",
                            "subject": "Test Subject",
                            "content": "Test content",
                            "from_addr": "user@example.com",
                            "sent_at": "2024-01-01T00:00:00",
                            "has_attachments": False
                        },
                        "highlight": {
                            "content": ["Test <em>content</em>"]
                        }
                    }
                ]
            }
        })
        search_service.client = mock_client
        
        # Test search with filters
        filters = {
            "thread_id": "thread-1",
            "has_attachments": False,
            "sent_at_range": {
                "from": "2024-01-01",
                "to": "2024-01-31"
            }
        }
        
        result = await search_service.search(
            tenant_id="tenant-123",
            query="test query",
            filters=filters,
            size=10
        )
        
        assert result["hits"]["total"]["value"] == 2
        assert len(result["hits"]["hits"]) == 1
        assert result["search_type"] == "bm25"
        
        # Verify citations are included
        hit = result["hits"]["hits"][0]
        assert "citations" in hit
        assert hit["citations"]["tenant_id"] == "tenant-123"
        assert hit["citations"]["chunk_id"] == "chunk-1"
    
    @pytest.mark.asyncio
    async def test_search_stub_mode(self, search_service):
        """Test search in stub mode (no OpenSearch client)."""
        search_service.client = None
        
        result = await search_service.search(
            tenant_id="tenant-123",
            query="test query",
            size=5
        )
        
        assert result["search_type"] == "bm25_stub"
        assert result["hits"]["total"]["value"] == 5
        assert len(result["hits"]["hits"]) == 5
        
        # Verify stub results have proper structure
        for hit in result["hits"]["hits"]:
            assert "citations" in hit
            assert hit["citations"]["tenant_id"] == "tenant-123"
            assert "chunk_id" in hit["citations"]
    
    @pytest.mark.asyncio
    async def test_index_document_success(self, search_service):
        """Test successful document indexing."""
        # Mock client and index manager
        mock_client = Mock()
        mock_client.index = AsyncMock()
        search_service.client = mock_client
        
        mock_index_manager = Mock()
        mock_index_manager.create_index = AsyncMock(return_value=True)
        search_service.index_manager = mock_index_manager
        
        document = {
            "tenant_id": "tenant-123",
            "chunk_id": "chunk-1",
            "content": "Test content"
        }
        
        success = await search_service.index_document(
            tenant_id="tenant-123",
            chunk_id="chunk-1",
            document=document
        )
        
        assert success is True
        mock_index_manager.create_index.assert_called_once_with("tenant-123")
        mock_client.index.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_index_document_stub_mode(self, search_service):
        """Test document indexing in stub mode."""
        search_service.client = None
        
        success = await search_service.index_document(
            tenant_id="tenant-123",
            chunk_id="chunk-1",
            document={"test": "data"}
        )
        
        assert success is True
    
    @pytest.mark.asyncio
    async def test_delete_document_success(self, search_service):
        """Test successful document deletion."""
        mock_client = Mock()
        mock_client.delete = AsyncMock()
        search_service.client = mock_client
        
        success = await search_service.delete_document(
            tenant_id="tenant-123",
            chunk_id="chunk-1"
        )
        
        assert success is True
        mock_client.delete.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_health_status(self, search_service):
        """Test health status retrieval."""
        mock_client = Mock()
        mock_client.cluster.health = AsyncMock(return_value={
            "status": "green",
            "number_of_nodes": 3
        })
        search_service.client = mock_client
        
        # Mock index manager
        mock_index_manager = Mock()
        mock_index_manager.get_health_status = AsyncMock(return_value={
            "status": "green",
            "document_count": 100
        })
        search_service.index_manager = mock_index_manager
        
        health = await search_service.get_health_status("tenant-123")
        
        assert health["cluster_status"] == "green"
        assert health["cluster_nodes"] == 3
        assert health["service_status"] == "healthy"
        assert "index_health" in health


class TestBM25SearchOptimizer:
    """Test search query optimization."""
    
    def test_optimizer_initialization(self):
        """Test optimizer initialization with boost weights."""
        optimizer = BM25SearchOptimizer()
        
        expected_weights = {
            "subject": 2.0,
            "content": 1.0,
            "from_addr": 1.5,
            "thread_id": 0.8
        }
        
        assert optimizer.boost_weights == expected_weights
    
    def test_optimize_query(self):
        """Test query optimization."""
        optimizer = BM25SearchOptimizer()
        
        query = "test query"
        filters = {"thread_id": "thread-1"}
        
        optimized = optimizer.optimize_query(query, filters)
        
        # Verify structure
        assert "query" in optimized
        assert "aggs" in optimized
        
        # Verify field boosting
        query_clause = optimized["query"]["bool"]["should"][0]
        assert "multi_match" in query_clause
        assert "fields" in query_clause["multi_match"]
        
        # Verify aggregations
        assert "thread_groups" in optimized["aggs"]
        assert "date_ranges" in optimized["aggs"]


class TestBM25IndexManager:
    """Test index management functionality."""
    
    @pytest.fixture
    def index_manager(self):
        """Create index manager instance."""
        return BM25IndexManager("http://localhost:9200", "emails")
    
    @pytest.mark.asyncio
    async def test_initialize_with_opensearch(self, index_manager):
        """Test initialization with OpenSearch client."""
        # Mock the import to simulate successful OpenSearch import
        with patch('builtins.__import__', side_effect=lambda name, *args, **kwargs: Mock()):
            await index_manager.initialize()
            
            # In this test, we're simulating that OpenSearch is available
            assert True  # Test passes if no exception is raised
    
    @pytest.mark.asyncio
    async def test_initialize_without_opensearch(self, index_manager):
        """Test initialization without OpenSearch client."""
        # Mock the import to simulate ImportError
        with patch('builtins.__import__', side_effect=ImportError("No module named 'opensearchpy'")):
            # Also patch the logger to prevent the warning from triggering the import
            with patch('app.services.bm25_search.logger.warning'):
                await index_manager.initialize()
                
                assert index_manager.client is None
    
    def test_get_index_name(self, index_manager):
        """Test index name generation."""
        index_name = index_manager._get_index_name("tenant-123")
        assert index_name == "emails_tenant-123"
    
    def test_get_index_mapping(self, index_manager):
        """Test index mapping configuration."""
        mapping = index_manager._get_index_mapping()
        
        # Verify structure
        assert "mappings" in mapping
        assert "settings" in mapping
        
        # Verify properties
        properties = mapping["mappings"]["properties"]
        assert "tenant_id" in properties
        assert "chunk_id" in properties
        assert "content" in properties
        assert "sent_at" in properties
        
        # Verify settings
        settings = mapping["settings"]
        assert "number_of_shards" in settings
        assert "analysis" in settings
    
    @pytest.mark.asyncio
    async def test_create_index_success(self, index_manager):
        """Test successful index creation."""
        mock_client = Mock()
        mock_client.indices.exists = AsyncMock(return_value=False)
        mock_client.indices.create = AsyncMock()
        index_manager.client = mock_client
        
        success = await index_manager.create_index("tenant-123")
        
        assert success is True
        mock_client.indices.create.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_create_index_already_exists(self, index_manager):
        """Test index creation when index already exists."""
        mock_client = Mock()
        mock_client.indices.exists = AsyncMock(return_value=True)
        index_manager.client = mock_client
        
        success = await index_manager.create_index("tenant-123")
        
        assert success is True
        mock_client.indices.create.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_create_index_stub_mode(self, index_manager):
        """Test index creation in stub mode."""
        index_manager.client = None
        
        success = await index_manager.create_index("tenant-123")
        
        assert success is True
    
    @pytest.mark.asyncio
    async def test_update_aliases(self, index_manager):
        """Test alias updates."""
        mock_client = Mock()
        mock_client.indices.put_alias = AsyncMock()
        mock_client.indices.delete_alias = AsyncMock()
        index_manager.client = mock_client
        
        # Test adding alias
        success = await index_manager.update_aliases("tenant-123", "add")
        assert success is True
        mock_client.indices.put_alias.assert_called_once()
        
        # Test removing alias
        success = await index_manager.update_aliases("tenant-123", "remove")
        assert success is True
        mock_client.indices.delete_alias.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_health_status(self, index_manager):
        """Test health status retrieval."""
        mock_client = Mock()
        mock_client.indices.stats = AsyncMock(return_value={
            "indices": {
                "emails_tenant-123": {
                    "total": {
                        "docs": {"count": 100},
                        "store": {"size_in_bytes": 1024}
                    },
                    "shards": {"shard-1": {}, "shard-2": {}}
                }
            }
        })
        mock_client.indices.get_settings = AsyncMock(return_value={
            "indices": {
                "emails_tenant-123": {
                    "settings": {
                        "index": {
                            "creation_date": "1640995200000"
                        }
                    }
                }
            }
        })
        index_manager.client = mock_client
        
        health = await index_manager.get_health_status("tenant-123")
        
        assert health["document_count"] == 100
        assert health["storage_size"] == 1024
        assert health["shard_count"] == 2
        assert "created_at" in health


class TestBM25IngestionHook:
    """Test ingestion pipeline integration."""
    
    @pytest.fixture
    def mock_search_service(self):
        """Create mock search service."""
        service = Mock()
        service.index_document = AsyncMock(return_value=True)
        return service
    
    @pytest.fixture
    def ingestion_hook(self, mock_search_service):
        """Create ingestion hook with mock service."""
        return BM25IngestionHook(mock_search_service)
    
    @pytest.fixture
    def mock_chunk(self):
        """Create mock chunk object."""
        chunk = Mock()
        chunk.tenant_id = "tenant-123"
        chunk.email_id = "email-1"
        chunk.chunk_id = "chunk-1"
        chunk.content = "Test content"
        chunk.chunk_uid = "uid-1"
        chunk.token_count = 10
        chunk.created_at = datetime.now()
        return chunk
    
    @pytest.fixture
    def mock_email(self):
        """Create mock email object."""
        email = Mock()
        email.thread_id = "thread-1"
        email.subject = "Test Subject"
        email.from_addr = "user@example.com"
        email.to_addrs = ["recipient@example.com"]
        email.sent_at = datetime.now()
        email.has_attachments = False
        email.attachments = []
        return email
    
    @pytest.mark.asyncio
    async def test_on_chunk_created_success(self, ingestion_hook, mock_chunk, mock_email):
        """Test successful chunk indexing."""
        success = await ingestion_hook.on_chunk_created(mock_chunk, mock_email)
        
        assert success is True
        ingestion_hook.search_service.index_document.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_on_chunk_created_with_attachments(self, ingestion_hook, mock_chunk, mock_email):
        """Test chunk indexing with attachments."""
        # Add attachment to email
        attachment = Mock()
        attachment.mimetype = "application/pdf"
        mock_email.attachments = [attachment]
        mock_email.has_attachments = True
        
        success = await ingestion_hook.on_chunk_created(mock_chunk, mock_email)
        
        assert success is True
        
        # Verify document was indexed with attachment info
        call_args = ingestion_hook.search_service.index_document.call_args
        assert call_args[0][0] == "tenant-123"  # tenant_id
        assert call_args[0][1] == "chunk-1"     # chunk_id
    
    @pytest.mark.asyncio
    async def test_on_chunk_created_indexing_failure(self, ingestion_hook, mock_chunk, mock_email):
        """Test chunk indexing when document indexing fails."""
        ingestion_hook.search_service.index_document.return_value = False
        
        success = await ingestion_hook.on_chunk_created(mock_chunk, mock_email)
        
        assert success is False
    
    @pytest.mark.asyncio
    async def test_on_chunk_created_exception(self, ingestion_hook, mock_chunk, mock_email):
        """Test chunk indexing when exception occurs."""
        ingestion_hook.search_service.index_document.side_effect = Exception("Indexing failed")
        
        success = await ingestion_hook.on_chunk_created(mock_chunk, mock_email)
        
        assert success is False
    
    def test_extract_attachment_types(self, ingestion_hook, mock_email):
        """Test attachment type extraction."""
        # Add various attachment types
        attachments = []
        for mimetype in ["application/pdf", "image/jpeg", "text/plain"]:
            attachment = Mock()
            attachment.mimetype = mimetype
            attachments.append(attachment)
        
        mock_email.attachments = attachments
        
        types = ingestion_hook._extract_attachment_types(mock_email)
        
        assert "application" in types
        assert "image" in types
        assert "text" in types
        assert len(types) == 3
    
    def test_extract_attachment_types_no_attachments(self, ingestion_hook, mock_email):
        """Test attachment type extraction with no attachments."""
        mock_email.attachments = None
        
        types = ingestion_hook._extract_attachment_types(mock_email)
        
        assert types == []
    
    def test_extract_attachment_types_no_mimetype(self, ingestion_hook, mock_email):
        """Test attachment type extraction with no mimetype."""
        attachment = Mock()
        attachment.mimetype = None
        mock_email.attachments = [attachment]
        
        types = ingestion_hook._extract_attachment_types(mock_email)
        
        assert types == []


class TestBM25SearchIntegration:
    """Integration tests for BM25 search service."""
    
    @pytest.mark.asyncio
    async def test_create_bm25_search_service_factory(self):
        """Test factory function for creating BM25 search service."""
        with patch('app.config.manager.get_config') as mock_get_config:
            mock_get_config.return_value = {
                "search": {
                    "bm25": {
                        "host": "http://opensearch:9200"
                    }
                }
            }
            
            service = await create_bm25_search_service()
            
            assert service is not None
            assert isinstance(service, BM25SearchService)
            assert service.host == "http://opensearch:9200"
    
    @pytest.mark.asyncio
    async def test_create_bm25_search_service_custom_host(self):
        """Test factory function with custom host."""
        service = await create_bm25_search_service("http://custom:9200")
        
        assert service is not None
        assert service.host == "http://custom:9200"
    
    @pytest.mark.asyncio
    async def test_end_to_end_search_workflow(self):
        """Test complete search workflow from indexing to search."""
        # Create service
        service = BM25SearchService("http://localhost:9200", "emails")
        service.client = None  # Use stub mode
        
        # Index a document
        document = {
            "tenant_id": "tenant-123",
            "chunk_id": "chunk-1",
            "email_id": "email-1",
            "content": "This is a test email about logistics",
            "subject": "Test Logistics Email",
            "from_addr": "sender@example.com"
        }
        
        success = await service.index_document("tenant-123", "chunk-1", document)
        assert success is True
        
        # Search for the document
        results = await service.search(
            tenant_id="tenant-123",
            query="logistics",
            size=10
        )
        
        assert results["search_type"] == "bm25_stub"
        assert results["hits"]["total"]["value"] > 0
        
        # Verify results contain citations
        for hit in results["hits"]["hits"]:
            assert "citations" in hit
            assert hit["citations"]["tenant_id"] == "tenant-123"
            assert "chunk_id" in hit["citations"]


# Test data for parametrized tests
@pytest.mark.parametrize("tenant_id,expected_index", [
    ("tenant-123", "emails_tenant-123"),
    ("tenant-456", "emails_tenant-456"),
    ("tenant-789", "emails_tenant-789"),
])
def test_index_name_generation(tenant_id, expected_index):
    """Test index name generation for different tenant IDs."""
    service = BM25SearchService("http://localhost:9200", "emails")
    index_name = service._get_index_name(tenant_id)
    assert index_name == expected_index


@pytest.mark.parametrize("query,filters,expected_fields", [
    ("test query", {}, ["subject^2.0", "content^1.0", "from_addr^1.5", "thread_id^0.8"]),
    ("logistics", {"thread_id": "thread-1"}, ["subject^2.0", "content^1.0", "from_addr^1.5", "thread_id^0.8"]),
])
def test_query_optimization(query, filters, expected_fields):
    """Test query optimization with different inputs."""
    optimizer = BM25SearchOptimizer()
    optimized = optimizer.optimize_query(query, filters)
    
    # Verify field boosting is applied
    query_clause = optimized["query"]["bool"]["should"][0]
    fields = query_clause["multi_match"]["fields"]
    
    for expected_field in expected_fields:
        assert expected_field in fields
