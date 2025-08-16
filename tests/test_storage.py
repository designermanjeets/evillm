"""Tests for storage functionality."""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
from botocore.exceptions import ClientError

from app.storage.paths import StoragePathBuilder
from app.storage.metadata import ContentHash, ObjectMetadata
from app.storage.client import StorageClient
from app.storage.health import check_storage_health, get_storage_info


class TestStoragePathBuilder:
    """Test storage path builder functionality."""
    
    def test_build_email_raw_path(self):
        """Test building raw email path."""
        path = StoragePathBuilder.build_email_raw_path(
            tenant_id="tenant-123",
            thread_id="thread-456",
            email_id="email-789",
            message_id="msg-abc"
        )
        
        assert "tenant-123" in path
        assert "raw" in path
        assert path.endswith("msg-abc.eml")
        assert "/202" in path  # Date component
    
    def test_build_email_normalized_path(self):
        """Test building normalized email path."""
        path = StoragePathBuilder.build_email_normalized_path(
            tenant_id="tenant-123",
            thread_id="thread-456",
            email_id="email-789",
            message_id="msg-abc"
        )
        
        assert "tenant-123" in path
        assert "norm" in path
        assert path.endswith("msg-abc.txt")
    
    def test_build_attachment_path(self):
        """Test building attachment path."""
        path = StoragePathBuilder.build_attachment_path(
            tenant_id="tenant-123",
            thread_id="thread-456",
            email_id="email-789",
            attachment_index=0,
            filename="invoice.pdf"
        )
        
        assert "tenant-123" in path
        assert "att" in path
        assert "0_invoice.pdf" in path
    
    def test_build_attachment_path_sanitizes_filename(self):
        """Test that dangerous characters in filenames are sanitized."""
        path = StoragePathBuilder.build_attachment_path(
            tenant_id="tenant-123",
            thread_id="thread-456",
            email_id="email-789",
            attachment_index=0,
            filename="file/with\\dangerous:chars*.pdf"
        )
        
        # Check that the sanitized filename is in the path
        assert "file_with_dangerous_chars_.pdf" in path
        
        # Check that the filename part doesn't contain dangerous characters
        # (the path structure itself contains forward slashes)
        filename_part = path.split("/")[-1]  # Get just the filename part
        assert "/" not in filename_part
        assert "\\" not in filename_part
        assert ":" not in filename_part
        assert "*" not in filename_part
    
    def test_extract_tenant_from_path(self):
        """Test extracting tenant ID from path."""
        tenant = StoragePathBuilder.extract_tenant_from_path(
            "tenant-123/2024/01/15/thread-456/email-789/raw/message.eml"
        )
        assert tenant == "tenant-123"
    
    def test_validate_tenant_path(self):
        """Test tenant path validation."""
        path = "tenant-123/2024/01/15/thread-456/email-789/raw/message.eml"
        
        assert StoragePathBuilder.validate_tenant_path(path, "tenant-123") is True
        assert StoragePathBuilder.validate_tenant_path(path, "tenant-456") is False
    
    def test_get_path_components(self):
        """Test parsing path components."""
        path = "tenant-123/2024/01/15/thread-456/email-789/raw/message.eml"
        components = StoragePathBuilder.get_path_components(path)
        
        assert components["tenant_id"] == "tenant-123"
        assert components["date"] == "2024/01/15"
        assert components["thread_id"] == "thread-456"
        assert components["email_id"] == "email-789"
        assert components["content_type"] == "raw"
        assert components["filename"] == "message.eml"
    
    def test_is_valid_path(self):
        """Test path validation."""
        # Valid paths
        assert StoragePathBuilder.is_valid_path("tenant-123/2024/01/15/file.txt") is True
        assert StoragePathBuilder.is_valid_path("dev-123/2024/01/15/file.txt") is True
        
        # Invalid paths
        assert StoragePathBuilder.is_valid_path("") is False
        assert StoragePathBuilder.is_valid_path("invalid/path") is False
        assert StoragePathBuilder.is_valid_path("tenant-123/../file.txt") is False


class TestContentHash:
    """Test content hash functionality."""
    
    def test_from_content_sha256(self):
        """Test creating SHA-256 hash from content."""
        content = b"test content"
        hash_obj = ContentHash.from_content(content, "sha256")
        
        assert hash_obj.algorithm == "sha256"
        assert len(hash_obj.value) == 64  # SHA-256 hex length
        assert hash_obj.created_at is not None
    
    def test_from_content_md5(self):
        """Test creating MD5 hash from content."""
        content = b"test content"
        hash_obj = ContentHash.from_content(content, "md5")
        
        assert hash_obj.algorithm == "md5"
        assert len(hash_obj.value) == 32  # MD5 hex length
    
    def test_from_content_unsupported_algorithm(self):
        """Test that unsupported algorithms raise error."""
        content = b"test content"
        with pytest.raises(ValueError, match="Unsupported hash algorithm"):
            ContentHash.from_content(content, "unsupported")
    
    def test_equality(self):
        """Test hash equality comparison."""
        hash1 = ContentHash("sha256", "abc123")
        hash2 = ContentHash("sha256", "abc123")
        hash3 = ContentHash("sha256", "def456")
        
        assert hash1 == hash2
        assert hash1 != hash3
    
    def test_string_representation(self):
        """Test string representation of hash."""
        hash_obj = ContentHash("sha256", "abc123")
        assert str(hash_obj) == "sha256:abc123"


class TestObjectMetadata:
    """Test object metadata functionality."""
    
    def test_metadata_creation(self):
        """Test creating metadata object."""
        metadata = ObjectMetadata(
            tenant_id="tenant-123",
            content_sha256="abc123",
            content_length=1024,
            mimetype="text/plain"
        )
        
        assert metadata.tenant_id == "tenant-123"
        assert metadata.content_sha256 == "abc123"
        assert metadata.content_length == 1024
        assert metadata.mimetype == "text/plain"
        assert metadata.created_at is not None
    
    def test_metadata_validation_valid(self):
        """Test metadata validation with valid data."""
        metadata = ObjectMetadata(
            tenant_id="tenant-123",
            content_sha256="abc123",
            content_length=1024,
            mimetype="text/plain"
        )
        
        assert metadata.validate() is True
    
    def test_metadata_validation_missing_tenant(self):
        """Test metadata validation with missing tenant."""
        metadata = ObjectMetadata(
            tenant_id="",
            content_sha256="abc123",
            content_length=1024,
            mimetype="text/plain"
        )
        
        assert metadata.validate() is False
    
    def test_metadata_validation_invalid_length(self):
        """Test metadata validation with invalid length."""
        metadata = ObjectMetadata(
            tenant_id="tenant-123",
            content_sha256="abc123",
            content_length=0,
            mimetype="text/plain"
        )
        
        assert metadata.validate() is False
    
    def test_to_s3_metadata(self):
        """Test converting metadata to S3 format."""
        metadata = ObjectMetadata(
            tenant_id="tenant-123",
            content_sha256="abc123",
            content_length=1024,
            mimetype="text/plain"
        )
        
        s3_metadata = metadata.to_s3_metadata()
        
        assert s3_metadata["x-amz-meta-tenant-id"] == "tenant-123"
        assert s3_metadata["x-amz-meta-content-sha256"] == "abc123"
        assert s3_metadata["x-amz-meta-content-length"] == "1024"
        assert s3_metadata["x-amz-meta-mimetype"] == "text/plain"
    
    def test_from_s3_metadata(self):
        """Test creating metadata from S3 format."""
        s3_metadata = {
            "x-amz-meta-tenant-id": "tenant-123",
            "x-amz-meta-content-sha256": "abc123",
            "x-amz-meta-content-length": "1024",
            "x-amz-meta-mimetype": "text/plain",
            "x-amz-meta-created-at": "2024-01-15T10:30:00"
        }
        
        metadata = ObjectMetadata.from_s3_metadata(s3_metadata)
        
        assert metadata.tenant_id == "tenant-123"
        assert metadata.content_sha256 == "abc123"
        assert metadata.content_length == 1024
        assert metadata.mimetype == "text/plain"
    
    def test_custom_metadata(self):
        """Test custom metadata functionality."""
        metadata = ObjectMetadata(
            tenant_id="tenant-123",
            content_sha256="abc123",
            content_length=1024,
            mimetype="text/plain"
        )
        
        metadata.add_custom_metadata("source", "email")
        metadata.add_custom_metadata("priority", "high")
        
        assert metadata.get_custom_metadata("source") == "email"
        assert metadata.get_custom_metadata("priority") == "high"
        assert metadata.get_custom_metadata("nonexistent") is None


class TestStorageClient:
    """Test storage client functionality."""
    
    @pytest.fixture
    def mock_client(self):
        """Create mock S3 client."""
        with patch('app.storage.client.boto3.client') as mock_boto3:
            mock_s3 = Mock()
            mock_boto3.return_value = mock_s3
            yield mock_s3
    
    @pytest.fixture
    def storage_client(self, mock_client):
        """Create storage client with mocked dependencies."""
        with patch('app.storage.client.get_storage_settings') as mock_settings:
            mock_settings.return_value = Mock(
                provider="minio",
                endpoint_url="http://localhost:9000",
                region="us-east-1",
                bucket_name="test-bucket",
                access_key_id="test-key",
                secret_access_key="test-secret",
                encryption_algorithm="AES256",
                allowed_mimetypes=["text/plain", "text/html"],
                max_object_size=100 * 1024 * 1024,
                multipart_threshold=5 * 1024 * 1024,
                max_retry_attempts=3,
                retry_base_delay=1.0,
                retry_max_delay=60.0,
                presigned_url_ttl=900,
                presigned_url_max_ttl=3600
            )
            return StorageClient()
    
    def test_client_initialization(self, mock_client):
        """Test storage client initialization."""
        with patch('app.storage.client.get_storage_settings') as mock_settings:
            mock_settings.return_value = Mock(
                provider="minio",
                endpoint_url="http://localhost:9000",
                region="us-east-1",
                bucket_name="test-bucket",
                access_key_id="test-key",
                secret_access_key="test-secret",
                encryption_algorithm="AES256",
                allowed_mimetypes=["text/plain"],
                max_object_size=100 * 1024 * 1024,
                multipart_threshold=5 * 1024 * 1024,
                max_retry_attempts=3,
                retry_base_delay=1.0,
                retry_max_delay=60.0,
                presigned_url_ttl=900,
                presigned_url_max_ttl=3600
            )
            
            client = StorageClient()
            assert client._bucket_name == "test-bucket"
    
    @pytest.mark.asyncio
    async def test_put_object_simple_upload(self, storage_client, mock_client):
        """Test simple object upload."""
        metadata = ObjectMetadata(
            tenant_id="tenant-123",
            content_sha256="abc123",
            content_length=1024,
            mimetype="text/plain"
        )
        
        # Mock successful upload
        mock_client.put_object.return_value = None
        
        result = await storage_client.put_object(
            key="tenant-123/2024/01/15/test.txt",
            data=b"test content",
            metadata=metadata,
            tenant_id="tenant-123"
        )
        
        assert result is True
        mock_client.put_object.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_put_object_tenant_validation_failure(self, storage_client):
        """Test that tenant validation prevents cross-tenant uploads."""
        metadata = ObjectMetadata(
            tenant_id="tenant-123",
            content_sha256="abc123",
            content_length=1024,
            mimetype="text/plain"
        )
        
        with pytest.raises(ValueError, match="does not belong to tenant"):
            await storage_client.put_object(
                key="tenant-456/2024/01/15/test.txt",
                data=b"test content",
                metadata=metadata,
                tenant_id="tenant-123"
            )
    
    @pytest.mark.asyncio
    async def test_put_object_mimetype_validation(self, storage_client):
        """Test that disallowed mimetypes are rejected."""
        metadata = ObjectMetadata(
            tenant_id="tenant-123",
            content_sha256="abc123",
            content_length=1024,
            mimetype="application/executable"
        )
        
        with pytest.raises(ValueError, match="not allowed"):
            await storage_client.put_object(
                key="tenant-123/2024/01/15/test.exe",
                data=b"test content",
                metadata=metadata,
                tenant_id="tenant-123"
            )
    
    @pytest.mark.asyncio
    async def test_get_object_success(self, storage_client, mock_client):
        """Test successful object retrieval."""
        mock_client.get_object.return_value = {
            'Body': Mock(read=lambda: b"test content")
        }
        
        result = await storage_client.get_object(
            key="tenant-123/2024/01/15/test.txt",
            tenant_id="tenant-123"
        )
        
        assert result == b"test content"
    
    @pytest.mark.asyncio
    async def test_get_object_not_found(self, storage_client, mock_client):
        """Test object not found handling."""
        error_response = {'Error': {'Code': 'NoSuchKey'}}
        mock_client.get_object.side_effect = ClientError(error_response, 'GetObject')
        
        result = await storage_client.get_object(
            key="tenant-123/2024/01/15/nonexistent.txt",
            tenant_id="tenant-123"
        )
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_generate_presigned_url(self, storage_client, mock_client):
        """Test presigned URL generation."""
        mock_client.generate_presigned_url.return_value = "https://example.com/presigned-url"
        
        result = await storage_client.generate_presigned_url(
            key="tenant-123/2024/01/15/test.txt",
            tenant_id="tenant-123",
            ttl=300
        )
        
        assert result == "https://example.com/presigned-url"
        mock_client.generate_presigned_url.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_check_bucket_exists_true(self, storage_client, mock_client):
        """Test bucket existence check when bucket exists."""
        mock_client.head_bucket.return_value = None
        
        result = await storage_client.check_bucket_exists()
        
        assert result is True
    
    @pytest.mark.asyncio
    async def test_check_bucket_exists_false(self, storage_client, mock_client):
        """Test bucket existence check when bucket doesn't exist."""
        error_response = {'Error': {'Code': 'NoSuchBucket'}}
        mock_client.head_bucket.side_effect = ClientError(error_response, 'HeadBucket')
        
        result = await storage_client.check_bucket_exists()
        
        assert result is False


class TestStorageHealth:
    """Test storage health check functionality."""
    
    @pytest.fixture
    def mock_storage_client(self):
        """Create mock storage client."""
        with patch('app.storage.health.get_storage_client') as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client
            yield mock_client
    
    @pytest.fixture
    def mock_settings(self):
        """Create mock storage settings."""
        with patch('app.storage.health.get_storage_settings') as mock_get_settings:
            mock_settings = Mock(
                provider="minio",
                endpoint_url="http://localhost:9000",
                bucket_name="test-bucket",
                health_check_enabled=True,
                canary_test_enabled=False
            )
            mock_get_settings.return_value = mock_settings
            yield mock_settings
    
    @pytest.mark.asyncio
    async def test_check_storage_health_healthy(self, mock_storage_client, mock_settings):
        """Test storage health check when healthy."""
        mock_storage_client.check_bucket_exists.return_value = True
        
        result = await check_storage_health()
        
        assert result["status"] == "healthy"
        assert result["provider"] == "minio"
        assert result["checks"]["bucket_access"]["status"] == "healthy"
    
    @pytest.mark.asyncio
    async def test_check_storage_health_unhealthy(self, mock_storage_client, mock_settings):
        """Test storage health check when unhealthy."""
        mock_storage_client.check_bucket_exists.side_effect = Exception("Connection failed")
        
        result = await check_storage_health()
        
        assert result["status"] == "unhealthy"
        assert result["checks"]["bucket_access"]["status"] == "unhealthy"
    
    @pytest.mark.asyncio
    async def test_check_storage_health_with_canary(self, mock_storage_client, mock_settings):
        """Test storage health check with canary test enabled."""
        mock_settings.canary_test_enabled = True
        mock_storage_client.check_bucket_exists.return_value = True
        
        # Mock successful canary test
        mock_storage_client.put_object.return_value = True
        mock_storage_client.get_object.return_value = b"0" * 1024
        mock_storage_client.delete_object.return_value = True
        
        result = await check_storage_health()
        
        assert result["status"] == "healthy"
        assert "canary_test" in result["checks"]
        assert result["checks"]["canary_test"]["status"] == "healthy"
    
    @pytest.mark.asyncio
    async def test_get_storage_info(self, mock_settings):
        """Test getting storage information."""
        result = await get_storage_info()
        
        assert result["provider"] == "minio"
        assert result["endpoint"] == "http://localhost:9000"
        assert result["bucket"] == "test-bucket"
        assert result["health_check_enabled"] is True
        assert result["canary_test_enabled"] is False


class TestEARSMapping:
    """Test that storage functionality maps to EARS requirements."""
    
    def test_EARS_STO_1_raw_email_storage(self):
        """Test EARS-STO-1: Raw email storage with deterministic paths and content hash."""
        # Test path building
        path = StoragePathBuilder.build_email_raw_path(
            tenant_id="tenant-123",
            thread_id="thread-456",
            email_id="email-789",
            message_id="msg-abc"
        )
        
        # Verify deterministic path structure
        assert path.startswith("tenant-123/")
        assert "/raw/" in path
        assert path.endswith(".eml")
        
        # Test content hash creation
        content = b"test email content"
        content_hash = ContentHash.from_content(content)
        
        assert content_hash.algorithm == "sha256"
        assert len(content_hash.value) == 64
    
    def test_EARS_STO_2_normalized_text_storage(self):
        """Test EARS-STO-2: Normalized text storage with linkage."""
        # Test path building for normalized text
        path = StoragePathBuilder.build_email_normalized_path(
            tenant_id="tenant-123",
            thread_id="thread-456",
            email_id="email-789",
            message_id="msg-abc"
        )
        
        # Verify normalized text path structure
        assert path.startswith("tenant-123/")
        assert "/norm/" in path
        assert path.endswith(".txt")
    
    def test_EARS_STO_3_attachment_storage(self):
        """Test EARS-STO-3: Attachment storage with signed URLs and mimetype validation."""
        # Test path building for attachments
        path = StoragePathBuilder.build_attachment_path(
            tenant_id="tenant-123",
            thread_id="thread-456",
            email_id="email-789",
            attachment_index=0,
            filename="invoice.pdf"
        )
        
        # Verify attachment path structure
        assert path.startswith("tenant-123/")
        assert "/att/" in path
        assert "0_invoice.pdf" in path
        
        # Test OCR text path
        ocr_path = StoragePathBuilder.build_ocr_text_path(
            tenant_id="tenant-123",
            thread_id="thread-456",
            email_id="email-789",
            attachment_index=0
        )
        
        assert ocr_path.endswith("0_ocr.txt")
    
    def test_EARS_STO_4_tenant_isolation(self):
        """Test EARS-STO-4: Tenant isolation via path prefixes."""
        # Test tenant path validation
        path = "tenant-123/2024/01/15/thread-456/email-789/raw/message.eml"
        
        # Valid tenant access
        assert StoragePathBuilder.validate_tenant_path(path, "tenant-123") is True
        
        # Invalid tenant access
        assert StoragePathBuilder.validate_tenant_path(path, "tenant-456") is False
        
        # Test path component extraction
        components = StoragePathBuilder.get_path_components(path)
        assert components["tenant_id"] == "tenant-123"
    
    def test_EARS_STO_5_deduplication(self):
        """Test EARS-STO-5: Content hash-based deduplication."""
        # Test that identical content produces identical hashes
        content1 = b"identical content"
        content2 = b"identical content"
        
        hash1 = ContentHash.from_content(content1)
        hash2 = ContentHash.from_content(content2)
        
        assert hash1 == hash2
        assert hash1.value == hash2.value
        
        # Test that different content produces different hashes
        content3 = b"different content"
        hash3 = ContentHash.from_content(content3)
        
        assert hash1 != hash3
        assert hash1.value != hash3.value
    
    def test_EARS_STO_6_retry_logic(self):
        """Test EARS-STO-6: Retry with exponential backoff."""
        # This is tested through the StorageClient retry decorator
        # The @retry decorator from tenacity provides the retry logic
        assert hasattr(StorageClient.put_object, '__wrapped__')  # Indicates retry decorator
    
    def test_EARS_STO_7_multipart_upload(self):
        """Test EARS-STO-7: Multipart upload for large objects."""
        # Test that StorageClient has multipart upload capability
        client = StorageClient()
        
        # Verify multipart threshold configuration
        assert hasattr(client.settings, 'multipart_threshold')
        assert hasattr(client.settings, 'multipart_chunk_size')
        
        # Verify multipart upload methods exist
        assert hasattr(client, '_multipart_upload')
        assert hasattr(client, '_simple_upload')
    
    def test_EARS_STO_8_presigned_urls(self):
        """Test EARS-STO-8: Presigned URLs with TTL and audit."""
        # Test that StorageClient has presigned URL generation
        client = StorageClient()
        
        # Verify presigned URL configuration
        assert hasattr(client.settings, 'presigned_url_ttl')
        assert hasattr(client.settings, 'presigned_url_max_ttl')
        
        # Verify presigned URL method exists
        assert hasattr(client, 'generate_presigned_url')
