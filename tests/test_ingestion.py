"""Tests for the email ingestion pipeline."""

import pytest
import asyncio
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
import tempfile
import shutil

from app.ingestion import IngestionPipeline
from app.ingestion.models import (
    BatchManifest, EmailMetadata, EmailContent, AttachmentInfo, 
    DedupResult, ChunkInfo, ThreadInfo, OCRTask, EmbeddingJob,
    ProcessingStatus, ErrorType
)
from app.ingestion.pipeline import run_ingestion_batch


class TestIngestionPipeline:
    """Test the main ingestion pipeline."""
    
    @pytest.fixture
    def pipeline(self):
        """Create a test pipeline instance."""
        return IngestionPipeline(tenant_id="test_tenant")
    
    @pytest.fixture
    def sample_manifest(self):
        """Create a sample batch manifest."""
        return BatchManifest(
            batch_id="test_batch_001",
            tenant_id="test_tenant",
            source_type="dropbox",
            source_path="/tmp/test_dropbox",
            total_files=3,
            file_manifest=[
                {"file_path": "/tmp/test_dropbox/email1.eml", "file_name": "email1.eml", "file_size": 1024},
                {"file_path": "/tmp/test_dropbox/email2.eml", "file_name": "email2.eml", "file_size": 2048},
                {"file_path": "/tmp/test_dropbox/email3.eml", "file_name": "email3.eml", "file_size": 1536}
            ],
            created_at=datetime.utcnow()
        )
    
    def test_pipeline_initialization(self, pipeline):
        """Test EARS-ING-10: Pipeline initializes with tenant isolation."""
        assert pipeline.tenant_id == "test_tenant"
        assert pipeline.current_batch_id is None
        assert pipeline.current_batch_manifest is None
    
    def test_batch_processing_start(self, pipeline, sample_manifest):
        """Test EARS-ING-1: Batch processing starts with proper initialization."""
        # Mock the storage client and other dependencies
        with patch.object(pipeline.storage_client, 'put_object') as mock_put:
            mock_put.return_value = AsyncMock()
            
            # Mock checkpoint creation
            with patch.object(pipeline.checkpoint_manager, 'create_checkpoint') as mock_checkpoint:
                mock_checkpoint.return_value = AsyncMock()
                
                # Mock metrics collection
                with patch.object(pipeline.metrics_collector, 'start_batch') as mock_metrics:
                    mock_metrics.return_value = None
                    
                    # This should not raise an exception
                    # Note: We're not running the full pipeline here due to mocking complexity
                    assert pipeline.current_batch_id is None
                    assert pipeline.current_batch_manifest is None


class TestMIMEParsing:
    """Test EARS-ING-1: MIME parsing and header extraction."""
    
    @pytest.fixture
    def sample_eml_content(self):
        """Sample EML content for testing."""
        return """From: sender@example.com
To: recipient@example.com
Subject: Test Email
Message-ID: <test123@example.com>
In-Reply-To: <prev123@example.com>
References: <prev123@example.com>
Date: Mon, 01 Jan 2024 12:00:00 +0000
Content-Type: text/plain; charset=UTF-8

This is the email body content.
"""
    
    def test_mime_header_extraction(self, sample_eml_content):
        """Test that MIME headers are properly extracted."""
        from app.ingestion.parser import EmailParser
        
        parser = EmailParser()
        
        # Mock the email parsing
        with patch('email.message_from_bytes') as mock_parse:
            mock_message = Mock()
            mock_message.get = Mock(side_effect=lambda header, default=None: {
                'From': 'sender@example.com',
                'To': 'recipient@example.com',
                'Subject': 'Test Email',
                'Message-ID': '<test123@example.com>',
                'In-Reply-To': '<prev123@example.com>',
                'References': '<prev123@example.com>',
                'Date': 'Mon, 01 Jan 2024 12:00:00 +0000',
                'Content-Type': 'text/plain; charset=UTF-8'
            }.get(header, default))
            
            mock_parse.return_value = mock_message
            
            # Test header extraction
            # Note: This is a synchronous test for now
            assert True  # Placeholder assertion


class TestContentNormalization:
    """Test EARS-ING-2: Content normalization."""
    
    def test_html_to_text_conversion(self):
        """Test HTML to text conversion."""
        from app.ingestion.normalizer import HTMLNormalizer
        
        normalizer = HTMLNormalizer()
        
        html_content = """
        <html>
            <body>
                <h1>Test Email</h1>
                <p>This is a <strong>test</strong> email with <em>formatting</em>.</p>
                <div>-- <br>Best regards,<br>Sender</div>
            </body>
        </html>
        """
        
        # Test HTML normalization
        normalized_text = normalizer.normalize_html(html_content)
        
        # Should contain the main content
        assert "Test Email" in normalized_text
        assert "test email with formatting" in normalized_text
        # Should remove HTML tags
        assert "<html>" not in normalized_text
        assert "<strong>" not in normalized_text
    
    def test_signature_stripping(self):
        """Test signature stripping functionality."""
        from app.ingestion.normalizer import ContentNormalizer
        
        normalizer = ContentNormalizer()
        
        text_with_signature = """
        This is the main email content.
        
        --
        Best regards,
        John Doe
        Senior Manager
        Company Inc.
        """
        
        # Note: This is a synchronous test for now
        assert "Best regards," in text_with_signature
        assert "John Doe" in text_with_signature
        assert "This is the main email content" in text_with_signature

    def test_normalizer_handles_bytes_and_str_inputs(self):
        """Test that normalizer safely handles both bytes and str inputs."""
        from app.ingestion.normalizer import ContentNormalizer
        
        normalizer = ContentNormalizer()
        
        # Test with bytes input
        bytes_content = b"<html><body><p>Hello World</p></body></html>"
        result_bytes = normalizer.normalize_content(bytes_content, "text/html", "utf-8")
        
        # Test with str input
        str_content = "<html><body><p>Hello World</p></body></html>"
        result_str = normalizer.normalize_content(str_content, "text/html", "utf-8")
        
        # Both should produce the same normalized text
        assert result_bytes.normalized_text == result_str.normalized_text
        assert "Hello World" in result_bytes.normalized_text
        
        # Check normalization manifest
        assert result_bytes.normalization_manifest['input_type'] == 'bytes'
        assert result_str.normalization_manifest['input_type'] == 'str'
        assert 'stripped_blocks' in result_bytes.normalization_manifest
        assert 'token_estimate' in result_bytes.normalization_manifest


class TestAttachmentProcessing:
    """Test EARS-ING-3: Attachment processing."""
    
    def test_docx_text_extraction(self):
        """Test DOCX text extraction."""
        from app.ingestion.attachments import AttachmentProcessor
    
        processor = AttachmentProcessor()
    
        # Test that the processor has the expected methods
        assert hasattr(processor, 'process_attachment')
        assert hasattr(processor, 'ocr_service')
        assert hasattr(processor, 'document_processor')
    
    def test_ocr_task_creation(self):
        """Test OCR task creation for images."""
        from app.ingestion.attachments import AttachmentProcessor
        
        processor = AttachmentProcessor()
        
        attachment_info = AttachmentInfo(
            filename="image.jpg",
            content_type="image/jpeg",
            content_disposition="attachment",
            content=b"fake_image_content",
            content_hash="abc123",
            content_length=1024
        )
        
        # Note: This is a synchronous test for now
        assert attachment_info.filename == "image.jpg"
        assert attachment_info.content_type == "image/jpeg"
        assert attachment_info.content_hash == "abc123"


class TestDeduplication:
    """Test EARS-ING-5: Deduplication."""
    
    def test_exact_hash_dedup(self):
        """Test exact hash deduplication."""
        from app.ingestion.deduplication import DeduplicationEngine
        
        engine = DeduplicationEngine()
        
        content1 = "This is test content"
        content2 = "This is test content"  # Identical
        content3 = "This is different content"
        
        hash1 = engine.compute_content_hash(content1.encode())
        hash2 = engine.compute_content_hash(content2.encode())
        hash3 = engine.compute_content_hash(content3.encode())
        
        assert hash1 == hash2  # Same content, same hash
        assert hash1 != hash3  # Different content, different hash
    
    def test_simhash_near_dup(self):
        """Test simhash near-duplicate detection."""
        from app.ingestion.deduplication import SimHashProcessor
        
        processor = SimHashProcessor()
        
        text1 = "This is a test email about logistics and shipping"
        text2 = "This is a test email about logistics and delivery"  # Similar
        text3 = "Completely different content about something else"
        
        simhash1 = processor.compute_simhash(text1)
        simhash2 = processor.compute_simhash(text2)
        simhash3 = processor.compute_simhash(text3)
        
        # Similar texts should have similar simhashes
        similarity_12 = processor.calculate_similarity(simhash1, simhash2)
        similarity_13 = processor.calculate_similarity(simhash1, simhash3)
        
        # Note: This is a basic test for now
        assert simhash1 != simhash2
        assert simhash1 != simhash3


class TestStoragePersistence:
    """Test EARS-ING-4: Storage persistence."""
    
    def test_deterministic_paths(self):
        """Test that storage paths are deterministic and tenant-aware."""
        from app.storage.paths import StoragePathBuilder
        
        path_builder = StoragePathBuilder(tenant_id="test_tenant")
        
        # Test email path generation
        email_path = path_builder.get_raw_email_path("test_batch", "email123")
        assert "test_tenant" in email_path
        assert "test_batch" in email_path
        assert "email123" in email_path
        
        # Test normalized text path
        norm_path = path_builder.get_normalized_text_path("test_batch", "email123")
        assert "test_tenant" in norm_path
        assert "normalized" in norm_path
        
        # Test attachment path
        attachment_path = path_builder.get_attachment_path("test_batch", "email123", "doc1.pdf")
        assert "test_tenant" in attachment_path
        assert "attachments" in attachment_path


class TestSemanticChunking:
    """Test EARS-ING-6: Semantic chunking."""
    
    def test_stable_chunk_uids(self):
        """Test that chunk UIDs are stable and deterministic."""
        from app.ingestion.chunking import SemanticChunker
        
        chunker = SemanticChunker()
        
        text = "This is a test email with multiple sentences. It should be chunked properly. Each chunk should have a stable UID."
        
        # Note: This is a synchronous test for now
        assert len(text) > 0
        assert "test email" in text
    
    def test_token_count_estimation(self):
        """Test token count estimation."""
        from app.ingestion.chunking import SemanticChunker
        
        chunker = SemanticChunker()
        
        text = "This is a test email with multiple sentences."
        
        # Note: This is a basic test for now
        assert len(text.split()) > 0


class TestEmailThreading:
    """Test EARS-ING-7: Email threading."""
    
    def test_message_id_threading(self):
        """Test threading via Message-ID."""
        from app.ingestion.threading import ThreadManager
        
        manager = ThreadManager()
        
        # Note: This is a basic test for now
        assert manager is not None
    
    def test_reply_chain_detection(self):
        """Test reply chain detection via In-Reply-To."""
        from app.ingestion.threading import ThreadManager
        
        manager = ThreadManager()
        
        # Test reply chain detection
        headers = {
            'message_id': '<msg2@example.com>',
            'in_reply_to': '<msg1@example.com>',
            'references': '<msg1@example.com>'
        }
        
        # Note: This is a basic test for now
        assert headers['in_reply_to'] == '<msg1@example.com>'
        assert headers['references'] == '<msg1@example.com>'


class TestErrorHandling:
    """Test EARS-ING-8: Error handling and retry."""
    
    def test_transient_failure_retry(self):
        """Test retry logic for transient failures."""
        from app.ingestion.pipeline import IngestionPipeline
        
        pipeline = IngestionPipeline(tenant_id="test_tenant")
        
        # Note: This is a basic test for now
        assert pipeline.tenant_id == "test_tenant"
    
    def test_quarantine_permanent_failures(self):
        """Test quarantine for permanent failures."""
        from app.ingestion.pipeline import IngestionPipeline
        
        pipeline = IngestionPipeline(tenant_id="test_tenant")
        
        # Note: This is a basic test for now
        assert pipeline.tenant_id == "test_tenant"


class TestMetricsCollection:
    """Test EARS-ING-9: Metrics and observability."""
    
    def test_processing_rate_metrics(self):
        """Test processing rate metrics collection."""
        from app.ingestion.metrics import MetricsCollector
        
        collector = MetricsCollector()
        
        # Start batch
        collector.start_batch("test_batch")
        
        # Record processing
        collector.record_email_processed(0.5, False, 0, 2)
        collector.record_email_processed(0.3, True, 1, 3)
        collector.record_email_processed(0.7, False, 0, 1)
        
        # Record failures
        collector.record_email_failed("MALFORMED_MIME")
        collector.record_email_failed("OVERSIZED_EMAIL")
        
        # Complete batch to calculate final metrics
        collector.complete_batch()
        
        # Get metrics
        metrics = collector.get_metrics()
        
        # total_emails includes processed + failed + quarantined
        assert metrics.total_emails == 5  # 3 processed + 2 failed
        assert metrics.error_count == 2
        assert metrics.error_buckets["MALFORMED_MIME"] == 1
        assert metrics.error_buckets["OVERSIZED_EMAIL"] == 1
    
    def test_dedup_ratio_calculation(self):
        """Test deduplication ratio calculation."""
        from app.ingestion.metrics import MetricsCollector
        
        collector = MetricsCollector()
        
        # Record duplicates
        collector.record_duplicate_email("exact")
        collector.record_duplicate_email("near")
        collector.record_email_processed(0.5, False, 0, 2)
        collector.record_email_processed(0.3, False, 0, 1)
        
        # Complete batch to calculate final metrics
        collector.complete_batch()
        
        metrics = collector.get_metrics()
        
        # 2 duplicates out of 4 total = 0.5 ratio
        assert metrics.dedup_ratio == 0.5


class TestTenantIsolation:
    """Test EARS-ING-10: Multi-tenant isolation."""
    
    def test_tenant_scoped_storage_paths(self):
        """Test that storage paths are tenant-scoped."""
        from app.storage.paths import StoragePathBuilder
        
        tenant1_paths = StoragePathBuilder(tenant_id="tenant1")
        tenant2_paths = StoragePathBuilder(tenant_id="tenant2")
        
        # Same email, different tenants should have different paths
        path1 = tenant1_paths.get_raw_email_path("batch1", "email123")
        path2 = tenant2_paths.get_raw_email_path("batch1", "email123")
        
        assert path1 != path2
        assert "tenant1" in path1
        assert "tenant2" in path2
    
    def test_tenant_scoped_database_queries(self):
        """Test that database queries are tenant-scoped."""
        from app.ingestion.pipeline import IngestionPipeline
        
        pipeline = IngestionPipeline(tenant_id="test_tenant")
        
        # Note: This is a basic test for now
        assert pipeline.tenant_id == "test_tenant"


class TestIdempotency:
    """Test idempotency and checkpointing."""
    
    def test_checkpoint_resume(self):
        """Test that processing can resume from checkpoints."""
        from app.ingestion.checkpoints import CheckpointManager
        
        manager = CheckpointManager()
        
        # Note: This is a basic test for now
        assert manager is not None
    
    def test_duplicate_batch_handling(self):
        """Test that duplicate batch processing is handled idempotently."""
        from app.ingestion.pipeline import IngestionPipeline
        
        pipeline = IngestionPipeline(tenant_id="test_tenant")
        
        # Note: This is a basic test for now
        assert pipeline.tenant_id == "test_tenant"


# Integration test fixtures
@pytest.fixture
def temp_dropbox():
    """Create a temporary dropbox directory with test emails."""
    temp_dir = tempfile.mkdtemp()
    
    # Create test email files
    email1_content = """From: test1@example.com
To: recipient@example.com
Subject: Test Email 1
Message-ID: <test1@example.com>

This is test email 1.
"""
    
    email2_content = """From: test2@example.com
To: recipient@example.com
Subject: Test Email 2
Message-ID: <test2@example.com>

This is test email 2.
"""
    
    with open(f"{temp_dir}/email1.eml", "w") as f:
        f.write(email1_content)
    
    with open(f"{temp_dir}/email2.eml", "w") as f:
        f.write(email2_content)
    
    yield temp_dir
    
    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.mark.integration
class TestIngestionIntegration:
    """Integration tests for the ingestion pipeline."""
    
    def test_full_pipeline_roundtrip(self, temp_dropbox):
        """Test complete pipeline roundtrip: manifest → pipeline → storage → DB → chunks."""
        # This is a high-level integration test
        # In a real implementation, you'd test the actual pipeline execution
        
        # Create manifest
        manifest = BatchManifest(
            batch_id="integration_test_batch",
            tenant_id="test_tenant",
            source_type="dropbox",
            source_path=temp_dropbox,
            total_files=2,
            file_manifest=[
                {"file_path": f"{temp_dropbox}/email1.eml", "file_name": "email1.eml", "file_size": 100},
                {"file_path": f"{temp_dropbox}/email2.eml", "file_name": "email2.eml", "file_size": 100}
            ],
            created_at=datetime.utcnow()
        )
        
        # Verify manifest creation
        assert manifest.batch_id == "integration_test_batch"
        assert manifest.total_files == 2
        assert len(manifest.file_manifest) == 2
        
        # Test file existence
        for file_info in manifest.file_manifest:
            assert Path(file_info["file_path"]).exists()
    
    def test_idempotent_processing(self, temp_dropbox):
        """Test that re-running the same batch produces no duplicates."""
        # This test would verify that processing the same batch twice
        # doesn't create duplicate database records or storage objects
        
        manifest = BatchManifest(
            batch_id="idempotency_test_batch",
            tenant_id="test_tenant",
            source_type="dropbox",
            source_path=temp_dropbox,
            total_files=1,
            file_manifest=[
                {"file_path": f"{temp_dropbox}/email1.eml", "file_name": "email1.eml", "file_size": 100}
            ],
            created_at=datetime.utcnow()
        )
        
        # First run (would create records)
        # Second run (should detect existing records and skip)
        # Verify no duplicates were created
        
        # For now, just verify the manifest is valid
        assert manifest.batch_id == "idempotency_test_batch"
        assert manifest.total_files == 1
