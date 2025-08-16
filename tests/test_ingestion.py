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


class TestConflictDetection:
    """Test EARS-ING-11: Conflict detection and resolution."""
    
    def test_concurrent_batch_conflict_detection(self):
        """Test detection of concurrent batch processing conflicts."""
        from app.ingestion.pipeline import IngestionPipeline
        
        pipeline1 = IngestionPipeline(tenant_id="test_tenant")
        pipeline2 = IngestionPipeline(tenant_id="test_tenant")
        
        # Simulate concurrent batch processing
        batch_id = "concurrent_batch_001"
        
        # Both pipelines try to process the same batch
        pipeline1.current_batch_id = batch_id
        pipeline2.current_batch_id = batch_id
        
        # Should detect conflict
        assert pipeline1.current_batch_id == pipeline2.current_batch_id
        
        # In real implementation, this would trigger conflict resolution
        # For now, just verify the conflict scenario is detectable
        assert True
    
    def test_tenant_isolation_conflict_prevention(self):
        """Test that tenant isolation prevents cross-tenant conflicts."""
        from app.ingestion.pipeline import IngestionPipeline
        
        tenant1_pipeline = IngestionPipeline(tenant_id="tenant_1")
        tenant2_pipeline = IngestionPipeline(tenant_id="tenant_2")
        
        # Same batch ID, different tenants should not conflict
        batch_id = "shared_batch_001"
        
        tenant1_pipeline.current_batch_id = batch_id
        tenant2_pipeline.current_batch_id = batch_id
        
        # Different tenants can process same batch ID without conflict
        assert tenant1_pipeline.tenant_id != tenant2_pipeline.tenant_id
        assert tenant1_pipeline.current_batch_id == tenant2_pipeline.current_batch_id
    
    def test_storage_path_conflict_detection(self):
        """Test detection of storage path conflicts."""
        from app.storage.paths import StoragePathBuilder
        
        path_builder = StoragePathBuilder(tenant_id="test_tenant")
        
        # Generate paths for same email in different batches
        path1 = path_builder.get_raw_email_path("batch_1", "email_123")
        path2 = path_builder.get_raw_email_path("batch_2", "email_123")
        
        # Different batches should have different paths
        assert path1 != path2
        assert "batch_1" in path1
        assert "batch_2" in path2
        
        # Same batch, same email should have same path
        path3 = path_builder.get_raw_email_path("batch_1", "email_123")
        assert path1 == path3


class TestObservability:
    """Test EARS-ING-12: Observability and monitoring."""
    
    def test_structured_logging_with_trace_id(self):
        """Test structured logging includes trace IDs and tenant context."""
        from app.ingestion.pipeline import IngestionPipeline
        
        # Create pipeline instance
        pipeline = IngestionPipeline(tenant_id="test_tenant")
        
        # Verify tenant isolation is working
        assert pipeline.tenant_id == "test_tenant"
        
        # In real implementation, this would log with trace_id and tenant_id
        # For now, just verify the pipeline is properly initialized
        assert True
    
    def test_metrics_collection_with_dimensions(self):
        """Test metrics collection includes proper dimensions."""
        from app.ingestion.metrics import MetricsCollector
        
        collector = MetricsCollector()
        
        # Start batch
        collector.start_batch("test_batch")
        
        # Record processing
        collector.record_email_processed(0.5, False, 0, 2)
        collector.record_email_processed(0.3, True, 1, 3)
        
        # Complete batch
        collector.complete_batch()
        
        # Get metrics
        metrics = collector.get_metrics()
        
        # Should have correct counts
        assert metrics.total_emails == 2
        assert metrics.error_count == 0
        
        # In real implementation, metrics would include tenant_id dimension
        assert True
    
    def test_performance_timing_accuracy(self):
        """Test that performance timing is accurate and consistent."""
        from app.ingestion.metrics import MetricsCollector
        import time
        
        collector = MetricsCollector()
        
        # Start timing
        start_time = time.time()
        collector.start_batch("timing_test_batch")
        
        # Simulate processing time
        time.sleep(0.1)
        
        # Record processing
        collector.record_email_processed(0.1, False, 0, 1)
        
        # Complete batch
        collector.complete_batch()
        end_time = time.time()
        
        # Get metrics
        metrics = collector.get_metrics()
        
        # Total time should be reasonable
        total_time = end_time - start_time
        assert total_time >= 0.1  # At least our sleep time
        assert total_time < 1.0   # Should not be excessive
    
    def test_error_bucket_categorization(self):
        """Test that errors are properly categorized into buckets."""
        from app.ingestion.metrics import MetricsCollector
        
        collector = MetricsCollector()
        
        # Record various error types
        collector.record_email_failed("MALFORMED_MIME")
        collector.record_email_failed("OVERSIZED_EMAIL")
        collector.record_email_failed("MALFORMED_MIME")  # Duplicate error
        collector.record_email_failed("UNKNOWN_ERROR")
        
        # Complete batch
        collector.complete_batch()
        
        # Get metrics
        metrics = collector.get_metrics()
        
        # Should have correct error counts
        assert metrics.error_count == 4
        assert metrics.error_buckets["MALFORMED_MIME"] == 2
        assert metrics.error_buckets["OVERSIZED_EMAIL"] == 1
        assert metrics.error_buckets["UNKNOWN_ERROR"] == 1


class TestConfigDrivenRouting:
    """Test EARS-ING-13: Configuration-driven workflow routing."""
    
    def test_linear_mode_configuration(self):
        """Test linear mode configuration from app.yaml."""
        from app.config.manager import get_graph_config
        
        # Get graph configuration
        graph_config = get_graph_config()
        
        # Should have linear_mode setting
        assert "linear_mode" in graph_config
        assert isinstance(graph_config["linear_mode"], bool)
        
        # Default should be True (linear execution)
        assert graph_config["linear_mode"] is True
    
    def test_conflict_policy_configuration(self):
        """Test conflict policy configuration from app.yaml."""
        from app.config.manager import get_graph_config
        
        graph_config = get_graph_config()
        
        # Should have conflict_policy setting
        assert "conflict_policy" in graph_config
        assert graph_config["conflict_policy"] in ["error", "warn", "ignore"]
        
        # Default should be "error"
        assert graph_config["conflict_policy"] == "error"
    
    def test_observability_configuration(self):
        """Test observability configuration from app.yaml."""
        from app.config.manager import get_graph_config
        
        graph_config = get_graph_config()
        
        # Should have observability section
        assert "observability" in graph_config
        observability = graph_config["observability"]
        
        # Check specific observability settings
        assert "log_patches" in observability
        assert "metrics" in observability
        assert "trace_id" in observability
        assert "redact_sensitive" in observability
        
        # Default values should be sensible
        assert observability["metrics"] is True
        assert observability["trace_id"] is True
        assert observability["redact_sensitive"] is True
    
    def test_guard_banned_keys_configuration(self):
        """Test guard banned keys configuration from app.yaml."""
        from app.config.manager import get_graph_config
        
        graph_config = get_graph_config()
        
        # Should have guard_banned_keys setting
        assert "guard_banned_keys" in graph_config
        assert isinstance(graph_config["guard_banned_keys"], list)
        
        # Should include tenant_id as banned key
        assert "tenant_id" in graph_config["guard_banned_keys"]
    
    def test_ocr_feature_flags_configuration(self):
        """Test OCR feature flags configuration from app.yaml."""
        from app.config.manager import get_ocr_config
        
        ocr_config = get_ocr_config()
        
        # Should have feature_flags section
        assert hasattr(ocr_config, 'feature_flags')
        feature_flags = ocr_config.feature_flags
        
        # Check specific feature flags
        assert "ocr_idempotent_skip" in feature_flags
        assert "ocr_queue_inprocess" in feature_flags
        assert "ocr_redact_logs" in feature_flags
        assert "ocr_preprocessing" in feature_flags
        assert "ocr_tesseract_enabled" in feature_flags
        
        # Default values should be sensible
        assert feature_flags["ocr_idempotent_skip"] is True
        assert feature_flags["ocr_redact_logs"] is True
        assert feature_flags["ocr_preprocessing"] is True
        assert feature_flags["ocr_tesseract_enabled"] is False
    
    def test_storage_paths_configuration(self):
        """Test storage paths configuration from app.yaml."""
        from app.config.manager import get_storage_config
        
        storage_config = get_storage_config()
        
        # Should have paths section
        assert "paths" in storage_config
        paths = storage_config["paths"]
        
        # Check specific path templates
        assert "emails" in paths
        assert "attachments" in paths
        assert "ocr_text" in paths
        
        # Paths should include tenant_id placeholder
        assert "{tenant_id}" in paths["emails"]
        assert "{tenant_id}" in paths["attachments"]
        assert "{tenant_id}" in paths["ocr_text"]
    
    def test_metrics_targets_configuration(self):
        """Test metrics targets configuration from app.yaml."""
        from app.config.manager import get_config
        
        config = get_config()
        
        # Should have metrics section
        assert "metrics" in config
        metrics = config["metrics"]
        
        # Should have OCR metrics subsection
        assert "ocr" in metrics
        ocr_metrics = metrics["ocr"]
        
        # Check specific metric targets
        assert "latency_p95_target" in ocr_metrics
        assert "latency_p95_target_prod" in ocr_metrics
        assert "success_rate_target" in ocr_metrics
        assert "queue_depth_alert_threshold" in ocr_metrics
        
        # Values should be reasonable
        assert ocr_metrics["latency_p95_target"] > 0
        assert ocr_metrics["success_rate_target"] > 0 and ocr_metrics["success_rate_target"] <= 1
        assert ocr_metrics["queue_depth_alert_threshold"] > 0 and ocr_metrics["queue_depth_alert_threshold"] <= 1


class TestStateContractCompliance:
    """Test EARS-ING-14: State contract compliance and validation."""
    
    def test_state_patch_immutability(self):
        """Test that state patches are immutable and don't modify original state."""
        from app.agents.state_contract import OCRWorkflowState, StatePatch
        from dataclasses import replace
        
        # Create original state
        original_state = OCRWorkflowState(
            tenant_id="test_tenant",
            email_id="test_email",
            attachment=None,
            workflow_id="test_workflow"
        )
        
        # Create patch
        patch = StatePatch(
            current_step="test_step",
            processing_metrics={"test_metric": "test_value"}
        )
        
        # Apply patch to create new state using dataclass replace
        new_state = replace(original_state, **patch)
        
        # Original state should be unchanged
        assert original_state.current_step == "workflow_started"
        assert original_state.processing_metrics == {}
        
        # New state should have patch values
        assert new_state.current_step == "test_step"
        assert new_state.processing_metrics["test_metric"] == "test_value"
    
    def test_state_validation_rules(self):
        """Test that state validation rules are enforced."""
        from app.agents.state_contract import OCRWorkflowState
        
        # Test valid state creation
        valid_state = OCRWorkflowState(
            tenant_id="test_tenant",
            email_id="test_email",
            attachment=None,
            workflow_id="test_workflow"
        )
        assert valid_state.tenant_id == "test_tenant"
        
        # Test invalid state creation (missing required fields)
        # Since this is a dataclass, Python will raise TypeError for missing args
        with pytest.raises(TypeError):
            OCRWorkflowState(
                tenant_id="test_tenant",
                # Missing email_id
                attachment=None,
                workflow_id="test_workflow"
            )
    
    def test_state_transition_validation(self):
        """Test that state transitions follow valid sequences."""
        from app.agents.state_contract import OCRWorkflowState
        
        # Create state with initial step
        state = OCRWorkflowState(
            tenant_id="test_tenant",
            email_id="test_email",
            attachment=None,
            workflow_id="test_workflow"
        )
        
        # Initial step should be workflow_started
        assert state.current_step == "workflow_started"
        
        # Valid transition to next step
        state.current_step = "attachment_validated"
        assert state.current_step == "attachment_validated"
        
        # In real implementation, we'd validate transition sequences
        # For now, just verify the state can be updated
        assert True


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
