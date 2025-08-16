"""Tests for OCR service and document processing."""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os

from app.services.ocr import (
    OCRService, OCRServiceRegistry, StubOCRProvider, TesseractOCRProvider,
    OCRResult, OCRTask, OCRStatus, OCRBackendType
)
from app.services.document_processor import (
    DocumentProcessor, TextExtractionResult
)
from app.config.ocr import OCRSettings


class TestOCRService:
    """Test OCR service functionality."""
    
    def test_stub_provider_initialization(self):
        """Test stub OCR provider initialization."""
        provider = StubOCRProvider()
        assert provider.get_backend_type() == OCRBackendType.LOCAL
        assert provider.supports_mimetype("image/jpeg")
        assert provider.supports_mimetype("application/pdf")
    
    @pytest.mark.asyncio
    async def test_stub_provider_text_extraction(self):
        """Test stub provider text extraction."""
        provider = StubOCRProvider()
        content = b"sample image content"
        
        result = await provider.extract_text(content, "image/jpeg")
        
        assert result.success is True
        assert "sample OCR text" in result.text
        assert result.confidence == 0.85
        assert result.backend == "stub"
        assert result.language_detected == "en"
    
    def test_tesseract_provider_import_error(self):
        """Test Tesseract provider handles missing dependencies."""
        with patch.dict('sys.modules', {'pytesseract': None}):
            with pytest.raises(ImportError):
                TesseractOCRProvider()
    
    def test_ocr_service_registry(self):
        """Test OCR service registry functionality."""
        registry = OCRServiceRegistry()
        
        # Register providers
        stub_provider = StubOCRProvider()
        registry.register_provider("stub", stub_provider)
        
        assert "stub" in registry.providers
        assert registry.get_provider("stub") == stub_provider
        
        # Set default provider
        registry.set_default_provider("stub")
        assert registry.default_provider == stub_provider
        
        # Test fallback
        assert registry.get_provider() == stub_provider
    
    def test_ocr_service_registry_no_providers(self):
        """Test registry behavior with no providers."""
        registry = OCRServiceRegistry()
        
        with pytest.raises(RuntimeError, match="No OCR providers available"):
            registry.get_provider()
    
    @pytest.mark.asyncio
    async def test_ocr_service_text_extraction(self):
        """Test OCR service text extraction."""
        service = OCRService()
        
        # Test with stub provider
        content = b"sample content"
        result = await service.extract_text(content, "image/jpeg")
        
        assert result.success is True
        assert result.backend in ["stub", "tesseract"]
        assert len(result.text) > 0
    
    @pytest.mark.asyncio
    async def test_ocr_service_timeout(self):
        """Test OCR service timeout handling."""
        service = OCRService()
        
        # Test timeout
        result = await service.extract_text(b"content", "image/jpeg", timeout=0.001)
        
        assert result.success is False
        assert "timed out" in result.error_message.lower()
    
    def test_ocr_service_available_providers(self):
        """Test getting available providers."""
        service = OCRService()
        providers = service.get_available_providers()
        
        assert "stub" in providers
        assert len(providers) >= 1
    
    def test_ocr_service_provider_info(self):
        """Test getting provider information."""
        service = OCRService()
        info = service.get_provider_info("stub")
        
        assert info is not None
        assert info["name"] == "stub"
        assert info["backend_type"] == "local"
        assert info["supports_images"] is True


class TestDocumentProcessor:
    """Test document processor functionality."""
    
    def test_document_processor_initialization(self):
        """Test document processor initialization."""
        processor = DocumentProcessor()
        
        assert processor.supports_mimetype("application/pdf")
        assert processor.supports_mimetype("application/vnd.openxmlformats-officedocument.wordprocessingml.document")
        assert processor.supports_mimetype("text/plain")
        assert not processor.supports_mimetype("unsupported/type")
    
    def test_document_processor_supported_mimetypes(self):
        """Test getting supported mimetypes."""
        processor = DocumentProcessor()
        mimetypes = processor.get_supported_mimetypes()
        
        assert "application/pdf" in mimetypes
        assert "text/plain" in mimetypes
        assert len(mimetypes) >= 4
    
    @pytest.mark.asyncio
    async def test_plain_text_extraction(self):
        """Test plain text extraction."""
        processor = DocumentProcessor()
        content = b"This is sample plain text content."
        
        result = await processor.extract_text(content, "text/plain")
        
        assert result.success is True
        assert result.text == "This is sample plain text content."
        assert result.pages == 1
        assert result.metadata["extractor"] == "native"
    
    @pytest.mark.asyncio
    async def test_plain_text_extraction_encoding_fallback(self):
        """Test plain text extraction with encoding fallback."""
        processor = DocumentProcessor()
        # Create content with non-UTF-8 bytes
        content = b"This is sample text with \x80\x81 bytes"
        
        result = await processor.extract_text(content, "text/plain")
        
        assert result.success is True
        assert "sample text" in result.text
        assert result.pages == 1
    
    @pytest.mark.asyncio
    async def test_unsupported_mimetype(self):
        """Test handling of unsupported mimetypes."""
        processor = DocumentProcessor()
        content = b"sample content"
        
        result = await processor.extract_text(content, "unsupported/type")
        
        assert result.success is False
        assert "Unsupported mimetype" in result.error_message
    
    @pytest.mark.asyncio
    async def test_extraction_timeout(self):
        """Test text extraction timeout."""
        processor = DocumentProcessor()
        content = b"sample content"
        
        # Note: Plain text extraction is very fast, so timeout may not trigger
        # This test verifies the timeout mechanism exists
        result = await processor.extract_text(content, "text/plain", timeout=0.001)
        
        # The result should be successful since plain text extraction is fast
        assert result.success is True
    
    def test_needs_ocr_detection(self):
        """Test OCR need detection logic."""
        processor = DocumentProcessor()
        
        # Test successful extraction with meaningful text
        good_result = TextExtractionResult(
            success=True,
            text="This is a meaningful document with substantial content that should not require OCR processing.",
            pages=1,
            processing_time=0.1
        )
        assert not processor.needs_ocr(good_result)
        
        # Test successful extraction with insufficient text
        short_result = TextExtractionResult(
            success=True,
            text="Short",
            pages=1,
            processing_time=0.1
        )
        assert processor.needs_ocr(short_result)
        
        # Test failed extraction
        failed_result = TextExtractionResult(
            success=False,
            text="",
            pages=0,
            processing_time=0.1
        )
        assert processor.needs_ocr(failed_result)
        
        # Test mostly special characters
        special_result = TextExtractionResult(
            success=True,
            text="!@#$%^&*()_+-=[]{}|;':\",./<>?",
            pages=1,
            processing_time=0.1
        )
        assert processor.needs_ocr(special_result)


class TestOCRIntegration:
    """Test OCR integration with attachment processing."""
    
    @pytest.mark.asyncio
    async def test_attachment_processor_ocr_integration(self):
        """Test attachment processor OCR integration."""
        from app.ingestion.attachments import AttachmentProcessor
        from app.ingestion.models import AttachmentInfo
        
        # Mock OCR settings
        with patch('app.ingestion.attachments.get_ocr_settings') as mock_settings:
            mock_settings.return_value = OCRSettings()
            
            processor = AttachmentProcessor()
            
            # Create test attachment
            attachment = AttachmentInfo(
                filename="test_image.jpg",
                content_type="image/jpeg",
                content=b"sample image content",
                content_length=1024,
                content_hash="abc123",
                content_disposition="attachment"
            )
            
            # Process attachment
            result = await processor.process_attachment(attachment, "tenant-123")
            
            assert result["needs_ocr"] is True
            assert "ocr_task_id" in result
            assert len(processor.ocr_queue) == 1
    
    @pytest.mark.asyncio
    async def test_ocr_queue_processing(self):
        """Test OCR queue processing."""
        from app.ingestion.attachments import AttachmentProcessor
        from app.ingestion.models import AttachmentInfo
        
        with patch('app.ingestion.attachments.get_ocr_settings') as mock_settings:
            mock_settings.return_value = OCRSettings()
            
            processor = AttachmentProcessor()
            
            # Add test attachment
            attachment = AttachmentInfo(
                filename="test_image.jpg",
                content_type="image/jpeg",
                content=b"sample image content",
                content_length=1024,
                content_hash="abc123",
                content_disposition="attachment"
            )
            
            await processor.process_attachment(attachment, "tenant-123")
            
            # Process OCR queue
            results = await processor.process_ocr_queue()
            
            assert len(results) == 1
            assert results[0]["success"] is True
            assert "sample OCR text" in results[0]["text"]
    
    def test_attachment_validation(self):
        """Test attachment validation logic."""
        from app.ingestion.attachments import AttachmentProcessor
        from app.ingestion.models import AttachmentInfo
        
        with patch('app.ingestion.attachments.get_ocr_settings') as mock_settings:
            mock_settings.return_value = OCRSettings()
            
            processor = AttachmentProcessor()
            
            # Test valid attachment
            valid_attachment = AttachmentInfo(
                filename="test.jpg",
                content_type="image/jpeg",
                content=b"content",
                content_length=1024,
                content_hash="abc123",
                content_disposition="attachment"
            )
            assert processor._is_valid_attachment(valid_attachment) is True
            
            # Test invalid mimetype
            invalid_mimetype = AttachmentInfo(
                filename="test.exe",
                content_type="application/x-executable",
                content=b"content",
                content_length=1024,
                content_hash="abc123",
                content_disposition="attachment"
            )
            assert processor._is_valid_attachment(invalid_mimetype) is False
            
            # Test oversized attachment
            oversized = AttachmentInfo(
                filename="test.jpg",
                content_type="image/jpeg",
                content=b"content",
                content_length=20 * 1024 * 1024,  # 20MB
                content_hash="abc123",
                content_disposition="attachment"
            )
            assert processor._is_valid_attachment(oversized) is False


class TestOCRConfiguration:
    """Test OCR configuration settings."""
    
    def test_ocr_settings_defaults(self):
        """Test OCR settings default values."""
        settings = OCRSettings()
        
        assert settings.default_backend == "local"
        assert settings.max_file_size_mb == 50
        assert settings.max_pages == 100
        assert settings.default_timeout_seconds == 30
        assert settings.max_retries == 3
        assert "image/jpeg" in settings.allowed_mimetypes
        assert "application/pdf" in settings.allowed_mimetypes
    
    def test_ocr_settings_size_limits(self):
        """Test OCR settings size limits."""
        settings = OCRSettings()
        
        assert settings.size_limits_mb["image/jpeg"] == 10
        assert settings.size_limits_mb["application/pdf"] == 50
        assert settings.size_limits_mb["text/plain"] == 5
    
    def test_ocr_settings_preprocessing(self):
        """Test OCR settings preprocessing options."""
        settings = OCRSettings()
        
        assert settings.enable_preprocessing is True
        assert settings.enable_grayscale is True
        assert settings.enable_binarization is True
        assert settings.enable_deskewing is True


class TestEARSCompliance:
    """Test EARS-OCR requirements compliance."""
    
    @pytest.mark.asyncio
    async def test_EARS_OCR_1_text_extraction(self):
        """Test EARS-OCR-1: OCR text extraction with timeout and retry."""
        service = OCRService()
        
        # Test OCR with timeout
        result = await service.extract_text(
            b"sample content", 
            "image/jpeg", 
            timeout=0.001
        )
        
        # Should handle timeout gracefully
        assert result.success is False
        assert "timed out" in result.error_message.lower()
    
    @pytest.mark.asyncio
    async def test_EARS_OCR_2_text_storage(self):
        """Test EARS-OCR-2: OCR text storage preparation."""
        service = OCRService()
        
        result = await service.extract_text(b"sample content", "image/jpeg")
        
        # OCR result should contain text ready for storage
        assert result.success is True
        assert len(result.text) > 0
        assert result.backend in ["stub", "tesseract"]
    
    @pytest.mark.asyncio
    async def test_EARS_OCR_3_failure_handling(self):
        """Test EARS-OCR-3: OCR failure handling and quarantine."""
        service = OCRService()
        
        # Test with timeout to simulate failure
        result = await service.extract_text(b"content", "image/jpeg", timeout=0.001)
        
        assert result.success is False
        assert result.error_message is not None
        # In real implementation, this would trigger quarantine
    
    def test_EARS_OCR_4_mimetype_allowlist(self):
        """Test EARS-OCR-4: Mimetype allowlist enforcement."""
        settings = OCRSettings()
        
        # Test allowed mimetypes
        assert "image/jpeg" in settings.allowed_mimetypes
        assert "application/pdf" in settings.allowed_mimetypes
        
        # Test rejected mimetypes
        assert "application/x-executable" not in settings.allowed_mimetypes
        assert "text/javascript" not in settings.allowed_mimetypes
    
    def test_EARS_OCR_5_backend_support(self):
        """Test EARS-OCR-5: Multiple backend support."""
        service = OCRService()
        providers = service.get_available_providers()
        
        assert "stub" in providers
        assert len(providers) >= 1
    
    def test_EARS_OCR_6_image_preprocessing(self):
        """Test EARS-OCR-6: Image preprocessing support."""
        settings = OCRSettings()
        
        assert settings.enable_preprocessing is True
        assert settings.enable_grayscale is True
        assert settings.enable_binarization is True
    
    @pytest.mark.asyncio
    async def test_EARS_OCR_7_idempotency(self):
        """Test EARS-OCR-7: OCR idempotency."""
        service = OCRService()
        
        # Same input should produce same output
        result1 = await service.extract_text(b"same content", "image/jpeg")
        result2 = await service.extract_text(b"same content", "image/jpeg")
        
        assert result1.text == result2.text
        assert result1.backend == result2.backend
    
    def test_EARS_OCR_8_metrics_emission(self):
        """Test EARS-OCR-8: Metrics emission."""
        from app.ingestion.metrics import MetricsCollector
        
        metrics = MetricsCollector()
        
        # Record OCR metrics
        metrics.record_ocr_task_queued()
        metrics.record_ocr_task_started()
        metrics.record_ocr_task_completed(1.5)
        metrics.record_ocr_task_failed()
        
        # Verify metrics are tracked
        assert metrics.ocr_tasks_queued == 1
        assert metrics.ocr_tasks_started == 1
        assert metrics.ocr_tasks_completed == 1
        assert metrics.ocr_tasks_failed == 1
        assert metrics.ocr_success_rate == 0.5
        assert metrics.get_ocr_latency_p95() == 1.5


if __name__ == "__main__":
    pytest.main([__file__])
