"""Tests for LangGraph OCR workflow and sub-agents."""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch, MagicMock

from app.agents.ocr_workflow import (
    OCRWorkflowState, AttachmentMiner, DocTextExtractor, OCRDecider,
    OCRWorker, StorageWriter, ComplianceGuard, MetricsAuditor,
    create_ocr_workflow, process_attachment_with_ocr
)
from app.ingestion.models import AttachmentInfo


class TestOCRWorkflowState:
    """Test OCR workflow state management."""
    
    def test_workflow_state_initialization(self):
        """Test workflow state initialization."""
        attachment = AttachmentInfo(
            filename="test.pdf",
            content_type="application/pdf",
            content=b"test content",
            content_length=1024,
            content_hash="abc123",
            content_disposition="attachment"
        )
        
        state = OCRWorkflowState(
            tenant_id="tenant-123",
            email_id="email-456",
            attachment=attachment
        )
        
        assert state.tenant_id == "tenant-123"
        assert state.email_id == "email-456"
        assert state.attachment == attachment
        assert state.workflow_id is not None
        assert state.current_step == "workflow_started"
        assert state.compliance_checks == []
        assert state.processing_metrics == {}


class TestAttachmentMiner:
    """Test AttachmentMiner sub-agent."""
    
    @pytest.mark.asyncio
    async def test_attachment_validation_success(self):
        """Test successful attachment validation."""
        miner = AttachmentMiner()
        
        attachment = AttachmentInfo(
            filename="test.jpg",
            content_type="image/jpeg",
            content=b"test content",
            content_length=1024,  # 1KB
            content_hash="abc123",
            content_disposition="attachment"
        )
        
        state = OCRWorkflowState(
            tenant_id="tenant-123",
            email_id="email-456",
            attachment=attachment
        )
        
        result = await miner.process(state)
        
        assert result["current_step"] == "attachment_validated"
        assert result.get("error_message") is None
        assert "validation_time" in result["processing_metrics"]
    
    @pytest.mark.asyncio
    async def test_attachment_validation_invalid_mimetype(self):
        """Test attachment validation with invalid mimetype."""
        miner = AttachmentMiner()
        
        attachment = AttachmentInfo(
            filename="test.exe",
            content_type="application/x-executable",
            content=b"test content",
            content_length=1024,
            content_hash="abc123",
            content_disposition="attachment"
        )
        
        state = OCRWorkflowState(
            tenant_id="tenant-123",
            email_id="email-456",
            attachment=attachment
        )
        
        result = await miner.process(state)
        
        assert result["current_step"] == "validation_failed"
        assert "not allowed" in result["error_message"]
    
    @pytest.mark.asyncio
    async def test_attachment_validation_oversized(self):
        """Test attachment validation with oversized file."""
        miner = AttachmentMiner()
        
        # Create oversized attachment (20MB when limit is 15MB)
        attachment = AttachmentInfo(
            filename="test.pdf",
            content_type="application/pdf",
            content=b"x" * (20 * 1024 * 1024),
            content_length=20 * 1024 * 1024,
            content_hash="abc123",
            content_disposition="attachment"
        )
        
        state = OCRWorkflowState(
            tenant_id="tenant-123",
            email_id="email-456",
            attachment=attachment
        )
        
        result = await miner.process(state)
        
        assert result["current_step"] == "validation_failed"
        assert "exceeds limit" in result["error_message"]


class TestDocTextExtractor:
    """Test DocTextExtractor sub-agent."""
    
    @pytest.mark.asyncio
    async def test_text_extraction_success(self):
        """Test successful text extraction."""
        extractor = DocTextExtractor()
        
        attachment = AttachmentInfo(
            filename="test.txt",
            content_type="text/plain",
            content=b"This is sample text content for testing.",
            content_length=1024,
            content_hash="abc123",
            content_disposition="attachment"
        )
        
        state = OCRWorkflowState(
            tenant_id="tenant-123",
            email_id="email-456",
            attachment=attachment
        )
        
        result = await extractor.process(state)
        
        assert result["current_step"] == "text_extracted"
        assert result["extracted_text"] == "This is sample text content for testing."
        assert result["extraction_metadata"] is not None
        assert "extraction_time" in result["processing_metrics"]
    
    @pytest.mark.asyncio
    async def test_text_extraction_failure(self):
        """Test text extraction failure."""
        extractor = DocTextExtractor()
        
        attachment = AttachmentInfo(
            filename="test.bin",
            content_type="application/octet-stream",
            content=b"\x00\x01\x02\x03",
            content_length=1024,
            content_hash="abc123",
            content_disposition="attachment"
        )
        
        state = OCRWorkflowState(
            tenant_id="tenant-123",
            email_id="email-456",
            attachment=attachment
        )
        
        result = await extractor.process(state)
        
        assert result["current_step"] == "extraction_failed"
        assert result["needs_ocr"] is True


class TestOCRDecider:
    """Test OCRDecider sub-agent."""
    
    @pytest.mark.asyncio
    async def test_ocr_decision_not_needed(self):
        """Test OCR decision when not needed."""
        decider = OCRDecider()
        
        state = OCRWorkflowState(
            tenant_id="tenant-123",
            email_id="email-456",
            attachment=AttachmentInfo(
                filename="test.txt",
                content_type="text/plain",
                content=b"test",
                content_length=1024,
                content_hash="abc123",
                content_disposition="attachment"
            )
        )
        state.extracted_text = "This is a long text document with sufficient content that should not require OCR processing."
        state.needs_ocr = False
        
        result = await decider.process(state)
        
        assert result["current_step"] == "ocr_not_needed"
        assert result["needs_ocr"] is False
    
    @pytest.mark.asyncio
    async def test_ocr_decision_required_image(self):
        """Test OCR decision for image files."""
        decider = OCRDecider()
        
        state = OCRWorkflowState(
            tenant_id="tenant-123",
            email_id="email-456",
            attachment=AttachmentInfo(
                filename="test.jpg",
                content_type="image/jpeg",
                content=b"test",
                content_length=1024,
                content_hash="abc123",
                content_disposition="attachment"
            )
        )
        
        result = await decider.process(state)
        
        assert result["current_step"] == "ocr_required_image"
        assert result["needs_ocr"] is True
    
    @pytest.mark.asyncio
    async def test_ocr_decision_insufficient_text(self):
        """Test OCR decision for documents with insufficient text."""
        decider = OCRDecider()
        
        state = OCRWorkflowState(
            tenant_id="tenant-123",
            email_id="email-456",
            attachment=AttachmentInfo(
                filename="test.pdf",
                content_type="application/pdf",
                content=b"test",
                content_length=1024,
                content_hash="abc123",
                content_disposition="attachment"
            )
        )
        state.extracted_text = "Short"
        
        result = await decider.process(state)
        
        assert result["current_step"] == "ocr_required_no_text"
        assert result["needs_ocr"] is True


class TestOCRWorker:
    """Test OCRWorker sub-agent."""
    
    @pytest.mark.asyncio
    async def test_ocr_processing_success(self):
        """Test successful OCR processing."""
        worker = OCRWorker()
        
        state = OCRWorkflowState(
            tenant_id="tenant-123",
            email_id="email-456",
            attachment=AttachmentInfo(
                filename="test.jpg",
                content_type="image/jpeg",
                content=b"test",
                content_length=1024,
                content_hash="abc123",
                content_disposition="attachment"
            )
        )
        state.needs_ocr = True
        
        result = await worker.process(state)
        
        # Should complete OCR processing
        assert result["ocr_text"] is not None
        assert result["ocr_backend"] is not None
        assert result["ocr_processing_time"] > 0
        assert "ocr_time" in result["processing_metrics"]
    
    @pytest.mark.asyncio
    async def test_ocr_processing_skipped(self):
        """Test OCR processing when not needed."""
        worker = OCRWorker()
        
        state = OCRWorkflowState(
            tenant_id="tenant-123",
            email_id="email-456",
            attachment=AttachmentInfo(
                filename="test.txt",
                content_type="text/plain",
                content=b"test",
                content_length=1024,
                content_hash="abc123",
                content_disposition="attachment"
            )
        )
        state.needs_ocr = False
        
        result = await worker.process(state)
        
        # Should skip OCR processing - returns empty dict
        assert result == {}


class TestStorageWriter:
    """Test StorageWriter sub-agent."""
    
    @pytest.mark.asyncio
    async def test_storage_write_success(self):
        """Test successful storage write."""
        writer = StorageWriter()
        
        state = OCRWorkflowState(
            tenant_id="tenant-123",
            email_id="email-456",
            attachment=AttachmentInfo(
                filename="test.jpg",
                content_type="image/jpeg",
                content=b"test",
                content_length=1024,
                content_hash="abc123",
                content_disposition="attachment"
            )
        )
        state.ocr_text = "Extracted OCR text content"
        
        result = await writer.process(state)
        
        assert result["storage_key"] is not None
        assert result["storage_path"] is not None
        assert result["current_step"] == "storage_completed"
        assert "storage_time" in result["processing_metrics"]
    
    def test_storage_path_generation(self):
        """Test storage path generation."""
        writer = StorageWriter()
        
        state = OCRWorkflowState(
            tenant_id="tenant-123",
            email_id="email-456",
            attachment=AttachmentInfo(
                filename="test.jpg",
                content_type="image/jpeg",
                content=b"test",
                content_length=1024,
                content_hash="abc123",
                content_disposition="attachment"
            )
        )
        
        path = writer._generate_storage_path(state)
        
        assert "tenant-123" in path
        assert "email-456" in path
        assert "ocr" in path
        assert "test.jpg_ocr.txt" in path


class TestComplianceGuard:
    """Test ComplianceGuard sub-agent."""
    
    @pytest.mark.asyncio
    async def test_compliance_checks_success(self):
        """Test successful compliance checks."""
        guard = ComplianceGuard()
        
        state = OCRWorkflowState(
            tenant_id="tenant-123",
            email_id="email-456",
            attachment=AttachmentInfo(
                filename="test.txt",
                content_type="text/plain",
                content=b"test",
                content_length=1024,
                content_hash="abc123",
                content_disposition="attachment"
            )
        )
        state.extracted_text = "This is normal document content."
        state.current_step = "text_extracted"
        
        result = await guard.process(state)
        
        assert result["current_step"] == "compliance_checked"
        assert "text_content_available" in result["compliance_checks"]
        assert "text_not_empty" in result["compliance_checks"]
        assert "processing_successful" in result["compliance_checks"]
        assert "compliance_time" in result["processing_metrics"]
    
    @pytest.mark.asyncio
    async def test_suspicious_content_detection(self):
        """Test suspicious content detection."""
        guard = ComplianceGuard()
        
        state = OCRWorkflowState(
            tenant_id="tenant-123",
            email_id="email-456",
            attachment=AttachmentInfo(
                filename="test.txt",
                content_type="text/plain",
                content=b"test",
                content_length=1024,
                content_hash="abc123",
                content_disposition="attachment"
            )
        )
        state.extracted_text = "Here is my password: secret123 and credit card: 1234-5678-9012-3456"
        
        result = await guard.process(state)
        
        assert "suspicious_content_detected" in result["compliance_checks"]
    
    def test_suspicious_content_check(self):
        """Test suspicious content pattern checking."""
        guard = ComplianceGuard()
        
        # Test suspicious patterns
        assert guard._check_suspicious_content("My password is secret") is True
        assert guard._check_suspicious_content("Credit card number: 1234-5678-9012-3456") is True
        assert guard._check_suspicious_content("SSN: 123-45-6789") is True
        
        # Test normal content
        assert guard._check_suspicious_content("This is normal document content") is False


class TestMetricsAuditor:
    """Test MetricsAuditor sub-agent."""
    
    @pytest.mark.asyncio
    async def test_metrics_recording_success(self):
        """Test successful metrics recording."""
        auditor = MetricsAuditor()
        
        state = OCRWorkflowState(
            tenant_id="tenant-123",
            email_id="email-456",
            attachment=AttachmentInfo(
                filename="test.jpg",
                content_type="image/jpeg",
                content=b"test",
                content_length=1024,
                content_hash="abc123",
                content_disposition="attachment"
            )
        )
        state.processing_metrics["validation_time"] = time.time() - 1.0  # 1 second ago
        state.ocr_text = "OCR extracted text"
        state.ocr_confidence = 0.85
        state.ocr_backend = "stub"
        state.ocr_processing_time = 0.5
        state.storage_key = "storage_key_123"
        state.compliance_checks = ["text_content_available", "processing_successful"]
        
        result = await auditor.process(state)
        
        assert result["current_step"] == "workflow_completed"
        assert result["processing_metrics"]["workflow_id"] == state.workflow_id
        assert result["processing_metrics"]["ocr_confidence"] == 0.85
        assert result["processing_metrics"]["ocr_backend"] == "stub"
        assert result["processing_metrics"]["storage_successful"] is True


class TestOCRWorkflow:
    """Test complete OCR workflow."""
    
    @pytest.mark.asyncio
    async def test_workflow_creation(self):
        """Test workflow graph creation."""
        workflow = create_ocr_workflow()
        
        # Check that workflow is created
        assert workflow is not None
        
        # Check that all nodes are present
        nodes = workflow.get_graph().nodes
        expected_nodes = [
            "attachment_miner", "doc_text_extractor", "ocr_decider",
            "ocr_worker", "storage_writer", "compliance_guard", "metrics_auditor"
        ]
        
        for node in expected_nodes:
            assert node in nodes
    
    @pytest.mark.asyncio
    async def test_workflow_execution_success(self):
        """Test successful workflow execution."""
        attachment = AttachmentInfo(
            filename="test.jpg",
            content_type="image/jpeg",
            content=b"test image content",
            content_length=1024,
            content_hash="abc123",
            content_disposition="attachment"
        )
        
        result = await process_attachment_with_ocr(
            tenant_id="tenant-123",
            email_id="email-456",
            attachment=attachment
        )
        
        # Check that workflow completed
        assert result.current_step == "workflow_completed"
        assert result.workflow_id is not None
        assert result.tenant_id == "tenant-123"
        assert result.email_id == "email-456"
        
        # Check that OCR was used for image
        assert result.needs_ocr is True
        assert result.ocr_text is not None
        assert result.storage_key is not None
    
    @pytest.mark.asyncio
    async def test_workflow_execution_text_document(self):
        """Test workflow execution for text document (no OCR needed)."""
        attachment = AttachmentInfo(
            filename="test.txt",
            content_type="text/plain",
            content=b"This is a text document with sufficient content that should not require OCR processing.",
            content_length=1024,
            content_hash="abc123",
            content_disposition="attachment"
        )
        
        result = await process_attachment_with_ocr(
            tenant_id="tenant-123",
            email_id="email-456",
            attachment=attachment
        )
        
        # Check that workflow completed
        assert result.current_step == "workflow_completed"
        
        # Check that OCR was not needed
        assert result.needs_ocr is False
        assert result.extracted_text is not None
        assert result.ocr_text is None  # No OCR text
        assert result.storage_key is not None  # But still stored


if __name__ == "__main__":
    pytest.main([__file__])
