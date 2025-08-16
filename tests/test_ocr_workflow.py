"""Tests for LangGraph OCR workflow and sub-agents."""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch, MagicMock
from unittest.mock import AsyncMock

from app.agents.ocr_workflow import (
    AttachmentMiner, DocTextExtractor, OCRDecider,
    OCRWorker, StorageWriter, ComplianceGuard, MetricsAuditor,
    StateReducer, create_ocr_workflow, process_attachment_with_ocr
)
from app.agents.state_contract import OCRWorkflowState, StatePatch
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
            attachment=attachment,
            workflow_id="test-workflow-123"
        )
        
        assert state.tenant_id == "tenant-123"
        assert state.email_id == "email-456"
        assert state.attachment == attachment
        assert state.workflow_id == "test-workflow-123"
        assert state.current_step == "workflow_started"
        assert state.compliance_checks == []
        assert state.processing_metrics == {}


class TestAttachmentMiner:
    """Test AttachmentMiner sub-agent."""
    
    @pytest.mark.asyncio
    async def test_attachment_miner_valid_attachment(self):
        """Test attachment miner with valid attachment."""
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
            attachment=attachment,
            workflow_id="test-workflow-123"
        )
        
        miner = AttachmentMiner()
        result = await miner.process(state)
        
        # Should return a patch (dict)
        assert isinstance(result, dict)
        assert "current_step" in result
        assert "needs_ocr" in result
        assert "processing_metrics" in result
        
        # Check values
        assert result["current_step"] == "attachment_validated"
        assert result["needs_ocr"] is False  # PDF doesn't need OCR
        assert "validation_time" in result["processing_metrics"]
    
    @pytest.mark.asyncio
    async def test_attachment_miner_image_attachment(self):
        """Test attachment miner with image that needs OCR."""
        attachment = AttachmentInfo(
            filename="test.png",
            content_type="image/png",
            content=b"test image content",
            content_length=2048,
            content_hash="def456",
            content_disposition="attachment"
        )
        
        state = OCRWorkflowState(
            tenant_id="tenant-123",
            email_id="email-456",
            attachment=attachment,
            workflow_id="test-workflow-123"
        )
        
        miner = AttachmentMiner()
        result = await miner.process(state)
        
        assert isinstance(result, dict)
        assert result["current_step"] == "attachment_validated"
        assert result["needs_ocr"] is True  # PNG needs OCR
        assert "validation_time" in result["processing_metrics"]


class TestDocTextExtractor:
    """Test DocTextExtractor sub-agent."""
    
    @pytest.mark.asyncio
    async def test_doc_text_extractor_success(self):
        """Test successful text extraction."""
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
            attachment=attachment,
            workflow_id="test-workflow-123",
            needs_ocr=False
        )
        
        # Mock the document processor
        with patch('app.agents.ocr_workflow.get_document_processor') as mock_get_processor:
            mock_processor = Mock()
            mock_result = Mock()
            mock_result.success = True
            mock_result.text = "Extracted text content"
            mock_result.confidence = 0.95
            mock_result.language = "en"
            mock_processor.extract_text = AsyncMock(return_value=mock_result)
            mock_get_processor.return_value = mock_processor
            
            extractor = DocTextExtractor()
            result = await extractor.process(state)
            
            assert isinstance(result, dict)
            assert result["current_step"] == "text_extracted"
            assert result["extracted_text"] == "Extracted text content"
            assert result["extraction_metadata"]["method"] == "native"
            assert result["extraction_metadata"]["confidence"] == 0.95


class TestOCRDecider:
    """Test OCRDecider sub-agent."""
    
    @pytest.mark.asyncio
    async def test_ocr_decider_ocr_required(self):
        """Test OCR decider when OCR is required."""
        attachment = AttachmentInfo(
            filename="test.png",
            content_type="image/png",
            content=b"test image",
            content_length=1024,
            content_hash="abc123",
            content_disposition="attachment"
        )
        
        state = OCRWorkflowState(
            tenant_id="tenant-123",
            email_id="email-456",
            attachment=attachment,
            workflow_id="test-workflow-123",
            needs_ocr=True
        )
        
        decider = OCRDecider()
        result = await decider.process(state)
        
        assert isinstance(result, dict)
        assert result["current_step"] == "ocr_required"
        assert result["processing_metrics"]["decision"] == "ocr_required"
    
    @pytest.mark.asyncio
    async def test_ocr_decider_ocr_not_needed(self):
        """Test OCR decider when OCR is not needed."""
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
            attachment=attachment,
            workflow_id="test-workflow-123",
            needs_ocr=False
        )
        
        decider = OCRDecider()
        result = await decider.process(state)
        
        assert isinstance(result, dict)
        assert result["current_step"] == "ocr_not_needed"
        assert result["processing_metrics"]["decision"] == "ocr_not_needed"


class TestOCRWorker:
    """Test OCRWorker sub-agent."""
    
    @pytest.mark.asyncio
    async def test_ocr_worker_ocr_skipped(self):
        """Test OCR worker when OCR is skipped."""
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
            attachment=attachment,
            workflow_id="test-workflow-123",
            needs_ocr=False
        )
        
        worker = OCRWorker()
        result = await worker.process(state)
        
        assert isinstance(result, dict)
        assert result["current_step"] == "ocr_skipped"
        assert result["processing_metrics"]["reason"] == "ocr_not_needed"
    
    @pytest.mark.asyncio
    async def test_ocr_worker_ocr_success(self):
        """Test OCR worker with successful OCR."""
        attachment = AttachmentInfo(
            filename="test.png",
            content_type="image/png",
            content=b"test image",
            content_length=1024,
            content_hash="abc123",
            content_disposition="attachment"
        )
        
        state = OCRWorkflowState(
            tenant_id="tenant-123",
            email_id="email-456",
            attachment=attachment,
            workflow_id="test-workflow-123",
            needs_ocr=True
        )
        
        # Mock the OCR service
        with patch('app.agents.ocr_workflow.get_ocr_service') as mock_get_service:
            mock_service = Mock()
            mock_result = Mock()
            mock_result.success = True
            mock_result.text = "OCR extracted text"
            mock_result.confidence = 0.88
            mock_result.backend = "test_backend"
            mock_service.process_attachment = AsyncMock(return_value=mock_result)
            mock_get_service.return_value = mock_service
            
            # Mock OCR config
            with patch('app.agents.ocr_workflow.get_ocr_config') as mock_get_config:
                mock_get_config.return_value = {"timeout_seconds": 20, "max_retries": 2}
                
                worker = OCRWorker()
                result = await worker.process(state)
                
                assert isinstance(result, dict)
                assert result["current_step"] == "ocr_completed"
                assert result["ocr_text"] == "OCR extracted text"
                assert result["ocr_confidence"] == 0.88
                assert result["ocr_backend"] == "test_backend"


class TestStorageWriter:
    """Test StorageWriter sub-agent."""
    
    @pytest.mark.asyncio
    async def test_storage_writer_success(self):
        """Test successful storage writing."""
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
            attachment=attachment,
            workflow_id="test-workflow-123",
            extracted_text="Extracted text content"
        )
        
        writer = StorageWriter()
        result = await writer.process(state)
        
        assert isinstance(result, dict)
        assert result["current_step"] == "storage_completed"
        assert "storage_key" in result
        assert "storage_path" in result
        assert result["processing_metrics"]["text_length"] == 22
    
    @pytest.mark.asyncio
    async def test_storage_writer_skipped(self):
        """Test storage writer when no text to store."""
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
            attachment=attachment,
            workflow_id="test-workflow-123"
        )
        
        writer = StorageWriter()
        result = await writer.process(state)
        
        assert isinstance(result, dict)
        assert result["current_step"] == "storage_skipped"
        assert result["processing_metrics"]["reason"] == "no_text_to_store"


class TestComplianceGuard:
    """Test ComplianceGuard sub-agent."""
    
    @pytest.mark.asyncio
    async def test_compliance_guard_success(self):
        """Test successful compliance check."""
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
            attachment=attachment,
            workflow_id="test-workflow-123",
            extracted_text="Normal text content without sensitive information"
        )
        
        guard = ComplianceGuard()
        result = await guard.process(state)
        
        assert isinstance(result, dict)
        assert result["current_step"] == "compliance_checked"
        assert isinstance(result["compliance_checks"], list)
        assert "checks_performed" in result["processing_metrics"]
    
    @pytest.mark.asyncio
    async def test_compliance_guard_sensitive_content(self):
        """Test compliance check with sensitive content."""
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
            attachment=attachment,
            workflow_id="test-workflow-123",
            extracted_text="Here is my password: secret123 and my credit card: 1234-5678-9012-3456"
        )
        
        guard = ComplianceGuard()
        result = await guard.process(state)
        
        assert isinstance(result, dict)
        assert result["current_step"] == "compliance_checked"
        assert "potential_password_exposure" in result["compliance_checks"]
        assert "potential_credit_card_exposure" in result["compliance_checks"]


class TestMetricsAuditor:
    """Test MetricsAuditor sub-agent."""
    
    @pytest.mark.asyncio
    async def test_metrics_auditor_success(self):
        """Test successful metrics audit."""
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
            attachment=attachment,
            workflow_id="test-workflow-123",
            extracted_text="Extracted text content",
            processing_metrics={"validation_time": 0.1, "extraction_time": 0.2}
        )
        
        auditor = MetricsAuditor()
        result = await auditor.process(state)
        
        assert isinstance(result, dict)
        assert result["current_step"] == "workflow_completed"
        assert "audit_time" in result["processing_metrics"]
        assert result["processing_metrics"]["success"] is True
    
    @pytest.mark.asyncio
    async def test_metrics_auditor_failure(self):
        """Test metrics audit with workflow failure."""
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
            attachment=attachment,
            workflow_id="test-workflow-123",
            error_message="Something went wrong"
        )
        
        auditor = MetricsAuditor()
        result = await auditor.process(state)
        
        assert isinstance(result, dict)
        assert result["current_step"] == "workflow_failed"
        assert result["processing_metrics"]["success"] is False


class TestOCRWorkflow:
    """Test complete OCR workflow."""
    
    @pytest.mark.asyncio
    async def test_workflow_creation(self):
        """Test workflow graph creation."""
        workflow = create_ocr_workflow()
        
        # Check that all nodes are present
        assert "attachment_miner" in workflow.nodes
        assert "doc_text_extractor" in workflow.nodes
        assert "ocr_decider" in workflow.nodes
        assert "ocr_worker" in workflow.nodes
        assert "storage_writer" in workflow.nodes
        assert "compliance_guard" in workflow.nodes
        assert "metrics_auditor" in workflow.nodes
    
    @pytest.mark.asyncio
    async def test_workflow_execution(self):
        """Test complete workflow execution."""
        attachment = AttachmentInfo(
            filename="test.pdf",
            content_type="application/pdf",
            content=b"test content",
            content_length=1024,
            content_hash="abc123",
            content_disposition="attachment"
        )
        
        # Mock all the services
        with patch('app.agents.ocr_workflow.get_document_processor') as mock_get_processor, \
             patch('app.agents.ocr_workflow.get_ocr_service') as mock_get_service, \
             patch('app.agents.ocr_workflow.get_ocr_config') as mock_get_config:
            
            # Mock document processor
            mock_processor = Mock()
            mock_result = Mock()
            mock_result.success = True
            mock_result.text = "Extracted text content"
            mock_result.confidence = 0.95
            mock_result.language = "en"
            mock_processor.extract_text = AsyncMock(return_value=mock_result)
            mock_get_processor.return_value = mock_processor
            
            # Mock OCR service
            mock_service = Mock()
            mock_get_service.return_value = mock_service
            
            # Mock OCR config
            mock_get_config.return_value = {
                "allow_mimetypes": ["application/pdf", "image/png"],
                "timeout_seconds": 20,
                "max_retries": 2
            }
            
            # Execute workflow
            result = await process_attachment_with_ocr(
                tenant_id="tenant-123",
                email_id="email-456",
                attachment=attachment
            )
            
            # Should return OCRWorkflowState
            assert isinstance(result, OCRWorkflowState)
            assert result.tenant_id == "tenant-123"
            assert result.email_id == "email-456"
            assert result.attachment == attachment
            
            # Check final step
            assert result.current_step in ["workflow_completed", "workflow_failed", "workflow_incomplete", "state_reconstruction_complete"]
            
            # Check that we have extracted text
            assert result.extracted_text is not None or result.ocr_text is not None

    @pytest.mark.asyncio
    async def test_workflow_config_driven_routing(self):
        """Test that workflow respects config-driven routing settings."""
        # Test with linear mode enabled (default)
        with patch('app.config.manager.get_graph_config') as mock_get_graph_config:
            mock_get_graph_config.return_value = {"linear_mode": True}
            
            workflow = create_ocr_workflow()
            
            # Should have all nodes in linear sequence
            assert "attachment_miner" in workflow.nodes
            assert "state_reducer" in workflow.nodes
            
            # Check that workflow is properly created
            assert workflow is not None
        
        # Test with linear mode disabled
        with patch('app.config.manager.get_graph_config') as mock_get_graph_config:
            mock_get_graph_config.return_value = {"linear_mode": False}
            
            workflow = create_ocr_workflow()
            
            # Should still have all nodes (fallback to linear mode for now)
            assert "attachment_miner" in workflow.nodes
            assert "state_reducer" in workflow.nodes
            
            # Check that workflow is properly created
            assert workflow is not None


class TestStateReducer:
    """Test StateReducer sub-agent."""
    
    @pytest.mark.asyncio
    async def test_state_reducer_success(self):
        """Test successful state reduction."""
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
            attachment=attachment,
            workflow_id="test-workflow-123",
            current_step="workflow_completed"
        )
        
        reducer = StateReducer()
        result = await reducer.process(state)
        
        assert isinstance(result, dict)
        assert result["current_step"] == "state_reconstruction_complete"
        assert "reduction_time" in result["processing_metrics"]
        assert result["processing_metrics"]["final_state_type"] == "OCRWorkflowState"


if __name__ == "__main__":
    pytest.main([__file__])
