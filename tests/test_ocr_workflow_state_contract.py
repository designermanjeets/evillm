"""Tests for OCR workflow state contract and patch utilities."""

import pytest
import time
from unittest.mock import Mock, patch

from app.agents.state_contract import (
    OCRWorkflowState, StatePatch, merge_patch, reconstruct_final_state,
    PatchValidationError, PolicyViolationError, PatchGuard
)
from app.agents.utils import returns_patch, validate_patch_keys
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
    
    def test_workflow_state_immutable_core_fields(self):
        """Test that core fields cannot be modified after creation."""
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
            workflow_id="fixed-workflow-id"
        )
        
        # Core fields should be immutable by design
        assert state.tenant_id == "tenant-123"
        assert state.email_id == "email-456"
        assert state.workflow_id == "fixed-workflow-id"
        
        # These can be modified via patches, but not directly
        state.current_step = "new_step"
        assert state.current_step == "new_step"


class TestStatePatch:
    """Test StatePatch type and validation."""
    
    def test_state_patch_creation(self):
        """Test creating valid state patches."""
        patch = StatePatch(
            current_step="attachment_validated",
            processing_metrics={"validation_time": time.time()}
        )
        
        assert patch["current_step"] == "attachment_validated"
        assert "validation_time" in patch["processing_metrics"]
    
    def test_state_patch_optional_fields(self):
        """Test that all fields in StatePatch are optional."""
        # Empty patch should be valid
        empty_patch = StatePatch()
        assert isinstance(empty_patch, dict)
        assert len(empty_patch) == 0
        
        # Partial patch should be valid
        partial_patch = StatePatch(current_step="completed")
        assert partial_patch["current_step"] == "completed"
        assert "extracted_text" not in partial_patch


class TestMergePatch:
    """Test patch merging functionality."""
    
    def test_merge_patch_basic(self):
        """Test basic patch merging."""
        attachment = AttachmentInfo(
            filename="test.pdf",
            content_type="application/pdf",
            content=b"test content",
            content_length=1024,
            content_hash="abc123",
            content_disposition="attachment"
        )
        
        initial_state = OCRWorkflowState(
            tenant_id="tenant-123",
            email_id="email-456",
            attachment=attachment,
            workflow_id="test-workflow-123"
        )
        
        patch = StatePatch(
            current_step="attachment_validated",
            processing_metrics={"validation_time": time.time()}
        )
        
        new_state = merge_patch(initial_state, patch)
        
        assert new_state.current_step == "attachment_validated"
        assert "validation_time" in new_state.processing_metrics
        assert new_state.tenant_id == "tenant-123"  # Unchanged
        assert new_state.email_id == "email-456"    # Unchanged
    
    def test_merge_patch_banned_key_violation(self):
        """Test that banned keys cannot be modified."""
        attachment = AttachmentInfo(
            filename="test.pdf",
            content_type="application/pdf",
            content=b"test content",
            content_length=1024,
            content_hash="abc123",
            content_disposition="attachment"
        )
        
        initial_state = OCRWorkflowState(
            tenant_id="tenant-123",
            email_id="email-456",
            attachment=attachment,
            workflow_id="test-workflow-123"
        )
        
        # Attempt to modify tenant_id (banned key)
        patch = StatePatch(tenant_id="new-tenant")
        
        with pytest.raises(PolicyViolationError) as exc_info:
            merge_patch(initial_state, patch)
        
        assert "tenant_id" in str(exc_info.value)
        assert "banned key" in str(exc_info.value)
    
    def test_merge_patch_invalid_key(self):
        """Test that invalid keys are rejected."""
        attachment = AttachmentInfo(
            filename="test.pdf",
            content_type="application/pdf",
            content=b"test content",
            content_length=1024,
            content_hash="abc123",
            content_disposition="attachment"
        )
        
        initial_state = OCRWorkflowState(
            tenant_id="tenant-123",
            email_id="email-456",
            attachment=attachment,
            workflow_id="test-workflow-123"
        )
        
        # Attempt to use invalid key
        patch = {"invalid_key": "value"}
        
        with pytest.raises(PatchValidationError) as exc_info:
            merge_patch(initial_state, patch)
        
        assert "invalid_key" in str(exc_info.value)
        assert "Invalid patch keys" in str(exc_info.value)
    
    def test_merge_patch_type_validation(self):
        """Test that patch values match expected types."""
        attachment = AttachmentInfo(
            filename="test.pdf",
            content_type="application/pdf",
            content=b"test content",
            content_length=1024,
            content_hash="abc123",
            content_disposition="attachment"
        )
        
        initial_state = OCRWorkflowState(
            tenant_id="tenant-123",
            email_id="email-456",
            attachment=attachment,
            workflow_id="test-workflow-123"
        )
        
        # Attempt to use wrong type for current_step
        patch = {"current_step": 123}  # Should be string
        
        with pytest.raises(PatchValidationError) as exc_info:
            merge_patch(initial_state, patch)
        
        assert "Type mismatch" in str(exc_info.value)
        assert "current_step" in str(exc_info.value)
    
    def test_merge_patch_none_values_ignored(self):
        """Test that None values in patches are ignored."""
        attachment = AttachmentInfo(
            filename="test.pdf",
            content_type="application/pdf",
            content=b"test content",
            content_length=1024,
            content_hash="abc123",
            content_disposition="attachment"
        )
        
        initial_state = OCRWorkflowState(
            tenant_id="tenant-123",
            email_id="email-456",
            attachment=attachment,
            workflow_id="test-workflow-123"
        )
        
        patch = StatePatch(
            current_step=None,  # Should be ignored
            extracted_text="Some text"  # Should be applied
        )
        
        new_state = merge_patch(initial_state, patch)
        
        # current_step should remain unchanged (None values ignored)
        assert new_state.current_step == "workflow_started"
        # extracted_text should be applied
        assert new_state.extracted_text == "Some text"


class TestPatchGuard:
    """Test PatchGuard context manager."""
    
    def test_patch_guard_context_manager(self):
        """Test PatchGuard as context manager."""
        with PatchGuard("test_node", "test-trace-id") as guard:
            patch = {"current_step": "completed"}
            validated_patch = guard.validate_patch(patch)
            
            assert validated_patch == patch
            assert guard.patch_size == 1
            assert guard.keys_changed == 1
            assert guard.conflicts_detected == 0
            assert guard.policy_violations == 0
    
    def test_patch_guard_validation_error(self):
        """Test PatchGuard validation error handling."""
        with PatchGuard("test_node") as guard:
            with pytest.raises(PatchValidationError):
                guard.validate_patch("not_a_dict")


class TestReturnsPatchDecorator:
    """Test the returns_patch decorator."""
    
    @pytest.mark.asyncio
    async def test_returns_patch_valid_patch(self):
        """Test decorator with valid patch return."""
        mock_state = Mock()
        
        # Create a simple async function to decorate
        async def test_func(state):
            return {
                "current_step": "completed",
                "processing_metrics": {"test_time": time.time()}
            }
        
        # Apply decorator manually for testing
        decorated_func = returns_patch("test_agent")(test_func)
        result = await decorated_func(mock_state)
        
        assert isinstance(result, dict)
        assert result["current_step"] == "completed"
        assert "test_time" in result["processing_metrics"]
    
    @pytest.mark.asyncio
    async def test_returns_patch_invalid_return(self):
        """Test decorator with invalid return type."""
        mock_state = Mock()
        
        # Create a simple async function that returns invalid result
        async def test_func(state):
            return "not_a_dict"
        
        # Apply decorator manually for testing
        decorated_func = returns_patch("test_agent")(test_func)
        
        with pytest.raises(PatchValidationError) as exc_info:
            await decorated_func(mock_state)
        
        assert "must return StatePatch (dict)" in str(exc_info.value)


class TestStateReconstruction:
    """Test final state reconstruction from patches."""
    
    def test_reconstruct_final_state(self):
        """Test reconstructing final state from patches."""
        attachment = AttachmentInfo(
            filename="test.pdf",
            content_type="application/pdf",
            content=b"test content",
            content_length=1024,
            content_hash="abc123",
            content_disposition="attachment"
        )
        
        initial_state = OCRWorkflowState(
            tenant_id="tenant-123",
            email_id="email-456",
            attachment=attachment,
            workflow_id="test-workflow-123"
        )
        
        patches = [
            {"current_step": "attachment_validated"},
            {"extracted_text": "Some text content"},
            {"current_step": "text_extracted"},  # Override previous
            {"needs_ocr": False}
        ]
        
        final_state = reconstruct_final_state(initial_state, patches)
        
        # Latest values should win
        assert final_state.current_step == "text_extracted"
        assert final_state.extracted_text == "Some text content"
        assert final_state.needs_ocr is False
        
        # Core fields should remain unchanged
        assert final_state.tenant_id == "tenant-123"
        assert final_state.email_id == "email-456"
    
    def test_reconstruct_final_state_with_error(self):
        """Test reconstruction fails when patch is invalid."""
        attachment = AttachmentInfo(
            filename="test.pdf",
            content_type="application/pdf",
            content=b"test content",
            content_length=1024,
            content_hash="abc123",
            content_disposition="attachment"
        )
        
        initial_state = OCRWorkflowState(
            tenant_id="tenant-123",
            email_id="email-456",
            attachment=attachment,
            workflow_id="test-workflow-123"
        )
        
        patches = [
            {"current_step": "attachment_validated"},
            {"invalid_key": "value"}  # Invalid patch
        ]
        
        with pytest.raises(PatchValidationError):
            reconstruct_final_state(initial_state, patches)


class TestPatchValidation:
    """Test patch validation utilities."""
    
    def test_validate_patch_keys_allowed(self):
        """Test validation with allowed keys."""
        patch = {"current_step": "completed", "extracted_text": "text"}
        allowed_keys = ["current_step", "extracted_text", "processing_metrics"]
        
        # Should not raise
        validate_patch_keys(patch, allowed_keys)
    
    def test_validate_patch_keys_disallowed(self):
        """Test validation with disallowed keys."""
        patch = {"current_step": "completed", "invalid_key": "value"}
        allowed_keys = ["current_step", "extracted_text"]
        
        with pytest.raises(PatchValidationError) as exc_info:
            validate_patch_keys(patch, allowed_keys)
        
        assert "invalid_key" in str(exc_info.value)
        assert "disallowed keys" in str(exc_info.value)


if __name__ == "__main__":
    pytest.main([__file__])
