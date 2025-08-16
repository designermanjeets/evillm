"""LangGraph OCR workflow with sub-agents for attachment processing."""

from typing import Dict, Any, List, Optional, Annotated
import asyncio
import time
import structlog
from uuid import uuid4

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from ..config.manager import get_ocr_config, get_feature_flag
from ..services.ocr import get_ocr_service, OCRResult
from ..services.document_processor import get_document_processor, TextExtractionResult
from ..ingestion.models import AttachmentInfo

# Import new state contract
from .state_contract import OCRWorkflowState, StatePatch, reconstruct_final_state
from .utils import returns_patch

logger = structlog.get_logger(__name__)


class AttachmentMiner:
    """Mine attachments and validate them for processing."""
    
    @returns_patch("attachment_miner")
    async def process(self, state: OCRWorkflowState) -> StatePatch:
        """Process attachment and determine if it needs OCR."""
        start_time = time.time()
        
        try:
            attachment = state.attachment
            
            # Validate mimetype
            config = get_ocr_config()
            # Handle both dict and object returns
            if hasattr(config, 'allow_mimetypes'):
                allowed_mimetypes = config.allow_mimetypes
            elif isinstance(config, dict):
                allowed_mimetypes = config.get("allow_mimetypes", [])
            else:
                # Fallback to default allowed mimetypes
                allowed_mimetypes = [
                    "application/pdf", "image/png", "image/jpeg", "image/gif", 
                    "image/bmp", "image/tiff", "text/plain"
                ]
            
            if attachment.content_type not in allowed_mimetypes:
                return {
                    "current_step": "attachment_rejected",
                    "error_message": f"Unsupported mimetype: {attachment.content_type}",
                    "processing_metrics": {
                        "validation_time": time.time() - start_time,
                        "mimetype": attachment.content_type,
                        "allowed_mimetypes": allowed_mimetypes
                    }
                }
            
            # Check if OCR is needed
            needs_ocr = attachment.content_type in ["image/png", "image/jpeg", "image/gif", "image/bmp", "image/tiff"]
            
            return {
                "current_step": "attachment_validated",
                "needs_ocr": needs_ocr,
                "processing_metrics": {
                    "validation_time": time.time() - start_time,
                    "mimetype": attachment.content_type,
                    "size_bytes": attachment.content_length
                }
            }
            
        except Exception as e:
            logger.error("Attachment mining failed", error=str(e), workflow_id=state.workflow_id)
            return {
                "current_step": "attachment_mining_failed",
                "error_message": str(e),
                "processing_metrics": {
                    "validation_time": time.time() - start_time,
                    "error": str(e)
                }
            }


class DocTextExtractor:
    """Extract text from documents that support native text extraction."""
    
    @returns_patch("doc_text_extractor")
    async def process(self, state: OCRWorkflowState) -> StatePatch:
        """Extract text from document if possible."""
        start_time = time.time()
        
        try:
            attachment = state.attachment
            
            # Skip if already determined to need OCR
            if state.needs_ocr:
                return {
                    "current_step": "ocr_required",
                    "processing_metrics": {
                        "extraction_time": time.time() - start_time,
                        "reason": "needs_ocr_flag_set"
                    }
                }
            
            # Try to extract text natively
            doc_processor = get_document_processor()
            result = await doc_processor.extract_text(attachment)
            
            if result.success and result.text:
                return {
                    "current_step": "text_extracted",
                    "extracted_text": result.text,
                    "extraction_metadata": {
                        "method": "native",
                        "confidence": result.confidence,
                        "language": result.language
                    },
                    "processing_metrics": {
                        "extraction_time": time.time() - start_time,
                        "method": "native",
                        "confidence": result.confidence
                    }
                }
            else:
                # Fall back to OCR
                return {
                    "current_step": "ocr_fallback",
                    "needs_ocr": True,
                    "processing_metrics": {
                        "extraction_time": time.time() - start_time,
                        "method": "native_failed",
                        "fallback_reason": "no_text_extracted"
                    }
                }
                
        except Exception as e:
            logger.error("Document text extraction failed", error=str(e), workflow_id=state.workflow_id)
            return {
                "current_step": "extraction_failed",
                "error_message": str(e),
                "needs_ocr": True,  # Fall back to OCR
                "processing_metrics": {
                    "extraction_time": time.time() - start_time,
                    "error": str(e),
                    "fallback_reason": "exception"
                }
            }


class OCRDecider:
    """Decide whether OCR is needed and route accordingly."""
    
    @returns_patch("ocr_decider")
    async def process(self, state: OCRWorkflowState) -> StatePatch:
        """Decide OCR routing based on current state."""
        start_time = time.time()
        
        try:
            if state.needs_ocr:
                return {
                    "current_step": "ocr_required",
                    "processing_metrics": {
                        "decision_time": time.time() - start_time,
                        "decision": "ocr_required"
                    }
                }
            else:
                return {
                    "current_step": "ocr_not_needed",
                    "processing_metrics": {
                        "decision_time": time.time() - start_time,
                        "decision": "ocr_not_needed"
                    }
                }
                
        except Exception as e:
            logger.error("OCR decision failed", error=str(e), workflow_id=state.workflow_id)
            return {
                "current_step": "ocr_decision_failed",
                "error_message": str(e),
                "processing_metrics": {
                    "decision_time": time.time() - start_time,
                    "error": str(e)
                }
            }


class OCRWorker:
    """Perform OCR processing on attachments."""
    
    @returns_patch("ocr_worker")
    async def process(self, state: OCRWorkflowState) -> StatePatch:
        """Process attachment with OCR."""
        start_time = time.time()
        
        try:
            if not state.needs_ocr:
                return {
                    "current_step": "ocr_skipped",
                    "processing_metrics": {
                        "ocr_time": time.time() - start_time,
                        "reason": "ocr_not_needed"
                    }
                }
            
            attachment = state.attachment
            config = get_ocr_config()
            
            # Handle both dict and object returns
            if hasattr(config, 'timeout_seconds'):
                timeout_seconds = config.timeout_seconds
                max_retries = config.max_retries
            elif isinstance(config, dict):
                timeout_seconds = config.get("timeout_seconds", 20)
                max_retries = config.get("max_retries", 2)
            else:
                timeout_seconds = 20
                max_retries = 2
            
            # Get OCR service
            ocr_service = get_ocr_service()
            
            # Process with OCR
            result = await ocr_service.process_attachment(
                attachment,
                timeout_seconds=timeout_seconds,
                max_retries=max_retries
            )
            
            if result.success:
                return {
                    "current_step": "ocr_completed",
                    "ocr_text": result.text,
                    "ocr_confidence": result.confidence,
                    "ocr_backend": result.backend,
                    "ocr_processing_time": time.time() - start_time,
                    "processing_metrics": {
                        "ocr_time": time.time() - start_time,
                        "backend": result.backend,
                        "confidence": result.confidence,
                        "text_length": len(result.text) if result.text else 0
                    }
                }
            else:
                return {
                    "current_step": "ocr_failed",
                    "error_message": result.error_message,
                    "processing_metrics": {
                        "ocr_time": time.time() - start_time,
                        "error": result.error_message,
                        "backend": result.backend
                    }
                }
                
        except Exception as e:
            logger.error("OCR processing failed", error=str(e), workflow_id=state.workflow_id)
            return {
                "current_step": "ocr_failed",
                "error_message": str(e),
                "processing_metrics": {
                    "ocr_time": time.time() - start_time,
                    "error": str(e)
                }
            }


class StorageWriter:
    """Write extracted text and OCR results to storage."""
    
    @returns_patch("storage_writer")
    async def process(self, state: OCRWorkflowState) -> StatePatch:
        """Write results to storage."""
        start_time = time.time()
        
        try:
            # Determine what text to store
            text_to_store = state.ocr_text if state.ocr_text else state.extracted_text
            
            if not text_to_store:
                return {
                    "current_step": "storage_skipped",
                    "processing_metrics": {
                        "storage_time": time.time() - start_time,
                        "reason": "no_text_to_store"
                    }
                }
            
            # TODO: Implement actual storage writing
            # For now, just simulate storage
            storage_key = f"{state.tenant_id}/emails/{state.email_id}/attachments/{state.attachment.filename}/text"
            storage_path = f"storage://{storage_key}"
            
            return {
                "current_step": "storage_completed",
                "storage_key": storage_key,
                "storage_path": storage_path,
                "processing_metrics": {
                    "storage_time": time.time() - start_time,
                    "storage_key": storage_key,
                    "text_length": len(text_to_store)
                }
            }
            
        except Exception as e:
            logger.error("Storage writing failed", error=str(e), workflow_id=state.workflow_id)
            return {
                "current_step": "storage_failed",
                "error_message": str(e),
                "processing_metrics": {
                    "storage_time": time.time() - start_time,
                    "error": str(e)
                }
            }


class ComplianceGuard:
    """Check compliance and security of extracted content."""
    
    @returns_patch("compliance_guard")
    async def process(self, state: OCRWorkflowState) -> StatePatch:
        """Check compliance of extracted content."""
        start_time = time.time()
        
        try:
            text_to_check = state.ocr_text if state.ocr_text else state.extracted_text
            
            if not text_to_check:
                return {
                    "current_step": "compliance_skipped",
                    "processing_metrics": {
                        "compliance_time": time.time() - start_time,
                        "reason": "no_text_to_check"
                    }
                }
            
            # TODO: Implement actual compliance checks
            # For now, just basic checks
            compliance_checks = []
            
            # Check for sensitive patterns
            sensitive_patterns = ["password", "credit_card", "ssn", "api_key"]
            for pattern in sensitive_patterns:
                if pattern.lower() in text_to_check.lower():
                    compliance_checks.append(f"potential_{pattern}_exposure")
            
            # Check for credit card patterns (more specific)
            import re
            credit_card_pattern = r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b'
            if re.search(credit_card_pattern, text_to_check):
                compliance_checks.append("potential_credit_card_exposure")
            
            # Check content length
            if len(text_to_check) > 10000:  # 10KB limit
                compliance_checks.append("content_too_large")
            
            return {
                "current_step": "compliance_checked",
                "compliance_checks": compliance_checks,
                "processing_metrics": {
                    "compliance_time": time.time() - start_time,
                    "checks_performed": len(compliance_checks),
                    "content_length": len(text_to_check)
                }
            }
            
        except Exception as e:
            logger.error("Compliance check failed", error=str(e), workflow_id=state.workflow_id)
            return {
                "current_step": "compliance_failed",
                "error_message": str(e),
                "processing_metrics": {
                    "compliance_time": time.time() - start_time,
                    "error": str(e)
                }
            }


class MetricsAuditor:
    """Audit and record final metrics."""
    
    @returns_patch("metrics_auditor")
    async def process(self, state: OCRWorkflowState) -> StatePatch:
        """Audit final metrics and complete workflow."""
        start_time = time.time()
        
        try:
            # Calculate overall processing time
            total_time = time.time() - start_time
            
            # Determine final status
            if state.error_message:
                final_step = "workflow_failed"
            elif state.ocr_text or state.extracted_text:
                final_step = "workflow_completed"
            else:
                final_step = "workflow_incomplete"
            
            # Record final metrics
            final_metrics = {
                "audit_time": total_time,
                "final_step": final_step,
                "total_processing_time": total_time,
                "success": final_step == "workflow_completed"
            }
            
            # Merge with existing metrics
            if state.processing_metrics:
                final_metrics.update(state.processing_metrics)
            
            return {
                "current_step": final_step,
                "processing_metrics": final_metrics
            }
            
        except Exception as e:
            logger.error("Metrics audit failed", error=str(e), workflow_id=state.workflow_id)
            return {
                "current_step": "audit_failed",
                "error_message": str(e),
                "processing_metrics": {
                    "audit_time": time.time() - start_time,
                    "error": str(e)
                }
            }


class StateReducer:
    """Final reducer that reconstructs typed state from accumulated patches."""
    
    @returns_patch("state_reducer")
    async def process(self, state: OCRWorkflowState) -> StatePatch:
        """Reconstruct final typed state from accumulated patches."""
        start_time = time.time()
        
        try:
            # This node doesn't actually modify state - it's just a marker
            # The actual reconstruction happens in process_attachment_with_ocr
            return {
                "current_step": "state_reconstruction_complete",
                "processing_metrics": {
                    "reduction_time": time.time() - start_time,
                    "final_state_type": "OCRWorkflowState"
                }
            }
            
        except Exception as e:
            logger.error("State reduction failed", error=str(e), workflow_id=state.workflow_id)
            return {
                "current_step": "reduction_failed",
                "error_message": str(e),
                "processing_metrics": {
                    "reduction_time": time.time() - start_time,
                    "error": str(e)
                }
            }


def create_ocr_workflow() -> StateGraph:
    """Create the OCR workflow graph."""
    
    # Get config for routing mode
    try:
        from ..config.manager import get_graph_config
        graph_config = get_graph_config()
        linear_mode = graph_config.get("linear_mode", True)
    except Exception:
        # Default to linear mode if config access fails
        linear_mode = True
    
    # Create workflow
    workflow = StateGraph(OCRWorkflowState)
    
    # Add nodes
    workflow.add_node("attachment_miner", AttachmentMiner().process)
    workflow.add_node("doc_text_extractor", DocTextExtractor().process)
    workflow.add_node("ocr_decider", OCRDecider().process)
    workflow.add_node("ocr_worker", OCRWorker().process)
    workflow.add_node("storage_writer", StorageWriter().process)
    workflow.add_node("compliance_guard", ComplianceGuard().process)
    workflow.add_node("metrics_auditor", MetricsAuditor().process)
    workflow.add_node("state_reducer", StateReducer().process)
    
    # Set entry point
    workflow.set_entry_point("attachment_miner")
    
    if linear_mode:
        # Linear workflow - all nodes execute in sequence
        workflow.add_edge("attachment_miner", "doc_text_extractor")
        workflow.add_edge("doc_text_extractor", "ocr_decider")
        workflow.add_edge("ocr_decider", "ocr_worker")
        workflow.add_edge("ocr_worker", "storage_writer")
        workflow.add_edge("storage_writer", "compliance_guard")
        workflow.add_edge("compliance_guard", "metrics_auditor")
        workflow.add_edge("metrics_auditor", "state_reducer")
        workflow.add_edge("state_reducer", END)
    else:
        # Conditional workflow with routing decisions
        # This would implement conditional edges based on state decisions
        # For now, fall back to linear mode
        workflow.add_edge("attachment_miner", "doc_text_extractor")
        workflow.add_edge("doc_text_extractor", "ocr_decider")
        workflow.add_edge("ocr_decider", "ocr_worker")
        workflow.add_edge("ocr_worker", "storage_writer")
        workflow.add_edge("storage_writer", "compliance_guard")
        workflow.add_edge("compliance_guard", "metrics_auditor")
        workflow.add_edge("metrics_auditor", "state_reducer")
        workflow.add_edge("state_reducer", END)
    
    # Compile the workflow
    compiled_workflow = workflow.compile()
    
    return compiled_workflow


async def process_attachment_with_ocr(
    tenant_id: str,
    email_id: str,
    attachment: AttachmentInfo
) -> OCRWorkflowState:
    """Process attachment using OCR workflow."""
    
    # Create initial state
    initial_state = OCRWorkflowState(
        tenant_id=tenant_id,
        email_id=email_id,
        attachment=attachment,
        workflow_id=str(uuid4())
    )
    
    # Get workflow
    workflow = create_ocr_workflow()
    
    # Execute workflow
    try:
        result = await workflow.ainvoke(initial_state)
        
        # If result is a dict (AddableValuesDict), reconstruct the state
        if isinstance(result, dict):
            # Extract patches from result
            patches = []
            for key, value in result.items():
                if key != "tenant_id" and key != "email_id" and key != "attachment" and key != "workflow_id":
                    patches.append({key: value})
            
            # Reconstruct final state
            final_state = reconstruct_final_state(initial_state, patches)
            return final_state
        else:
            # Result is already a state object
            return result
            
    except Exception as e:
        logger.error("OCR workflow execution failed", error=str(e), workflow_id=initial_state.workflow_id)
        
        # Return failed state
        failed_state = OCRWorkflowState(
            tenant_id=tenant_id,
            email_id=email_id,
            attachment=attachment,
            workflow_id=initial_state.workflow_id,
            current_step="workflow_failed",
            error_message=str(e)
        )
        return failed_state
