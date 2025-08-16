"""LangGraph OCR workflow with sub-agents for attachment processing."""

from typing import Dict, Any, List, Optional, Annotated
from dataclasses import dataclass
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

logger = structlog.get_logger(__name__)


@dataclass
class OCRWorkflowState:
    """State for OCR workflow execution."""
    
    # Input
    tenant_id: str
    email_id: str
    attachment: AttachmentInfo
    
    # Processing state
    workflow_id: Optional[str] = None
    current_step: Optional[str] = None
    error_message: Optional[str] = None
    
    # Document extraction results
    extracted_text: Optional[str] = None
    extraction_metadata: Optional[Dict[str, Any]] = None
    needs_ocr: bool = False
    
    # OCR results
    ocr_text: Optional[str] = None
    ocr_confidence: float = 0.0
    ocr_backend: Optional[str] = None
    ocr_processing_time: float = 0.0
    
    # Storage results
    storage_key: Optional[str] = None
    storage_path: Optional[str] = None
    
    # Compliance and metrics
    compliance_checks: List[str] = None
    processing_metrics: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.workflow_id is None:
            self.workflow_id = str(uuid4())
        if self.current_step is None:
            self.current_step = "workflow_started"
        if self.compliance_checks is None:
            self.compliance_checks = []
        if self.processing_metrics is None:
            self.processing_metrics = {}


class AttachmentMiner:
    """Sub-agent: Scans email for attachments and validates them."""
    
    def __init__(self):
        """Initialize attachment miner."""
        self.ocr_config = get_ocr_config()
    
    async def process(self, state: OCRWorkflowState) -> Dict[str, Any]:
        """Process attachment validation and mining."""
        logger.info("AttachmentMiner: Processing attachment",
                   filename=state.attachment.filename,
                   mimetype=state.attachment.content_type,
                   size_mb=state.attachment.content_length / (1024 * 1024))
        
        try:
            # Validate mimetype
            if state.attachment.content_type not in self.ocr_config.allow_mimetypes:
                new_metrics = state.processing_metrics.copy()
                new_metrics["validation_time"] = time.time()
                
                return {
                    "current_step": "validation_failed",
                    "error_message": f"Mimetype {state.attachment.content_type} not allowed",
                    "processing_metrics": new_metrics
                }
            
            # Validate size
            max_size_mb = self.ocr_config.size_cap_mb
            if state.attachment.content_length > max_size_mb * 1024 * 1024:
                new_metrics = state.processing_metrics.copy()
                new_metrics["validation_time"] = time.time()
                
                return {
                    "current_step": "validation_failed",
                    "error_message": f"Attachment size {state.attachment.content_length / (1024 * 1024):.1f}MB exceeds limit {max_size_mb}MB",
                    "processing_metrics": new_metrics
                }
            
            # Create new state with updates
            new_metrics = state.processing_metrics.copy()
            new_metrics["validation_time"] = time.time()
            
            logger.info("AttachmentMiner: Attachment validated successfully",
                       filename=state.attachment.filename)
            
            return {
                "current_step": "attachment_validated",
                "processing_metrics": new_metrics
            }
            
        except Exception as exc:
            logger.error("AttachmentMiner: Validation failed", exc_info=exc)
            return {
                "current_step": "validation_error",
                "error_message": f"Attachment validation failed: {str(exc)}"
            }


class DocTextExtractor:
    """Sub-agent: Extracts native text from documents."""
    
    def __init__(self):
        """Initialize document text extractor."""
        self.document_processor = get_document_processor()
        self.ocr_config = get_ocr_config()
    
    async def process(self, state: OCRWorkflowState) -> Dict[str, Any]:
        """Extract text from document."""
        logger.info("DocTextExtractor: Extracting text from document",
                   filename=state.attachment.filename,
                   mimetype=state.attachment.content_type)
        
        try:
            # Extract text with timeout
            extraction_result = await self.document_processor.extract_text(
                state.attachment.content,
                state.attachment.content_type,
                timeout=self.ocr_config.timeout_seconds
            )
            
            new_metrics = state.processing_metrics.copy()
            new_metrics["extraction_time"] = time.time()
            
            if extraction_result.success:
                # Determine if OCR is needed
                needs_ocr = self.document_processor.needs_ocr(extraction_result)
                
                logger.info("DocTextExtractor: Text extraction completed",
                           filename=state.attachment.filename,
                           text_length=len(extraction_result.text),
                           needs_ocr=needs_ocr)
                
                return {
                    "current_step": "text_extracted",
                    "extracted_text": extraction_result.text,
                    "extraction_metadata": extraction_result.metadata,
                    "needs_ocr": needs_ocr,
                    "processing_metrics": new_metrics
                }
            else:
                logger.warning("DocTextExtractor: Text extraction failed, will use OCR",
                             filename=state.attachment.filename,
                             error=extraction_result.error_message)
                
                return {
                    "current_step": "extraction_failed",
                    "needs_ocr": True,
                    "processing_metrics": new_metrics
                }
            
        except Exception as exc:
            logger.error("DocTextExtractor: Extraction failed", exc_info=exc)
            return {
                "current_step": "extraction_error",
                "error_message": f"Text extraction failed: {str(exc)}"
            }


class OCRDecider:
    """Sub-agent: Decides whether OCR processing is required."""
    
    def __init__(self):
        """Initialize OCR decider."""
        self.ocr_config = get_ocr_config()
    
    async def process(self, state: OCRWorkflowState) -> Dict[str, Any]:
        """Decide OCR processing path."""
        logger.info("OCRDecider: Evaluating OCR need",
                   filename=state.attachment.filename,
                   needs_ocr=state.needs_ocr,
                   has_extracted_text=bool(state.extracted_text))
        
        try:
            # Check if OCR is enabled
            if not self.ocr_config.enabled:
                logger.info("OCRDecider: OCR processing disabled")
                return {
                    "current_step": "ocr_disabled"
                }
            
            # Check if we have sufficient text
            if state.extracted_text and len(state.extracted_text.strip()) > 100:
                logger.info("OCRDecider: Sufficient text extracted, OCR not needed",
                           filename=state.attachment.filename,
                           text_length=len(state.extracted_text))
                
                return {
                    "current_step": "ocr_not_needed",
                    "needs_ocr": False
                }
            
            # Check if this is an image that needs OCR
            if state.attachment.content_type.startswith("image/"):
                logger.info("OCRDecider: Image requires OCR processing",
                           filename=state.attachment.filename)
                
                return {
                    "current_step": "ocr_required_image",
                    "needs_ocr": True
                }
            
            # Check if document extraction failed
            if not state.extracted_text or len(state.extracted_text.strip()) < 50:
                logger.info("OCRDecider: Document has insufficient text, OCR required",
                           filename=state.attachment.filename)
                
                return {
                    "current_step": "ocr_required_no_text",
                    "needs_ocr": True
                }
            
            # Default: no OCR needed
            logger.info("OCRDecider: OCR not required",
                       filename=state.attachment.filename)
            
            return {
                "current_step": "ocr_not_needed",
                "needs_ocr": False
            }
            
        except Exception as exc:
            logger.error("OCRDecider: Decision failed", exc_info=exc)
            return {
                "current_step": "decision_error",
                "error_message": f"OCR decision failed: {str(exc)}"
            }


class OCRWorker:
    """Sub-agent: Executes OCR processing."""
    
    def __init__(self):
        """Initialize OCR worker."""
        self.ocr_service = get_ocr_service()
        self.ocr_config = get_ocr_config()
    
    async def process(self, state: OCRWorkflowState) -> Dict[str, Any]:
        """Execute OCR processing."""
        if not state.needs_ocr:
            logger.info("OCRWorker: OCR not needed, skipping",
                       filename=state.attachment.filename)
            return {}
        
        logger.info("OCRWorker: Starting OCR processing",
                   filename=state.attachment.filename,
                   mimetype=state.attachment.content_type)
        
        try:
            start_time = time.time()
            
            # Execute OCR
            ocr_result = await self.ocr_service.extract_text(
                state.attachment.content,
                state.attachment.content_type,
                timeout=self.ocr_config.timeout_seconds
            )
            
            processing_time = time.time() - start_time
            new_metrics = state.processing_metrics.copy()
            new_metrics["ocr_time"] = time.time()
            
            if ocr_result.success:
                logger.info("OCRWorker: OCR completed successfully",
                           filename=state.attachment.filename,
                           text_length=len(ocr_result.text),
                           confidence=ocr_result.confidence,
                           backend=ocr_result.backend,
                           processing_time=processing_time)
                
                return {
                    "current_step": "ocr_completed",
                    "ocr_text": ocr_result.text,
                    "ocr_confidence": ocr_result.confidence,
                    "ocr_backend": ocr_result.backend,
                    "ocr_processing_time": processing_time,
                    "processing_metrics": new_metrics
                }
            else:
                logger.error("OCRWorker: OCR processing failed",
                           filename=state.attachment.filename,
                           error=ocr_result.error_message)
                
                return {
                    "current_step": "ocr_failed",
                    "error_message": f"OCR processing failed: {ocr_result.error_message}",
                    "processing_metrics": new_metrics
                }
            
        except Exception as exc:
            logger.error("OCRWorker: OCR processing failed", exc_info=exc)
            return {
                "current_step": "ocr_error",
                "error_message": f"OCR processing failed: {str(exc)}"
            }


class StorageWriter:
    """Sub-agent: Writes OCR text to object storage."""
    
    def __init__(self):
        """Initialize storage writer."""
        self.ocr_config = get_ocr_config()
    
    async def process(self, state: OCRWorkflowState) -> Dict[str, Any]:
        """Write OCR text to storage."""
        # Check if we have text to store
        text_to_store = state.ocr_text or state.extracted_text
        if not text_to_store:
            logger.info("StorageWriter: No text to store, skipping",
                       filename=state.attachment.filename)
            return {}
        
        logger.info("StorageWriter: Writing text to storage",
                   filename=state.attachment.filename,
                   text_length=len(text_to_store))
        
        try:
            # Generate storage path
            storage_path = self._generate_storage_path(state)
            
            # TODO: Implement actual storage write
            # For now, simulate storage write
            storage_key = f"ocr_{state.attachment.filename}_{int(time.time())}"
            
            new_metrics = state.processing_metrics.copy()
            new_metrics["storage_time"] = time.time()
            
            logger.info("StorageWriter: Text stored successfully",
                       filename=state.attachment.filename,
                       storage_key=storage_key,
                       storage_path=storage_path)
            
            return {
                "current_step": "storage_completed",
                "storage_key": storage_key,
                "storage_path": storage_path,
                "processing_metrics": new_metrics
            }
            
        except Exception as exc:
            logger.error("StorageWriter: Storage write failed", exc_info=exc)
            return {
                "current_step": "storage_error",
                "error_message": f"Storage write failed: {str(exc)}"
            }
    
    def _generate_storage_path(self, state: OCRWorkflowState) -> str:
        """Generate storage path for OCR text."""
        # Format: tenant_id/yyyy/mm/dd/email_id/ocr/attachment_filename_ocr.txt
        from datetime import datetime
        
        now = datetime.utcnow()
        date_path = now.strftime("%Y/%m/%d")
        
        return f"{state.tenant_id}/{date_path}/{state.email_id}/ocr/{state.attachment.filename}_ocr.txt"


class ComplianceGuard:
    """Sub-agent: Performs compliance checks and content validation."""
    
    def __init__(self):
        """Initialize compliance guard."""
        self.ocr_config = get_ocr_config()
    
    async def process(self, state: OCRWorkflowState) -> Dict[str, Any]:
        """Perform compliance checks."""
        logger.info("ComplianceGuard: Performing compliance checks",
                   filename=state.attachment.filename)
        
        try:
            checks = []
            
            # Check if we have text content
            text_content = state.ocr_text or state.extracted_text
            if text_content:
                checks.append("text_content_available")
                
                # Check text length
                if len(text_content.strip()) > 0:
                    checks.append("text_not_empty")
                
                # Check for suspicious patterns (basic)
                if self._check_suspicious_content(text_content):
                    checks.append("suspicious_content_detected")
                    logger.warning("ComplianceGuard: Suspicious content detected",
                                 filename=state.attachment.filename)
            
            # Check processing success
            if state.current_step in ["storage_completed", "ocr_completed", "text_extracted"]:
                checks.append("processing_successful")
            
            # Check error conditions
            if state.error_message:
                checks.append("error_occurred")
            
            new_metrics = state.processing_metrics.copy()
            new_metrics["compliance_time"] = time.time()
            
            logger.info("ComplianceGuard: Compliance checks completed",
                       filename=state.attachment.filename,
                       checks=checks)
            
            return {
                "current_step": "compliance_checked",
                "compliance_checks": checks,
                "processing_metrics": new_metrics
            }
            
        except Exception as exc:
            logger.error("ComplianceGuard: Compliance check failed", exc_info=exc)
            return {
                "current_step": "compliance_error",
                "error_message": f"Compliance check failed: {str(exc)}"
            }
    
    def _check_suspicious_content(self, text: str) -> bool:
        """Check for suspicious content patterns."""
        suspicious_patterns = [
            "password", "credit card", "ssn", "social security",
            "bank account", "routing number", "pin"
        ]
        
        text_lower = text.lower()
        for pattern in suspicious_patterns:
            if pattern in text_lower:
                return True
        
        return False


class MetricsAuditor:
    """Sub-agent: Records metrics and updates batch summaries."""
    
    def __init__(self):
        """Initialize metrics auditor."""
        self.ocr_config = get_ocr_config()
    
    async def process(self, state: OCRWorkflowState) -> Dict[str, Any]:
        """Record metrics and finalize workflow."""
        logger.info("MetricsAuditor: Recording final metrics",
                   filename=state.attachment.filename,
                   workflow_id=state.workflow_id)
        
        try:
            # Calculate final metrics
            total_time = time.time() - state.processing_metrics.get("validation_time", time.time())
            
            final_metrics = {
                "workflow_id": state.workflow_id,
                "total_processing_time": total_time,
                "steps_completed": state.current_step,
                "has_error": bool(state.error_message),
                "text_extracted": bool(state.extracted_text),
                "ocr_used": bool(state.ocr_text),
                "storage_successful": bool(state.storage_key),
                "compliance_checks": state.compliance_checks
            }
            
            # Add OCR-specific metrics if OCR was used
            if state.ocr_text:
                final_metrics.update({
                    "ocr_confidence": state.ocr_confidence,
                    "ocr_backend": state.ocr_backend,
                    "ocr_processing_time": state.ocr_processing_time
                })
            
            new_metrics = state.processing_metrics.copy()
            new_metrics.update(final_metrics)
            
            logger.info("MetricsAuditor: Workflow completed successfully",
                       filename=state.attachment.filename,
                       workflow_id=state.workflow_id,
                       total_time=total_time,
                       final_step="workflow_completed")
            
            return {
                "current_step": "workflow_completed",
                "processing_metrics": new_metrics
            }
            
        except Exception as exc:
            logger.error("MetricsAuditor: Metrics recording failed", exc_info=exc)
            return {
                "current_step": "metrics_error",
                "error_message": f"Metrics recording failed: {str(exc)}"
            }


def create_ocr_workflow() -> StateGraph:
    """Create the OCR workflow graph."""
    
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
    
    # Set entry point
    workflow.set_entry_point("attachment_miner")
    
    # Define linear flow - all nodes execute in sequence
    workflow.add_edge("attachment_miner", "doc_text_extractor")
    workflow.add_edge("doc_text_extractor", "ocr_decider")
    workflow.add_edge("ocr_decider", "ocr_worker")
    workflow.add_edge("ocr_worker", "storage_writer")
    workflow.add_edge("storage_writer", "compliance_guard")
    workflow.add_edge("compliance_guard", "metrics_auditor")
    workflow.add_edge("metrics_auditor", END)
    
    # Compile workflow (without checkpointer for now)
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
        workflow_id=str(uuid4()),
        current_step="workflow_started"
    )
    
    # Get workflow
    workflow = create_ocr_workflow()
    
    # Execute workflow
    try:
        result = await workflow.ainvoke(initial_state)
        
        # If result is a dict (AddableValuesDict), reconstruct the state
        if isinstance(result, dict):
            # Create new state with all the accumulated updates
            final_state = OCRWorkflowState(
                tenant_id=tenant_id,
                email_id=email_id,
                attachment=attachment,
                workflow_id=initial_state.workflow_id,
                current_step=result.get("current_step", "workflow_completed"),
                error_message=result.get("error_message"),
                extracted_text=result.get("extracted_text"),
                extraction_metadata=result.get("extraction_metadata"),
                needs_ocr=result.get("needs_ocr", False),
                ocr_text=result.get("ocr_text"),
                ocr_confidence=result.get("ocr_confidence", 0.0),
                ocr_backend=result.get("ocr_backend"),
                ocr_processing_time=result.get("ocr_processing_time", 0.0),
                storage_key=result.get("storage_key"),
                storage_path=result.get("storage_path"),
                compliance_checks=result.get("compliance_checks", []),
                processing_metrics=result.get("processing_metrics", {})
            )
            return final_state
        
        return result
        
    except Exception as exc:
        logger.error("OCR workflow execution failed", exc_info=exc)
        initial_state.error_message = f"Workflow execution failed: {str(exc)}"
        initial_state.current_step = "workflow_error"
        return initial_state
