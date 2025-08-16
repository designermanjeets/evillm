"""LangGraph state contract and patch utilities for OCR workflow."""

from typing import Dict, Any, List, Optional, TypedDict, get_origin, get_args
from dataclasses import dataclass, replace
import time
import uuid
import structlog
from contextlib import contextmanager
from functools import wraps

from ..ingestion.models import AttachmentInfo

logger = structlog.get_logger(__name__)


@dataclass
class OCRWorkflowState:
    """Strongly-typed state object for OCR workflow execution."""
    
    # Immutable core fields (never change)
    tenant_id: str
    email_id: str
    attachment: AttachmentInfo
    workflow_id: str
    
    # Patchable processing fields
    current_step: Optional[str] = None
    error_message: Optional[str] = None
    extracted_text: Optional[str] = None
    extraction_metadata: Optional[Dict[str, Any]] = None
    needs_ocr: bool = False
    ocr_text: Optional[str] = None
    ocr_confidence: float = 0.0
    ocr_backend: Optional[str] = None
    ocr_processing_time: float = 0.0
    storage_key: Optional[str] = None
    storage_path: Optional[str] = None
    compliance_checks: List[str] = None
    processing_metrics: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.workflow_id is None:
            self.workflow_id = str(uuid.uuid4())
        if self.current_step is None:
            self.current_step = "workflow_started"
        if self.compliance_checks is None:
            self.compliance_checks = []
        if self.processing_metrics is None:
            self.processing_metrics = {}


class StatePatch(TypedDict, total=False):
    """Patch object containing only the fields that change.
    
    All fields are optional - only include what the sub-agent modifies.
    """
    current_step: Optional[str]
    error_message: Optional[str]
    extracted_text: Optional[str]
    extraction_metadata: Optional[Dict[str, Any]]
    needs_ocr: Optional[bool]
    ocr_text: Optional[str]
    ocr_confidence: Optional[float]
    ocr_backend: Optional[str]
    ocr_processing_time: Optional[float]
    storage_key: Optional[str]
    storage_path: Optional[str]
    compliance_checks: Optional[List[str]]
    processing_metrics: Optional[Dict[str, Any]]


class PatchValidationError(Exception):
    """Raised when patch validation fails."""
    pass


class PolicyViolationError(Exception):
    """Raised when a banned key is attempted to be modified."""
    pass


def _is_valid_type(value: Any, expected_type: Any) -> bool:
    """Check if value matches expected type, handling generics properly."""
    # Handle None for Optional types
    if value is None:
        return True
    
    # Handle basic types
    if expected_type in (str, int, float, bool):
        return isinstance(value, expected_type)
    
    # Handle Optional types
    if get_origin(expected_type) is type(None) or expected_type == Optional[str]:
        # For Optional[str], check if it's a string
        if expected_type == Optional[str]:
            return isinstance(value, str)
        # For other Optional types, get the inner type
        args = get_args(expected_type)
        if args:
            inner_type = args[0]
            return _is_valid_type(value, inner_type)
        return True
    
    # Handle List types
    if get_origin(expected_type) is list:
        if not isinstance(value, list):
            return False
        args = get_args(expected_type)
        if args:
            inner_type = args[0]
            return all(_is_valid_type(item, inner_type) for item in value)
        return True
    
    # Handle Dict types
    if get_origin(expected_type) is dict:
        if not isinstance(value, dict):
            return False
        args = get_args(expected_type)
        if args:
            key_type, value_type = args
            # For Dict[str, Any], we only check that keys are strings
            if key_type == str:
                return all(isinstance(k, str) for k in value.keys())
        return True
    
    # Default case - use isinstance
    try:
        return isinstance(value, expected_type)
    except TypeError:
        # If isinstance fails, assume it's valid (for complex generic types)
        return True


def merge_patch(
    state: OCRWorkflowState,
    patch: StatePatch,
    *,
    banned_keys: List[str] = ["tenant_id"],
    conflict_policy: str = "error"
) -> OCRWorkflowState:
    """Merge patch into state with validation and conflict detection.
    
    Args:
        state: Current workflow state
        patch: Patch containing field updates
        banned_keys: Keys that cannot be modified (default: ["tenant_id"])
        conflict_policy: How to handle conflicts ("error", "warn", "ignore")
    
    Returns:
        New state with patch applied
        
    Raises:
        PolicyViolationError: If banned key is modified
        PatchValidationError: If patch validation fails
    """
    # Validate banned keys
    for banned_key in banned_keys:
        if banned_key in patch:
            raise PolicyViolationError(
                f"Attempted to modify banned key: {banned_key}. "
                f"Banned keys: {banned_keys}"
            )
    
    # Validate patch structure
    if not isinstance(patch, dict):
        raise PatchValidationError(f"Patch must be dict, got {type(patch)}")
    
    # Check for invalid keys
    valid_keys = set(OCRWorkflowState.__annotations__.keys())
    patch_keys = set(patch.keys())
    invalid_keys = patch_keys - valid_keys
    
    if invalid_keys:
        raise PatchValidationError(
            f"Invalid patch keys: {invalid_keys}. "
            f"Valid keys: {valid_keys}"
        )
    
    # Apply patch with type validation
    state_dict = state.__dict__.copy()
    
    for key, value in patch.items():
        if value is not None:  # Only apply non-None values
            # Type validation
            expected_type = OCRWorkflowState.__annotations__[key]
            if not _is_valid_type(value, expected_type):
                raise PatchValidationError(
                    f"Type mismatch for {key}: expected {expected_type}, got {type(value)}"
                )
            
            state_dict[key] = value
    
    # Create new state object
    return OCRWorkflowState(**state_dict)


class PatchGuard:
    """Context manager and decorator for patch validation and metrics."""
    
    def __init__(self, node_name: str, trace_id: Optional[str] = None):
        self.node_name = node_name
        self.trace_id = trace_id or str(uuid.uuid4())
        self.start_time = time.time()
        self.patch_size = 0
        self.keys_changed = 0
        self.conflicts_detected = 0
        self.policy_violations = 0
    
    def __enter__(self):
        logger.info(
            "PatchGuard: Starting patch validation",
            node=self.node_name,
            trace_id=self.trace_id
        )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = (time.time() - self.start_time) * 1000  # Convert to ms
        
        # Log results
        logger.info(
            "PatchGuard: Patch validation completed",
            node=self.node_name,
            trace_id=self.trace_id,
            patch_size=self.patch_size,
            keys_changed=self.keys_changed,
            conflicts_detected=self.conflicts_detected,
            policy_violations=self.policy_violations,
            duration_ms=duration
        )
        
        # TODO: Emit metrics when metrics system is available
        # metrics.increment("graph.patch_size", self.patch_size)
        # metrics.increment("graph.keys_changed", self.keys_changed)
        # metrics.increment("graph.conflicts", self.conflicts_detected)
        # metrics.increment("graph.policy_violations", self.policy_violations)
        # metrics.record("graph.node_latency_ms", duration)
        
        return False  # Don't suppress exceptions
    
    def validate_patch(self, patch: StatePatch) -> StatePatch:
        """Validate and record patch metrics."""
        if not isinstance(patch, dict):
            raise PatchValidationError(f"Patch must be dict, got {type(patch)}")
        
        self.patch_size = len(patch)
        self.keys_changed = len([k for k, v in patch.items() if v is not None])
        
        # Check for empty patches
        if self.patch_size == 0:
            logger.warning(
                "PatchGuard: Empty patch detected",
                node=self.node_name,
                trace_id=self.trace_id
            )
        
        return patch


def returns_patch(node_name: str):
    """Decorator to ensure sub-agents return valid patches.
    
    Args:
        node_name: Name of the sub-agent for logging and metrics
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            trace_id = str(uuid.uuid4())
            
            with PatchGuard(node_name, trace_id) as guard:
                # Execute the sub-agent
                result = await func(*args, **kwargs)
                
                # Validate result is a patch
                if not isinstance(result, dict):
                    raise PatchValidationError(
                        f"Sub-agent {node_name} must return StatePatch (dict), got {type(result)}"
                    )
                
                # Validate patch structure
                validated_patch = guard.validate_patch(result)
                
                # Log patch contents (with PII redaction)
                redacted_patch = _redact_pii(validated_patch)
                logger.debug(
                    "Sub-agent patch generated",
                    node=node_name,
                    trace_id=trace_id,
                    patch=redacted_patch
                )
                
                return validated_patch
        
        return wrapper
    return decorator


def _redact_pii(patch: StatePatch) -> StatePatch:
    """Redact potentially sensitive values from patch for logging."""
    redacted = patch.copy()
    
    # Redact text content that might contain PII
    sensitive_keys = ["extracted_text", "ocr_text", "error_message"]
    for key in sensitive_keys:
        if key in redacted and redacted[key]:
            if isinstance(redacted[key], str):
                redacted[key] = f"[REDACTED: {len(redacted[key])} chars]"
    
    # Redact processing metrics that might contain sensitive data
    if "processing_metrics" in redacted and redacted["processing_metrics"]:
        redacted["processing_metrics"] = "[REDACTED: processing_metrics]"
    
    return redacted


def reconstruct_final_state(
    initial_state: OCRWorkflowState,
    patches: List[StatePatch]
) -> OCRWorkflowState:
    """Reconstruct final state from initial state and accumulated patches.
    
    Args:
        initial_state: Starting state object
        patches: List of patches to apply in sequence
        
    Returns:
        Final typed OCRWorkflowState object
    """
    current_state = initial_state
    
    for i, patch in enumerate(patches):
        try:
            current_state = merge_patch(
                current_state, 
                patch,
                banned_keys=["tenant_id"],
                conflict_policy="error"
            )
        except (PatchValidationError, PolicyViolationError) as e:
            logger.error(
                "Failed to apply patch during state reconstruction",
                patch_index=i,
                patch=patch,
                error=str(e)
            )
            raise
    
    logger.info(
        "State reconstruction completed",
        initial_workflow_id=initial_state.workflow_id,
        patches_applied=len(patches),
        final_step=current_state.current_step
    )
    
    return current_state
