"""Utility functions and decorators for LangGraph sub-agents."""

from typing import Dict, Any, List, Optional
import time
import uuid
import structlog
from functools import wraps

from .state_contract import StatePatch, PatchValidationError, PolicyViolationError

logger = structlog.get_logger(__name__)


def returns_patch(node_name: str):
    """Decorator to ensure sub-agents return valid patches.
    
    Args:
        node_name: Name of the sub-agent for logging and metrics
        
    Usage:
        @returns_patch("attachment_miner")
        async def process(self, state: OCRWorkflowState) -> StatePatch:
            # Process state and return only changes
            return {
                "current_step": "attachment_validated",
                "processing_metrics": {"validation_time": time.time()}
            }
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            trace_id = str(uuid.uuid4())
            start_time = time.time()
            
            logger.info(
                "Sub-agent execution started",
                node=node_name,
                trace_id=trace_id
            )
            
            try:
                # Execute the sub-agent
                result = await func(*args, **kwargs)
                
                # Validate result is a patch
                if not isinstance(result, dict):
                    raise PatchValidationError(
                        f"Sub-agent {node_name} must return StatePatch (dict), got {type(result)}"
                    )
                
                # Record metrics
                patch_size = len(result)
                keys_changed = len([k for k, v in result.items() if v is not None])
                duration = (time.time() - start_time) * 1000  # Convert to ms
                
                # Log patch contents (with PII redaction)
                redacted_patch = _redact_pii(result)
                logger.debug(
                    "Sub-agent patch generated",
                    node=node_name,
                    trace_id=trace_id,
                    patch=redacted_patch
                )
                
                # Log completion
                logger.info(
                    "Sub-agent execution completed",
                    node=node_name,
                    trace_id=trace_id,
                    patch_size=patch_size,
                    keys_changed=keys_changed,
                    duration_ms=duration
                )
                
                # TODO: Emit metrics when metrics system is available
                # metrics.increment("graph.patch_size", patch_size)
                # metrics.increment("graph.keys_changed", keys_changed)
                # metrics.record("graph.node_latency_ms", duration)
                
                return result
                
            except Exception as e:
                duration = (time.time() - start_time) * 1000
                logger.error(
                    "Sub-agent execution failed",
                    node=node_name,
                    trace_id=trace_id,
                    error=str(e),
                    duration_ms=duration
                )
                raise
        
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


def validate_patch_keys(patch: StatePatch, allowed_keys: List[str]) -> None:
    """Validate that patch only contains allowed keys.
    
    Args:
        patch: Patch to validate
        allowed_keys: List of keys that are allowed to be modified
        
    Raises:
        PatchValidationError: If patch contains disallowed keys
    """
    patch_keys = set(patch.keys())
    allowed_set = set(allowed_keys)
    disallowed = patch_keys - allowed_set
    
    if disallowed:
        raise PatchValidationError(
            f"Patch contains disallowed keys: {disallowed}. "
            f"Allowed keys: {allowed_keys}"
        )


def create_metrics_patch(metrics: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """Create a patch for processing metrics with optional prefix.
    
    Args:
        metrics: Metrics to add
        prefix: Optional prefix for metric keys
        
    Returns:
        Patch containing processing_metrics update
    """
    if prefix:
        prefixed_metrics = {f"{prefix}_{k}": v for k, v in metrics.items()}
    else:
        prefixed_metrics = metrics
    
    return {
        "processing_metrics": prefixed_metrics
    }


def merge_metrics_patch(
    existing_metrics: Dict[str, Any],
    new_metrics: Dict[str, Any]
) -> Dict[str, Any]:
    """Merge new metrics into existing metrics.
    
    Args:
        existing_metrics: Current metrics dictionary
        new_metrics: New metrics to add
        
    Returns:
        Merged metrics dictionary
    """
    merged = existing_metrics.copy()
    merged.update(new_metrics)
    return merged
