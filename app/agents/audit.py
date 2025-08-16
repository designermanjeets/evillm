"""Audit trace service for workflow execution tracking."""

import json
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class StepTiming:
    """Timing information for a workflow step."""
    step_name: str
    start_time: float
    end_time: float
    duration_ms: float
    status: str  # "started", "completed", "failed"


@dataclass
class PatchInfo:
    """Information about state patches."""
    step_name: str
    keys_changed: List[str]
    patch_size: int
    conflicts_detected: int


@dataclass
class CitationUsage:
    """Citation usage tracking."""
    citation_uid: str
    email_id: str
    chunk_uid: str
    score: float
    used_in_draft: bool


@dataclass
class AuditTrace:
    """Complete audit trace for workflow execution."""
    workflow_id: str
    tenant_id: str
    email_id: str
    query: str
    start_time: datetime
    end_time: datetime
    total_duration_ms: float
    step_timings: List[StepTiming]
    patch_info: List[PatchInfo]
    citation_usage: List[CitationUsage]
    performance_metrics: Dict[str, Any]
    final_state: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), default=str, indent=2)


class AuditService:
    """Service for compiling and exporting audit traces."""
    
    def __init__(self):
        self.metrics = {
            "drafts_started_total": 0,
            "drafts_failed_eval_total": 0,
            "drafts_stream_latency_ms_p95": 0.0,
            "retrieval_hit_k": 0.0,
            "eval_scores_grounding_avg": 0.0,
            "eval_scores_completeness_avg": 0.0,
            "eval_scores_tone_avg": 0.0,
            "eval_scores_policy_avg": 0.0
        }
        self._step_timings: Dict[str, StepTiming] = {}
        self._patch_info: List[PatchInfo] = []
        self._citation_usage: List[CitationUsage] = []
    
    def start_step(self, step_name: str) -> None:
        """Start timing a workflow step."""
        self._step_timings[step_name] = StepTiming(
            step_name=step_name,
            start_time=time.time(),
            end_time=0.0,
            duration_ms=0.0,
            status="started"
        )
    
    def end_step(self, step_name: str, status: str = "completed") -> None:
        """End timing a workflow step."""
        if step_name in self._step_timings:
            step_timing = self._step_timings[step_name]
            step_timing.end_time = time.time()
            step_timing.duration_ms = (step_timing.end_time - step_timing.start_time) * 1000
            step_timing.status = status
    
    def record_patch(self, step_name: str, keys_changed: List[str], patch_size: int, conflicts_detected: int = 0) -> None:
        """Record information about a state patch."""
        patch_info = PatchInfo(
            step_name=step_name,
            keys_changed=keys_changed,
            patch_size=patch_size,
            conflicts_detected=conflicts_detected
        )
        self._patch_info.append(patch_info)
    
    def record_citation_usage(self, citations: List[Any], used_citations: List[Any]) -> None:
        """Record citation usage information."""
        # Track all citations
        for citation in citations:
            citation_usage = CitationUsage(
                citation_uid=getattr(citation, 'chunk_uid', 'unknown'),
                email_id=getattr(citation, 'email_id', 'unknown'),
                chunk_uid=getattr(citation, 'chunk_uid', 'unknown'),
                score=getattr(citation, 'score', 0.0),
                used_in_draft=False
            )
            self._citation_usage.append(citation_usage)
        
        # Mark used citations
        used_uids = {getattr(c, 'chunk_uid', 'unknown') for c in used_citations}
        for citation_usage in self._citation_usage:
            if citation_usage.citation_uid in used_uids:
                citation_usage.used_in_draft = True
    
    def update_metrics(self, metrics: Dict[str, Any]) -> None:
        """Update performance metrics."""
        for key, value in metrics.items():
            if key in self.metrics:
                if isinstance(value, (int, float)):
                    self.metrics[key] = value
                elif isinstance(value, list) and len(value) > 0:
                    # For lists, calculate average
                    if all(isinstance(v, (int, float)) for v in value):
                        self.metrics[key] = sum(value) / len(value)
    
    def export_trace(
        self, 
        workflow_id: str,
        tenant_id: str,
        email_id: str,
        query: str,
        start_time: datetime,
        final_state: Any
    ) -> AuditTrace:
        """Export complete audit trace."""
        end_time = datetime.now()
        total_duration_ms = (end_time - start_time).total_seconds() * 1000
        
        # Convert final state to dict for serialization
        if hasattr(final_state, 'to_dict'):
            final_state_dict = final_state.to_dict()
        elif hasattr(final_state, '__dict__'):
            final_state_dict = final_state.__dict__
        else:
            final_state_dict = str(final_state)
        
        # Calculate performance metrics
        if self._step_timings:
            durations = [timing.duration_ms for timing in self._step_timings.values() if timing.duration_ms > 0]
            if durations:
                # Calculate p95 latency
                sorted_durations = sorted(durations)
                p95_index = int(len(sorted_durations) * 0.95)
                self.metrics["drafts_stream_latency_ms_p95"] = sorted_durations[p95_index]
        
        # Calculate citation hit rate
        if self._citation_usage:
            total_citations = len(self._citation_usage)
            used_citations = sum(1 for c in self._citation_usage if c.used_in_draft)
            if total_citations > 0:
                self.metrics["retrieval_hit_k"] = used_citations / total_citations
        
        # Calculate average evaluation scores if available
        if hasattr(final_state, 'step_results') and 'eval_gate' in final_state.step_results:
            eval_data = final_state.step_results['eval_gate'].get('evaluation', {})
            scores = eval_data.get('scores', {})
            
            if 'grounding' in scores:
                self.metrics["eval_scores_grounding_avg"] = scores['grounding']
            if 'completeness' in scores:
                self.metrics["eval_scores_completeness_avg"] = scores['completeness']
            if 'tone' in scores:
                self.metrics["eval_scores_tone_avg"] = scores['tone']
            if 'policy' in scores:
                self.metrics["eval_scores_policy_avg"] = scores['policy']
        
        # Update draft counts
        self.metrics["drafts_started_total"] += 1
        
        # Check if evaluation failed
        if hasattr(final_state, 'validation_errors') and final_state.validation_errors:
            eval_failures = [e for e in final_state.validation_errors if 'Evaluation failed' in e]
            if eval_failures:
                self.metrics["drafts_failed_eval_total"] += 1
        
        audit_trace = AuditTrace(
            workflow_id=workflow_id,
            tenant_id=tenant_id,
            email_id=email_id,
            query=query,
            start_time=start_time,
            end_time=end_time,
            total_duration_ms=total_duration_ms,
            step_timings=list(self._step_timings.values()),
            patch_info=self._patch_info.copy(),
            citation_usage=self._citation_usage.copy(),
            performance_metrics=self.metrics.copy(),
            final_state=final_state_dict
        )
        
        return audit_trace
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        return self.metrics.copy()
    
    def reset(self) -> None:
        """Reset audit state for new workflow."""
        self._step_timings.clear()
        self._patch_info.clear()
        self._citation_usage.clear()
