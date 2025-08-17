"""Custom exceptions for the application."""

from typing import Optional, Dict, Any


class SearchDependencyUnavailable(Exception):
    """Raised when a search dependency is unavailable in strict mode."""
    
    def __init__(self, service: str, action: str = "search", trace_id: Optional[str] = None, tenant_id: Optional[str] = None):
        self.service = service
        self.action = action
        self.trace_id = trace_id
        self.tenant_id = tenant_id
        self.message = f"Search dependency '{service}' unavailable for {action}"
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON response."""
        return {
            "error": "dependency_unavailable",
            "service": self.service,
            "action": self.action,
            "trace_id": self.trace_id,
            "tenant_id": self.tenant_id,
            "message": self.message
        }


class LlmDependencyUnavailable(Exception):
    """Raised when LLM dependency is unavailable in strict mode."""
    
    def __init__(self, provider: str, trace_id: Optional[str] = None, tenant_id: Optional[str] = None):
        self.provider = provider
        self.trace_id = trace_id
        self.tenant_id = tenant_id
        self.message = f"LLM provider '{provider}' unavailable"
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON response."""
        return {
            "error": "dependency_unavailable",
            "service": "llm",
            "provider": self.provider,
            "action": "draft",
            "trace_id": self.trace_id,
            "tenant_id": self.tenant_id,
            "message": self.message
        }


class NoEvidenceFoundError(Exception):
    """Raised when no evidence/citations are found for a query."""
    
    def __init__(self, query: str, trace_id: Optional[str] = None, tenant_id: Optional[str] = None):
        self.query = query
        self.trace_id = trace_id
        self.tenant_id = tenant_id
        self.message = f"No evidence found for query: {query}"
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON response."""
        return {
            "error": "no_evidence_found",
            "query": self.query,
            "trace_id": self.trace_id,
            "tenant_id": self.tenant_id,
            "message": self.message
        }


class EvalGateBlockedError(Exception):
    """Raised when EvalGate blocks a draft."""
    
    def __init__(self, scores: Dict[str, float], reasons: list, trace_id: Optional[str] = None, tenant_id: Optional[str] = None):
        self.scores = scores
        self.reasons = reasons
        self.trace_id = trace_id
        self.tenant_id = tenant_id
        self.message = f"Draft blocked by EvalGate: {', '.join(reasons)}"
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON response."""
        return {
            "error": "eval_gate_blocked",
            "scores": self.scores,
            "reasons": self.reasons,
            "trace_id": self.trace_id,
            "tenant_id": self.tenant_id,
            "message": self.message
        }
