"""Draft workflow orchestration using LangGraph with sub-agents."""

import asyncio
import json
import re
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, AsyncGenerator
from dataclasses import dataclass, replace
from collections import defaultdict
import structlog

logger = structlog.get_logger(__name__)

# Import services
from app.services.retriever import RetrieverAdapter, CitationItem
from app.services.llm import LLMClient, LLMMessage
from app.services.eval_gate import EvalGate
from app.agents.audit import AuditService

# Try to import LangGraph components
try:
    from langgraph.graph import StateGraph, END
    from langgraph.checkpoint.memory import MemorySaver
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    logger.warning("LangGraph not available, using stub mode")


@dataclass
class DraftWorkflowState:
    """State for the draft workflow with patch-only updates."""
    
    tenant_id: str
    email_id: str
    query: str
    current_step: str = "coordinator"
    step_results: Dict[str, Any] = None
    final_draft: Optional[str] = None
    citations: List[CitationItem] = None
    used_citations: List[CitationItem] = None
    validation_errors: List[str] = None
    created_at: datetime = None
    updated_at: datetime = None
    token_count: int = 0
    cost_estimate: float = 0.0
    audit_trace: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.step_results is None:
            self.step_results = {}
        if self.citations is None:
            self.citations = []
        if self.used_citations is None:
            self.used_citations = []
        if self.validation_errors is None:
            self.validation_errors = []
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()
    
    def patch(self, **updates) -> 'DraftWorkflowState':
        """Create new state with updates (immutable updates)."""
        return replace(self, **updates, updated_at=datetime.now())


class QueryAnalyzer:
    """Analyzes user queries to determine requirements and constraints."""
    
    def __init__(self, llm_client=None):
        self.llm_client = llm_client
    
    async def analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze query for intent, style, and requirements."""
        if self.llm_client:
            # Use the query analyzer
            return await self.llm_client.analyze_query(query)
        else:
            # Fallback to rule-based analysis
            return self._rule_based_analysis(query)
    
    def _rule_based_analysis(self, query: str) -> Dict[str, Any]:
        """Fallback rule-based query analysis."""
        query_lower = query.lower()
        
        # Detect style indicators
        style_indicators = {
            "formal": ["please", "kindly", "would you", "could you", "request"],
            "casual": ["hey", "hi", "thanks", "cool", "awesome"],
            "technical": ["implement", "configure", "deploy", "optimize", "integrate"],
            "urgent": ["asap", "urgent", "immediately", "critical", "emergency"]
        }
        
        detected_styles = {}
        for style, indicators in style_indicators.items():
            count = sum(1 for indicator in indicators if indicator in query_lower)
            if count > 0:
                detected_styles[style] = count
        
        # Determine primary style
        primary_style = max(detected_styles.items(), key=lambda x: x[1])[0] if detected_styles else "professional"
        
        return {
            "intent": "draft_email",
            "style": primary_style,
            "tone": "neutral",
            "key_points": ["Professional communication"],
            "constraints": [],
            "analysis_method": "rule_based"
        }


class NumericVerifier:
    """Verifies numeric claims against retrieved documents."""
    
    async def verify_numeric_claim(self, claim: Dict[str, Any], citations: List[Dict]) -> bool:
        """Verify if a numeric claim is supported by citations."""
        # Simple verification - check if any citation contains the numeric value
        claim_text = claim["text"]
        
        for citation in citations:
            content = citation.get("content", "")
            if claim_text in content:
                return True
        
        return False
    
    def extract_numeric_claims(self, text: str) -> List[Dict[str, Any]]:
        """Extract numeric claims from text."""
        # Simple numeric extraction
        numeric_pattern = r'(\d+(?:\.\d+)?)\s*([a-zA-Z%$€£¥]+)?'
        matches = re.findall(numeric_pattern, text)
        
        claims = []
        for match in matches:
            value, unit = match
            # Handle currency symbols that come before the number
            if unit == "$" and text.find(f"${value}") != -1:
                claims.append({
                    "text": f"${value}",
                    "value": float(value),
                    "unit": "$",
                    "confidence": 0.8
                })
            else:
                claims.append({
                    "text": f"{value} {unit}".strip(),
                    "value": float(value),
                    "unit": unit or "",
                    "confidence": 0.8
                })
        
        return claims


class ComplianceGuard:
    """Checks draft compliance with policies and guidelines."""
    
    def __init__(self, tenant_id: str):
        self.tenant_id = tenant_id
    
    async def check_compliance(self, draft: str) -> Dict[str, Any]:
        """Check draft against compliance policies."""
        violations = []
        compliance_score = 100.0
        
        # Check for sensitive information patterns
        sensitive_patterns = [
            (r'\b\d{3}-\d{2}-\d{4}\b', 30),  # SSN
            (r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b', 30),  # Credit card
        ]
        
        for pattern, penalty in sensitive_patterns:
            if re.search(pattern, draft):
                violations.append({
                    "type": "sensitive_info",
                    "description": "Draft contains potentially sensitive information",
                    "severity": "high"
                })
                compliance_score -= penalty
        
        return {
            "violations": violations,
            "score": max(0.0, compliance_score),
            "passed": compliance_score >= 70.0
        }


class DraftWorkflow:
    """LangGraph workflow for orchestrating draft generation."""
    
    def __init__(self, config: Dict[str, Any], hybrid_search_service=None, llm_client=None, retriever=None):
        self.config = config
        self.hybrid_search_service = hybrid_search_service
        self.llm_client = llm_client or LLMClient(config)
        self.retriever = retriever or RetrieverAdapter(config)
        self.query_analyzer = QueryAnalyzer(self.llm_client)
        self.numeric_verifier = NumericVerifier()
        self.eval_gate = EvalGate(config)
        self.audit_service = AuditService()
        
        if LANGGRAPH_AVAILABLE:
            self.graph = self._build_graph()
        else:
            self.graph = None
            logger.warning("LangGraph not available, using stub mode")
    
    def _build_graph(self):
        """Build the LangGraph workflow."""
        if not LANGGRAPH_AVAILABLE:
            return None
        
        workflow = StateGraph(DraftWorkflowState)
        
        # Add nodes for each sub-agent
        workflow.add_node("coordinator", self._coordinator_node)
        workflow.add_node("analyzer", self._analyzer_node)
        workflow.add_node("retriever", self._retriever_node)
        workflow.add_node("numeric_verifier", self._numeric_verifier_node)
        workflow.add_node("drafter", self._drafter_node)
        workflow.add_node("compliance_guard", self._compliance_guard_node)
        workflow.add_node("eval_gate", self._eval_gate_node)
        
        # Define the linear workflow
        workflow.set_entry_point("coordinator")
        workflow.add_edge("coordinator", "analyzer")
        workflow.add_edge("analyzer", "retriever")
        workflow.add_edge("retriever", "numeric_verifier")
        workflow.add_edge("numeric_verifier", "drafter")
        workflow.add_edge("drafter", "compliance_guard")
        workflow.add_edge("compliance_guard", "eval_gate")
        workflow.add_edge("eval_gate", END)
        
        return workflow.compile()
    
    async def _coordinator_node(self, state: DraftWorkflowState) -> DraftWorkflowState:
        """Coordinate the draft workflow and set initial parameters."""
        try:
            # Analyze the query and set workflow parameters
            analysis = await self._analyze_query(state.query)
            
            # Set workflow configuration
            step_results = {
                "coordinator": {
                    "query_analysis": analysis,
                    "workflow_config": {
                        "max_tokens": self.config.get("max_tokens", 1000),
                        "temperature": self.config.get("temperature", 0.7),
                        "style": analysis.get("style", "professional"),
                        "tone": analysis.get("tone", "neutral")
                    }
                }
            }
            
            return state.patch(
                current_step="analyzer",
                step_results=step_results
            )
            
        except Exception as e:
            logger.error("Coordinator failed", error=str(e), tenant_id=state.tenant_id)
            return state.patch(
                validation_errors=state.validation_errors + [f"Coordinator error: {str(e)}"]
            )
    
    async def _analyzer_node(self, state: DraftWorkflowState) -> DraftWorkflowState:
        """Analyze the email context and requirements."""
        try:
            # Get email context (stub implementation)
            email_context = await self._get_email_context(state.email_id, state.tenant_id)
            
            # Analyze requirements
            requirements = await self._analyze_requirements(state.query, email_context)
            
            step_results = state.step_results.copy()
            step_results["analyzer"] = {
                "email_context": email_context,
                "requirements": requirements,
                "key_points": requirements.get("key_points", []),
                "constraints": requirements.get("constraints", [])
            }
            
            return state.patch(
                current_step="retriever",
                step_results=step_results
            )
            
        except Exception as e:
            logger.error("Analyzer failed", error=str(e), tenant_id=state.tenant_id)
            return state.patch(
                validation_errors=state.validation_errors + [f"Analyzer error: {str(e)}"]
            )
    
    async def _retriever_node(self, state: DraftWorkflowState) -> DraftWorkflowState:
        """Retrieve relevant information using hybrid search."""
        try:
            # Security: Validate and sanitize query
            if not state.query or not state.query.strip():
                raise ValueError("Query cannot be empty")
            
            # Security: Clamp k to prevent resource exhaustion
            max_k = min(20, self.config.get("retriever.max_results", 10))
            query = state.query.strip()[:500]  # Limit query length
            
            # Use RetrieverAdapter for search with tenant isolation
            citations = await self.retriever.retrieve(
                state.tenant_id,
                query,
                k=max_k
            )
            
            # Security: Ensure all citations belong to the same tenant
            tenant_citations = [c for c in citations if hasattr(c, 'tenant_id') and c.tenant_id == state.tenant_id]
            if len(tenant_citations) != len(citations):
                logger.warning("Some citations had different tenant_id, filtered out", 
                             tenant_id=state.tenant_id, 
                             total_citations=len(citations),
                             tenant_citations=len(tenant_citations))
                citations = tenant_citations
            
            # Track citation usage for audit
            self.audit_service.record_citation_usage(citations, [])
            
            step_results = state.step_results.copy()
            step_results["retriever"] = {
                "query": query,
                "results": citations,
                "total_hits": len(citations),
                "max_k": max_k,
                "query_length": len(query)
            }
            
            return state.patch(
                current_step="numeric_verifier",
                step_results=step_results,
                citations=citations
            )
            
        except Exception as e:
            logger.error("Retriever failed", error=str(e), tenant_id=state.tenant_id)
            return state.patch(
                validation_errors=state.validation_errors + [f"Retriever error: {str(e)}"]
            )
    
    async def _numeric_verifier_node(self, state: DraftWorkflowState) -> DraftWorkflowState:
        """Verify numeric claims and reject ungrounded numerics."""
        try:
            # Extract numeric claims from query and context
            numeric_claims = self.numeric_verifier.extract_numeric_claims(state.query)
            
            # Verify against retrieved documents
            verified_claims = []
            rejected_claims = []
            
            for claim in numeric_claims:
                if await self.numeric_verifier.verify_numeric_claim(claim, state.citations):
                    verified_claims.append(claim)
                else:
                    rejected_claims.append(claim)
            
            step_results = state.step_results.copy()
            step_results["numeric_verifier"] = {
                "verified_claims": verified_claims,
                "rejected_claims": rejected_claims,
                "verification_confidence": len(verified_claims) / len(numeric_claims) if numeric_claims else 1.0
            }
            
            # Add rejected claims to validation errors
            validation_errors = state.validation_errors.copy()
            for claim in rejected_claims:
                validation_errors.append(f"Unverified numeric claim: {claim['text']}")
            
            return state.patch(
                current_step="drafter",
                step_results=step_results,
                validation_errors=validation_errors
            )
            
        except Exception as e:
            logger.error("Numeric verifier failed", error=str(e), tenant_id=state.tenant_id)
            return state.patch(
                validation_errors=state.validation_errors + [f"Numeric verifier error: {str(e)}"]
            )
    
    async def _drafter_node(self, state: DraftWorkflowState) -> DraftWorkflowState:
        """Generate the draft response using LLM."""
        try:
            # Prepare prompt with context and citations
            prompt = self._build_draft_prompt(state)
            
            # Generate draft
            draft_response = await self._generate_draft(prompt, state)
            
            step_results = state.step_results.copy()
            step_results["drafter"] = {
                "prompt": prompt,
                "draft_response": draft_response,
                "token_count": draft_response.get("token_count", 0),
                "cost_estimate": draft_response.get("cost_estimate", 0.0)
            }
            
            # Update citation usage tracking with final used citations
            used_citations = draft_response.get("used_citations", [])
            self.audit_service.record_citation_usage(state.citations, used_citations)
            
            return state.patch(
                current_step="compliance_guard",
                step_results=step_results,
                final_draft=draft_response.get("content", ""),
                token_count=draft_response.get("token_count", 0),
                cost_estimate=draft_response.get("cost_estimate", 0.0),
                used_citations=used_citations
            )
            
        except Exception as e:
            logger.error("Drafter failed", error=str(e), tenant_id=state.tenant_id)
            return state.patch(
                validation_errors=state.validation_errors + [f"Drafter error: {str(e)}"]
            )
    
    async def _compliance_guard_node(self, state: DraftWorkflowState) -> DraftWorkflowState:
        """Check draft compliance with policies and guidelines."""
        try:
            # Check compliance rules
            compliance_guard = ComplianceGuard(state.tenant_id)
            compliance_checks = await compliance_guard.check_compliance(state.final_draft)
            
            step_results = state.step_results.copy()
            step_results["compliance_guard"] = {
                "compliance_checks": compliance_checks,
                "policy_violations": compliance_checks.get("violations", []),
                "compliance_score": compliance_checks.get("score", 0.0)
            }
            
            # Add compliance violations to validation errors
            validation_errors = state.validation_errors.copy()
            for violation in compliance_checks.get("violations", []):
                validation_errors.append(f"Compliance violation: {violation['description']}")
            
            return state.patch(
                current_step="eval_gate",
                step_results=step_results,
                validation_errors=validation_errors
            )
            
        except Exception as e:
            logger.error("Compliance guard failed", error=str(e), tenant_id=state.tenant_id)
            return state.patch(
                validation_errors=state.validation_errors + [f"Compliance guard error: {str(e)}"]
            )
    
    async def _eval_gate_node(self, state: DraftWorkflowState) -> DraftWorkflowState:
        """Final evaluation gate for quality and safety."""
        try:
            # Get retrieval results for evaluation
            retrieval_results = state.step_results.get("retriever", {}).get("results", [])
            
            # Run evaluation using the evaluation gate service
            evaluation = await self.eval_gate.evaluate_draft(
                state.final_draft,
                state.used_citations,
                retrieval_results
            )
            
            step_results = state.step_results.copy()
            step_results["eval_gate"] = {
                "evaluation": {
                    "scores": {
                        "grounding": evaluation.scores.grounding,
                        "completeness": evaluation.scores.completeness,
                        "tone": evaluation.scores.tone,
                        "policy": evaluation.scores.policy
                    },
                    "passed": evaluation.passed,
                    "reasons": evaluation.reasons,
                    "overall_score": evaluation.overall_score
                },
                "overall_score": evaluation.overall_score,
                "passed": evaluation.passed
            }
            
            # Add evaluation failures to validation errors
            validation_errors = state.validation_errors.copy()
            if not evaluation.passed:
                for reason in evaluation.reasons:
                    validation_errors.append(f"Evaluation failed: {reason}")
            
            return state.patch(
                current_step="completed",
                step_results=step_results,
                validation_errors=validation_errors
            )
            
        except Exception as e:
            logger.error("Eval gate failed", error=str(e), tenant_id=state.tenant_id)
            return state.patch(
                validation_errors=state.validation_errors + [f"Eval gate error: {str(e)}"]
            )
    
    async def _analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze query for intent, style, and requirements."""
        # Use the query analyzer
        return await self.query_analyzer.analyze_query(query)
    
    async def _get_email_context(self, email_id: str, tenant_id: str) -> Dict[str, Any]:
        """Get email context (stub implementation)."""
        # In production, fetch from database
        return {
            "subject": "Sample Email Subject",
            "from_addr": "sender@example.com",
            "to_addrs": ["recipient@example.com"],
            "sent_at": datetime.now().isoformat(),
            "has_attachments": False
        }
    
    async def _analyze_requirements(self, query: str, email_context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze requirements based on query and context."""
        # Simple requirement analysis
        return {
            "key_points": ["Professional communication", "Clear response"],
            "constraints": ["Maintain professional tone", "Include relevant context"]
        }
    

    
    def _build_draft_prompt(self, state: DraftWorkflowState) -> str:
        """Build prompt for draft generation."""
        workflow_config = state.step_results.get("coordinator", {}).get("workflow_config", {})
        
        prompt = f"""
        Generate a professional email response based on the following:
        
        Query: {state.query}
        Style: {workflow_config.get('style', 'professional')}
        Tone: {workflow_config.get('tone', 'neutral')}
        Max Tokens: {workflow_config.get('max_tokens', 1000)}
        
        Context from search results:
        {self._format_citations_for_prompt(state.citations)}
        
        Requirements:
        {chr(10).join(state.step_results.get('analyzer', {}).get('requirements', {}).get('key_points', []))}
        
        Generate a professional, well-structured email response.
        """
        
        return prompt
    
    def _format_citations_for_prompt(self, citations: List[CitationItem]) -> str:
        """Format citations for inclusion in prompt."""
        if not citations:
            return "No relevant context found."
        
        formatted = []
        for i, citation in enumerate(citations[:3], 1):  # Limit to 3 citations
            formatted.append(f"{i}. Email {citation.email_id}: {citation.content_preview}")
        
        return "\n".join(formatted)
    
    async def _generate_draft(self, prompt: str, state: DraftWorkflowState) -> Dict[str, Any]:
        """Generate draft using LLM with citation tracking."""
        try:
            # Security: Sanitize prompt to prevent injection
            sanitized_prompt = self._sanitize_prompt(prompt)
            
            # Build messages for LLM with security constraints
            system_message = LLMMessage(
                role="system",
                content="""You are a professional email assistant. Generate a well-structured, 
                professional email response based on the user's query and the provided context. 
                Use ONLY the citations provided to ground your response in factual information.
                Do NOT make up information or use external knowledge.
                Respond only with evidence-backed facts from the provided citations."""
            )
            
            user_message = LLMMessage(role="user", content=sanitized_prompt)
            
            # Generate draft using LLM with streaming support
            response = await self.llm_client.chat_completion([system_message, user_message])
            
            # Security: Analyze response for citation usage and grounding
            used_citations = self._analyze_citation_usage(response.content, state.citations)
            
            # Security: Validate response doesn't contain ungrounded claims
            if not used_citations:
                logger.warning("Generated draft has no citations - may be ungrounded", 
                             tenant_id=state.tenant_id)
            
            return {
                "content": response.content,
                "token_count": response.tokens_used,
                "cost_estimate": response.cost_estimate,
                "used_citations": used_citations,
                "model_used": response.model_used,
                "grounding_score": len(used_citations) / max(len(state.citations), 1)
            }
            
        except Exception as e:
            logger.error("Draft generation failed", error=str(e), tenant_id=state.tenant_id)
            # Fallback to stub response
            return {
                "content": f"Stub draft response for: {state.query}",
                "token_count": len(prompt.split()),
                "cost_estimate": 0.0,
                "used_citations": [],
                "model_used": "stub",
                "grounding_score": 0.0
            }
    
    def _sanitize_prompt(self, prompt: str) -> str:
        """Sanitize prompt to prevent injection attacks."""
        # Remove potentially dangerous characters and limit length
        sanitized = prompt.replace("<script>", "").replace("javascript:", "")
        sanitized = sanitized.replace("data:", "").replace("vbscript:", "")
        return sanitized[:2000]  # Limit prompt length
    
    def _analyze_citation_usage(self, content: str, citations: List[CitationItem]) -> List[CitationItem]:
        """Analyze which citations were actually used in the generated content."""
        if not citations or not content:
            return []
        
        used_citations = []
        content_lower = content.lower()
        
        for citation in citations:
            # Check if citation content appears in the generated text
            citation_text = citation.content_preview.lower()
            if len(citation_text) > 10 and citation_text[:50] in content_lower:
                used_citations.append(citation)
            # Also check email_id references
            elif citation.email_id.lower() in content_lower:
                used_citations.append(citation)
        
        # Return top citations by relevance score
        used_citations.sort(key=lambda x: x.score, reverse=True)
        return used_citations[:min(5, len(used_citations))]  # Limit to top 5
    

    
    async def execute(self, tenant_id: str, email_id: str, query: str) -> DraftWorkflowState:
        """Execute the draft workflow."""
        # Start audit tracking
        workflow_id = f"{tenant_id}_{email_id}_{int(time.time())}"
        start_time = datetime.now()
        
        # Reset audit service for new workflow
        self.audit_service.reset()
        
        try:
            if self.graph:
                # Use LangGraph workflow
                initial_state = DraftWorkflowState(tenant_id, email_id, query)
                final_state = await self.graph.ainvoke(initial_state)
            else:
                # Use stub workflow
                final_state = await self._execute_stub_workflow(tenant_id, email_id, query)
            
            # Export audit trace
            audit_trace = self.audit_service.export_trace(
                workflow_id, tenant_id, email_id, query, start_time, final_state
            )
            
            # Attach audit trace to final state
            final_state = final_state.patch(audit_trace=audit_trace.to_dict())
            
            return final_state
            
        except Exception as e:
            logger.error("Workflow execution failed", error=str(e), tenant_id=tenant_id)
            # Create error state with audit trace
            error_state = DraftWorkflowState(tenant_id, email_id, query)
            error_state = error_state.patch(
                validation_errors=[f"Workflow execution failed: {str(e)}"],
                current_step="failed"
            )
            
            # Export audit trace for failed workflow
            audit_trace = self.audit_service.export_trace(
                workflow_id, tenant_id, email_id, query, start_time, error_state
            )
            
            error_state = error_state.patch(audit_trace=audit_trace.to_dict())
            return error_state
    
    async def _execute_stub_workflow(self, tenant_id: str, email_id: str, query: str) -> DraftWorkflowState:
        """Execute stub workflow when LangGraph is not available."""
        state = DraftWorkflowState(tenant_id, email_id, query)
        
        # Execute each step sequentially
        state = await self._coordinator_node(state)
        state = await self._analyzer_node(state)
        state = await self._retriever_node(state)
        state = await self._numeric_verifier_node(state)
        state = await self._drafter_node(state)
        state = await self._compliance_guard_node(state)
        state = await self._eval_gate_node(state)
        
        return state


class DraftStreamingService:
    """Handles streaming output for draft generation."""
    
    def __init__(self, workflow: DraftWorkflow):
        self.workflow = workflow
    
    async def stream_draft(self, tenant_id: str, email_id: str, query: str) -> AsyncGenerator[str, None]:
        """Stream draft generation progress."""
        try:
            # Initialize workflow state
            initial_state = DraftWorkflowState(tenant_id, email_id, query)
            
            if self.workflow.graph:
                # Stream workflow execution
                async for event in self.workflow.graph.astream(initial_state):
                    if event["type"] == "on_chain_start":
                        yield f"data: {json.dumps({'type': 'step_start', 'step': event['name']})}\n\n"
                    
                    elif event["type"] == "on_chain_end":
                        step_result = event["outputs"]["step_results"].get(event["name"], {})
                        yield f"data: {json.dumps({'type': 'step_complete', 'step': event['name'], 'result': step_result})}\n\n"
                    
                    elif event["type"] == "on_chain_error":
                        error_msg = f"Error in step {event['name']}: {event['error']}"
                        yield f"data: {json.dumps({'type': 'error', 'step': event['name'], 'error': error_msg})}\n\n"
                
                # Send completion event
                final_state = await self.workflow.graph.ainvoke(initial_state)
                yield f"data: {json.dumps({'type': 'complete', 'draft': final_state.final_draft, 'citations': final_state.citations})}\n\n"
            else:
                # Stub streaming for non-LangGraph mode
                steps = ["coordinator", "analyzer", "retriever", "numeric_verifier", "drafter", "compliance_guard", "eval_gate"]
                
                for step in steps:
                    yield f"data: {json.dumps({'type': 'step_start', 'step': step})}\n\n"
                    await asyncio.sleep(0.01)  # Simulate processing time
                    yield f"data: {json.dumps({'type': 'step_complete', 'step': step, 'result': {'status': 'completed'}})}\n\n"
                
                # Execute workflow and send completion
                final_state = await self.workflow.execute(tenant_id, email_id, query)
                yield f"data: {json.dumps({'type': 'complete', 'draft': final_state.final_draft, 'citations': final_state.citations})}\n\n"
            
        except Exception as e:
            error_msg = f"Workflow execution failed: {str(e)}"
            yield f"data: {json.dumps({'type': 'error', 'error': error_msg})}\n\n"


# Factory function for creating draft workflow
def create_draft_workflow(config: Dict[str, Any] = None, hybrid_search_service=None, llm_client=None, retriever=None) -> DraftWorkflow:
    """Create and configure draft workflow."""
    if config is None:
        config = {
            "max_tokens": 1000,
            "temperature": 0.7,
            "enable_streaming": True
        }
    
    return DraftWorkflow(config, hybrid_search_service, llm_client, retriever)
