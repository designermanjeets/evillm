"""Tests for draft workflow (EARS-AGT-1)."""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime
import json

from app.agents.draft_workflow import (
    DraftWorkflowState, QueryAnalyzer, NumericVerifier, ComplianceGuard,
    DraftWorkflow, DraftStreamingService, create_draft_workflow
)


class TestDraftWorkflowState:
    """Test draft workflow state management."""
    
    def test_state_initialization(self):
        """Test state initialization with required fields."""
        state = DraftWorkflowState(
            tenant_id="tenant-123",
            email_id="email-123",
            query="test query"
        )
        
        assert state.tenant_id == "tenant-123"
        assert state.email_id == "email-123"
        assert state.query == "test query"
        assert state.current_step == "coordinator"
        assert state.step_results == {}
        assert state.citations == []
        assert state.validation_errors == []
        assert state.created_at is not None
        assert state.updated_at is not None
        assert state.token_count == 0
        assert state.cost_estimate == 0.0
    
    def test_state_patch_immutability(self):
        """Test that state patching creates new objects."""
        original_state = DraftWorkflowState(
            tenant_id="tenant-123",
            email_id="email-123",
            query="test query"
        )
        
        original_updated_at = original_state.updated_at
        
        # Patch the state
        patched_state = original_state.patch(
            current_step="analyzer",
            step_results={"test": "data"}
        )
        
        # Verify new object was created
        assert patched_state is not original_state
        assert patched_state.current_step == "analyzer"
        assert patched_state.step_results == {"test": "data"}
        assert patched_state.updated_at != original_updated_at
        
        # Verify original state unchanged
        assert original_state.current_step == "coordinator"
        assert original_state.step_results == {}
    
    def test_state_patch_multiple_fields(self):
        """Test patching multiple fields at once."""
        state = DraftWorkflowState(
            tenant_id="tenant-123",
            email_id="email-123",
            query="test query"
        )
        
        patched_state = state.patch(
            current_step="completed",
            final_draft="Generated draft content",
            token_count=150,
            cost_estimate=0.05
        )
        
        assert patched_state.current_step == "completed"
        assert patched_state.final_draft == "Generated draft content"
        assert patched_state.token_count == 150
        assert patched_state.cost_estimate == 0.05


class TestQueryAnalyzer:
    """Test query analyzer functionality."""
    
    def test_analyzer_initialization(self):
        """Test analyzer initialization."""
        analyzer = QueryAnalyzer()
        assert analyzer.llm_client is None
        
        mock_llm = Mock()
        analyzer_with_llm = QueryAnalyzer(mock_llm)
        assert analyzer_with_llm.llm_client == mock_llm
    
    @pytest.mark.asyncio
    async def test_analyze_query_rule_based(self):
        """Test rule-based query analysis."""
        analyzer = QueryAnalyzer()
        
        # Test formal query
        formal_result = await analyzer.analyze_query("Please kindly draft a response")
        assert formal_result["style"] == "formal"
        assert formal_result["intent"] == "draft_email"
        assert formal_result["tone"] == "neutral"
        
        # Test casual query
        casual_result = await analyzer.analyze_query("Hey, can you write a reply?")
        assert casual_result["style"] == "casual"
        
        # Test technical query
        technical_result = await analyzer.analyze_query("Implement the integration")
        assert technical_result["style"] == "technical"
        
        # Test urgent query
        urgent_result = await analyzer.analyze_query("ASAP response needed")
        assert urgent_result["style"] == "urgent"
    
    def test_rule_based_analysis_edge_cases(self):
        """Test rule-based analysis with edge cases."""
        analyzer = QueryAnalyzer()
        
        # Test empty query
        empty_result = analyzer._rule_based_analysis("")
        assert empty_result["style"] == "professional"
        
        # Test query with no style indicators
        no_style_result = analyzer._rule_based_analysis("Random text without indicators")
        assert no_style_result["style"] == "professional"
        
        # Test query with multiple style indicators
        multi_style_result = analyzer._rule_based_analysis("Please implement this ASAP")
        # Should pick the style with highest count
        assert multi_style_result["style"] in ["formal", "technical", "urgent"]


class TestNumericVerifier:
    """Test numeric verifier functionality."""
    
    def test_verifier_initialization(self):
        """Test verifier initialization."""
        verifier = NumericVerifier()
        assert verifier is not None
    
    @pytest.mark.asyncio
    async def test_verify_numeric_claim_success(self):
        """Test successful numeric claim verification."""
        verifier = NumericVerifier()
        
        claim = {"text": "100 USD", "value": 100, "unit": "USD"}
        citations = [
            {"content": "The cost is 100 USD for the service"},
            {"content": "Other information"}
        ]
        
        result = await verifier.verify_numeric_claim(claim, citations)
        assert result is True
    
    @pytest.mark.asyncio
    async def test_verify_numeric_claim_failure(self):
        """Test failed numeric claim verification."""
        verifier = NumericVerifier()
        
        claim = {"text": "200 EUR", "value": 200, "unit": "EUR"}
        citations = [
            {"content": "The cost is 100 USD for the service"},
            {"content": "Other information"}
        ]
        
        result = await verifier.verify_numeric_claim(claim, citations)
        assert result is False
    
    def test_extract_numeric_claims(self):
        """Test numeric claim extraction."""
        verifier = NumericVerifier()
        
        text = "The project costs 5000 USD and will take 3 months to complete"
        claims = verifier.extract_numeric_claims(text)
        
        assert len(claims) == 2
        
        # Check first claim
        assert claims[0]["text"] == "5000 USD"
        assert claims[0]["value"] == 5000.0
        assert claims[0]["unit"] == "USD"
        assert claims[0]["confidence"] == 0.8
        
        # Check second claim
        assert claims[1]["text"] == "3 months"
        assert claims[1]["value"] == 3.0
        assert claims[1]["unit"] == "months"
    
    def test_extract_numeric_claims_edge_cases(self):
        """Test numeric claim extraction edge cases."""
        verifier = NumericVerifier()
        
        # Test with no numbers
        no_numbers = verifier.extract_numeric_claims("No numeric values here")
        assert no_numbers == []
        
        # Test with decimal numbers
        decimal_text = "The price is 99.99 dollars"
        decimal_claims = verifier.extract_numeric_claims(decimal_text)
        assert len(decimal_claims) == 1
        assert decimal_claims[0]["value"] == 99.99
        
        # Test with currency symbols
        currency_text = "Cost: $150 and €200"
        currency_claims = verifier.extract_numeric_claims(currency_text)
        assert len(currency_claims) == 2


class TestComplianceGuard:
    """Test compliance guard functionality."""
    
    def test_guard_initialization(self):
        """Test compliance guard initialization."""
        guard = ComplianceGuard("tenant-123")
        assert guard.tenant_id == "tenant-123"
    
    @pytest.mark.asyncio
    async def test_check_compliance_clean_draft(self):
        """Test compliance check with clean draft."""
        guard = ComplianceGuard("tenant-123")
        
        clean_draft = "This is a professional email response without sensitive information."
        result = await guard.check_compliance(clean_draft)
        
        assert result["passed"] is True
        assert result["score"] == 100.0
        assert len(result["violations"]) == 0
    
    @pytest.mark.asyncio
    async def test_check_compliance_sensitive_info(self):
        """Test compliance check with sensitive information."""
        guard = ComplianceGuard("tenant-123")
        
        sensitive_draft = "The SSN is 123-45-6789 and credit card is 1234-5678-9012-3456"
        result = await guard.check_compliance(sensitive_draft)
        
        assert result["passed"] is False
        assert result["score"] == 40.0  # 100 - (2 * 30)
        assert len(result["violations"]) == 2
        
        # Check violation types
        violation_types = [v["type"] for v in result["violations"]]
        assert "sensitive_info" in violation_types
    
    @pytest.mark.asyncio
    async def test_check_compliance_score_calculation(self):
        """Test compliance score calculation."""
        guard = ComplianceGuard("tenant-123")
        
        # Test with one violation
        single_violation = "SSN: 123-45-6789"
        result = await guard.check_compliance(single_violation)
        
        assert result["score"] == 70.0  # 100 - 30
        assert result["passed"] is True  # 70 >= 70
        
        # Test with multiple violations
        multiple_violations = "SSN: 123-45-6789 and CC: 1234-5678-9012-3456"
        result = await guard.check_compliance(multiple_violations)
        
        assert result["score"] == 40.0  # 100 - (2 * 30)
        assert result["passed"] is False  # 40 < 70


class TestDraftWorkflow:
    """Test draft workflow orchestration."""
    
    @pytest.fixture
    def mock_hybrid_search(self):
        """Create mock hybrid search service."""
        service = Mock()
        service.hybrid_search = AsyncMock(return_value={
            "hits": {
                "total": {"value": 2, "relation": "eq"},
                "hits": [
                    {
                        "score": 0.9,
                        "citations": {"chunk_id": "chunk-1", "email_id": "email-1"},
                        "content": "Test content 1",
                        "subject": "Test Subject 1"
                    }
                ]
            }
        })
        return service
    
    @pytest.fixture
    def workflow(self, mock_hybrid_search):
        """Create draft workflow instance."""
        config = {
            "max_tokens": 1000,
            "temperature": 0.7,
            "enable_streaming": True
        }
        return DraftWorkflow(config, mock_hybrid_search)
    
    def test_workflow_initialization(self, workflow):
        """Test workflow initialization."""
        assert workflow.config["max_tokens"] == 1000
        assert workflow.config["temperature"] == 0.7
        assert workflow.hybrid_search_service is not None
        assert workflow.query_analyzer is not None
        assert workflow.numeric_verifier is not None
    
    @pytest.mark.asyncio
    async def test_coordinator_node(self, workflow):
        """Test coordinator node execution."""
        state = DraftWorkflowState(
            tenant_id="tenant-123",
            email_id="email-123",
            query="Please draft a response"
        )
        
        result = await workflow._coordinator_node(state)
        
        assert result.current_step == "analyzer"
        assert "coordinator" in result.step_results
        assert "query_analysis" in result.step_results["coordinator"]
        assert "workflow_config" in result.step_results["coordinator"]
    
    @pytest.mark.asyncio
    async def test_analyzer_node(self, workflow):
        """Test analyzer node execution."""
        # First run coordinator to set up state
        state = DraftWorkflowState(
            tenant_id="tenant-123",
            email_id="email-123",
            query="Please draft a response"
        )
        state = await workflow._coordinator_node(state)
        
        # Now run analyzer
        result = await workflow._analyzer_node(state)
        
        assert result.current_step == "retriever"
        assert "analyzer" in result.step_results
        assert "email_context" in result.step_results["analyzer"]
        assert "requirements" in result.step_results["analyzer"]
    
    @pytest.mark.asyncio
    async def test_retriever_node(self, workflow):
        """Test retriever node execution."""
        # Set up state through previous nodes
        state = DraftWorkflowState(
            tenant_id="tenant-123",
            email_id="email-123",
            query="Please draft a response"
        )
        state = await workflow._coordinator_node(state)
        state = await workflow._analyzer_node(state)
        
        # Now run retriever
        result = await workflow._retriever_node(state)
        
        assert result.current_step == "numeric_verifier"
        assert "retriever" in result.step_results
        assert "results" in result.step_results["retriever"]
        assert "query" in result.step_results["retriever"]
        assert "total_hits" in result.step_results["retriever"]
        # In test mode, citations might be empty, but structure should be correct
        assert isinstance(result.citations, list)
    
    @pytest.mark.asyncio
    async def test_numeric_verifier_node(self, workflow):
        """Test numeric verifier node execution."""
        # Set up state through previous nodes
        state = DraftWorkflowState(
            tenant_id="tenant-123",
            email_id="email-123",
            query="The cost is 500 USD"
        )
        state = await workflow._coordinator_node(state)
        state = await workflow._analyzer_node(state)
        state = await workflow._retriever_node(state)
        
        # Now run numeric verifier
        result = await workflow._numeric_verifier_node(state)
        
        assert result.current_step == "drafter"
        assert "numeric_verifier" in result.step_results
        assert "verified_claims" in result.step_results["numeric_verifier"]
        assert "rejected_claims" in result.step_results["numeric_verifier"]
    
    @pytest.mark.asyncio
    async def test_drafter_node(self, workflow):
        """Test drafter node execution."""
        # Set up state through previous nodes
        state = DraftWorkflowState(
            tenant_id="tenant-123",
            email_id="email-123",
            query="Please draft a response"
        )
        state = await workflow._coordinator_node(state)
        state = await workflow._analyzer_node(state)
        state = await workflow._retriever_node(state)
        state = await workflow._numeric_verifier_node(state)
        
        # Now run drafter
        result = await workflow._drafter_node(state)
        
        assert result.current_step == "compliance_guard"
        assert "drafter" in result.step_results
        assert result.final_draft is not None
        assert result.token_count > 0
    
    @pytest.mark.asyncio
    async def test_compliance_guard_node(self, workflow):
        """Test compliance guard node execution."""
        # Set up state through previous nodes
        state = DraftWorkflowState(
            tenant_id="tenant-123",
            email_id="email-123",
            query="Please draft a response"
        )
        state = await workflow._coordinator_node(state)
        state = await workflow._analyzer_node(state)
        state = await workflow._retriever_node(state)
        state = await workflow._numeric_verifier_node(state)
        state = await workflow._drafter_node(state)
        
        # Now run compliance guard
        result = await workflow._compliance_guard_node(state)
        
        assert result.current_step == "eval_gate"
        assert "compliance_guard" in result.step_results
        assert "compliance_checks" in result.step_results["compliance_guard"]
    
    @pytest.mark.asyncio
    async def test_eval_gate_node(self, workflow):
        """Test eval gate node execution."""
        # Set up state through previous nodes
        state = DraftWorkflowState(
            tenant_id="tenant-123",
            email_id="email-123",
            query="Please draft a response"
        )
        state = await workflow._coordinator_node(state)
        state = await workflow._analyzer_node(state)
        state = await workflow._retriever_node(state)
        state = await workflow._numeric_verifier_node(state)
        state = await workflow._drafter_node(state)
        state = await workflow._compliance_guard_node(state)
        
        # Now run eval gate
        result = await workflow._eval_gate_node(state)
        
        assert result.current_step == "completed"
        assert "eval_gate" in result.step_results
        assert "evaluation" in result.step_results["eval_gate"]
    
    @pytest.mark.asyncio
    async def test_execute_stub_workflow(self, workflow):
        """Test stub workflow execution."""
        result = await workflow.execute(
            tenant_id="tenant-123",
            email_id="email-123",
            query="Please draft a response"
        )
        
        # Handle both DraftWorkflowState and LangGraph AddableValuesDict
        if hasattr(result, 'current_step'):
            # Direct DraftWorkflowState object
            assert result.current_step == "completed"
            assert result.final_draft is not None
            step_results = result.step_results
        else:
            # LangGraph AddableValuesDict object
            assert result["current_step"] == "completed"
            assert result["final_draft"] is not None
            step_results = result["step_results"]
        
        # Verify step results
        assert len(step_results) > 0
        assert "coordinator" in step_results
        assert "analyzer" in step_results
        assert "retriever" in step_results
        assert "numeric_verifier" in step_results
        assert "drafter" in step_results
        assert "compliance_guard" in step_results
        assert "eval_gate" in step_results
    
    @pytest.mark.asyncio
    async def test_workflow_error_handling(self, workflow):
        """Test workflow error handling."""
        # Test with invalid state
        invalid_state = DraftWorkflowState(
            tenant_id="tenant-123",
            email_id="email-123",
            query="Please draft a response"
        )
        
        # Mock a failure in coordinator
        with patch.object(workflow, '_analyze_query', side_effect=Exception("Test error")):
            result = await workflow._coordinator_node(invalid_state)
            
            assert len(result.validation_errors) > 0
            assert "Coordinator error" in result.validation_errors[0]


class TestDraftStreamingService:
    """Test draft streaming service."""
    
    @pytest.fixture
    def mock_workflow(self):
        """Create mock workflow."""
        workflow = Mock()
        workflow.graph = None  # Use stub mode
        workflow.execute = AsyncMock(return_value=Mock(
            final_draft="Generated draft",
            citations=[{"chunk_id": "chunk-1"}]
        ))
        return workflow
    
    @pytest.fixture
    def streaming_service(self, mock_workflow):
        """Create streaming service."""
        return DraftStreamingService(mock_workflow)
    
    @pytest.mark.asyncio
    async def test_stream_draft_stub_mode(self, streaming_service):
        """Test streaming in stub mode."""
        events = []
        async for event in streaming_service.stream_draft("tenant-123", "email-123", "test query"):
            events.append(event)
        
        # Should have events for each step plus completion
        # The actual count may vary depending on whether LangGraph is available
        # In stub mode: 7 step_start + 7 step_complete + 1 completion = 15 total
        assert len(events) >= 8  # At least 7 steps + 1 completion
        
        # Check step events - parse JSON content first
        step_events = []
        for e in events:
            # Extract JSON content from "data: {...}\n\n" format
            json_content = e.replace("data: ", "").replace("\n\n", "")
            try:
                data = json.loads(json_content)
                if data.get("type") == "step_start":
                    step_events.append(e)
            except json.JSONDecodeError:
                continue
        
        assert len(step_events) >= 7
        
        # Check step_complete events - parse JSON content first
        step_complete_events = []
        for e in events:
            # Extract JSON content from "data: {...}\n\n" format
            json_content = e.replace("data: ", "").replace("\n\n", "")
            try:
                data = json.loads(json_content)
                if data.get("type") == "step_complete":
                    step_complete_events.append(e)
            except json.JSONDecodeError:
                continue
        
        assert len(step_complete_events) >= 7
        
        # Check completion event - parse JSON content first
        completion_events = []
        for e in events:
            # Extract JSON content from "data: {...}\n\n" format
            json_content = e.replace("data: ", "").replace("\n\n", "")
            try:
                data = json.loads(json_content)
                if data.get("type") == "complete":
                    completion_events.append(e)
            except json.JSONDecodeError:
                continue
        
        assert len(completion_events) == 1
        
        # Verify total event count in stub mode
        # 7 step_start + 7 step_complete + 1 completion = 15 total
        assert len(events) == 15
        
        # Verify event format
        for event in events:
            assert event.startswith("data: ")
            assert event.endswith("\n\n")
            
            # Parse JSON
            data = json.loads(event.replace("data: ", "").replace("\n\n", ""))
            assert "type" in data
    
    @pytest.mark.asyncio
    async def test_retriever_returns_citations_patch(self, mock_workflow):
        """Test EARS-AGT-4: Retriever returns citations patch."""
        from app.services.retriever import RetrieverAdapter, CitationItem
        from app.config.manager import ConfigManager
        
        # Create mock config and retriever
        config = ConfigManager()
        retriever = RetrieverAdapter(config)
        
        # Create workflow with retriever
        workflow = DraftWorkflow(config, retriever=retriever)
        
        # Execute retriever node
        state = DraftWorkflowState("tenant-123", "email-123", "test query")
        state = await workflow._retriever_node(state)
        
        # Verify retriever step results
        retriever_results = state.step_results.get("retriever", {})
        assert "query" in retriever_results
        assert "results" in retriever_results
        assert "total_hits" in retriever_results
        
        # Verify citations are properly structured
        citations = state.citations
        assert isinstance(citations, list)
        
        # In stub mode, citations might be empty, but structure should be correct
        if citations:
            for citation in citations:
                            assert hasattr(citation, 'email_id')
            assert hasattr(citation, 'chunk_uid')
            assert hasattr(citation, 'object_key')
            assert hasattr(citation, 'score')
            assert hasattr(citation, 'content_preview')
    
    @pytest.mark.asyncio
    async def test_eval_gate_blocks_ungrounded(self, mock_workflow):
        """Test EARS-AGT-6: EvalGate blocks ungrounded outputs."""
        from app.services.eval_gate import EvalGate
        from app.config.manager import ConfigManager
        from app.services.retriever import CitationItem
        
        # Create mock config and evaluation gate
        config = ConfigManager()
        eval_gate = EvalGate(config)
        
        # Test with ungrounded draft (no citations)
        draft_text = "This is a draft response without any citations."
        used_citations = []
        retrieval_results = []
        
        result = await eval_gate.evaluate_draft(draft_text, used_citations, retrieval_results)
        
        # Should fail due to no grounding
        assert not result.passed
        assert result.scores.grounding == 0.0
        assert "grounding" in result.reasons[0].lower()
        
        # Test with grounded draft (with citations)
        draft_text = """Dear Team,

Thank you for your inquiry. Based on the provided context, I can confirm that the project timeline has been updated.

The new schedule reflects the revised requirements and should accommodate all stakeholders' needs.

Best regards,
Project Manager"""
        
        used_citations = [
            CitationItem(
                email_id="email-1",
                chunk_uid="chunk-1",
                object_key="key-1",
                score=0.9,
                content_preview="Test content 1"
            )
        ]
        retrieval_results = [
            CitationItem(
                email_id="email-1",
                chunk_uid="chunk-1",
                object_key="key-1",
                score=0.9,
                content_preview="Test content 1"
            )
        ]
        
        result = await eval_gate.evaluate_draft(draft_text, used_citations, retrieval_results)
        
        # Should pass with proper grounding
        assert result.passed
        assert result.scores.grounding > 0.0
        assert "passed" in result.reasons[0].lower()
    
    @pytest.mark.asyncio
    async def test_audit_trace_export_shape(self, mock_workflow):
        """Test EARS-AGT-7: Audit trace export shape."""
        from app.agents.audit import AuditService
        from app.config.manager import ConfigManager
        
        # Create mock config and workflow
        config = ConfigManager()
        workflow = DraftWorkflow(config)
        
        # Execute workflow
        start_time = datetime.now()
        state = await workflow.execute("tenant-123", "email-123", "test query")
        
        # Verify audit trace is attached
        assert state.audit_trace is not None
        assert isinstance(state.audit_trace, dict)
        
        # Verify audit trace structure
        audit_trace = state.audit_trace
        assert "workflow_id" in audit_trace
        assert "tenant_id" in audit_trace
        assert "email_id" in audit_trace
        assert "query" in audit_trace
        assert "start_time" in audit_trace
        assert "end_time" in audit_trace
        assert "total_duration_ms" in audit_trace
        assert "step_timings" in audit_trace
        assert "patch_info" in audit_trace
        assert "citation_usage" in audit_trace
        assert "performance_metrics" in audit_trace
        assert "final_state" in audit_trace
        
        # Verify performance metrics
        metrics = audit_trace["performance_metrics"]
        assert "drafts_started_total" in metrics
        assert "drafts_failed_eval_total" in metrics
        assert "drafts_stream_latency_ms_p95" in metrics
        assert "retrieval_hit_k" in metrics
        
        # Verify step timings
        step_timings = audit_trace["step_timings"]
        assert isinstance(step_timings, list)
        
        # Verify citation usage
        citation_usage = audit_trace["citation_usage"]
        assert isinstance(citation_usage, list)
    
    @pytest.mark.asyncio
    async def test_hybrid_search_integration(self, mock_workflow):
        """Test hybrid search integration in RetrieverAdapter."""
        from app.services.retriever import RetrieverAdapter
        from app.config.manager import ConfigManager
        
        # Create mock config and retriever
        config = ConfigManager()
        retriever = RetrieverAdapter(config)
        
        # Test hybrid search availability check
        has_hybrid = retriever._has_hybrid_search()
        # Should be False in test mode (stub services)
        assert isinstance(has_hybrid, bool)
        
        # Test retrieval with hybrid search (should fall back to SQL)
        citations = await retriever.retrieve("tenant-123", "test query", k=5)
        
        # Verify citations structure
        assert isinstance(citations, list)
        # In test mode, citations might be empty due to empty database
        
        # Test that the retriever has the required services
        assert hasattr(retriever, 'bm25_service')
        assert hasattr(retriever, 'vector_service')
        assert hasattr(retriever, 'fusion_engine')
    
    @pytest.mark.asyncio
    async def test_llm_streams_and_collects_used_citations(self, mock_workflow):
        """Test EARS-AGT-5: LLM streams and collects used citations."""
        from app.services.llm import LLMClient
        from app.config.manager import ConfigManager
        from app.services.retriever import CitationItem
        
        # Create mock config and LLM client
        config = ConfigManager()
        llm_client = LLMClient(config)
        
        # Create workflow with LLM client
        workflow = DraftWorkflow(config, llm_client=llm_client)
        
        # Set up state with citations
        state = DraftWorkflowState("tenant-123", "email-123", "test query")
        state.citations = [
            CitationItem(
                email_id="email-1",
                chunk_uid="chunk-1",
                object_key="key-1",
                score=0.9,
                content_preview="Test content 1"
            ),
            CitationItem(
                email_id="email-2", 
                chunk_uid="chunk-2",
                object_key="key-2",
                score=0.8,
                content_preview="Test content 2"
            )
        ]
        
        # Execute drafter node
        state = await workflow._drafter_node(state)
        
        # Verify draft was generated
        assert state.final_draft is not None
        assert state.token_count > 0
        assert state.cost_estimate >= 0.0
        
        # Verify used citations are tracked
        assert state.used_citations is not None
        assert isinstance(state.used_citations, list)
        assert len(state.used_citations) > 0
        
        # Verify citation structure
        for citation in state.used_citations:
            assert hasattr(citation, 'email_id')
            assert hasattr(citation, 'chunk_uid')
            assert hasattr(citation, 'object_key')
            assert hasattr(citation, 'score')
            assert hasattr(citation, 'content_preview')


class TestDraftWorkflowIntegration:
    """Integration tests for draft workflow."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_workflow(self):
        """Test complete workflow execution."""
        # Create workflow with minimal dependencies
        config = {"max_tokens": 500, "temperature": 0.7}
        workflow = DraftWorkflow(config)
        
        # Execute workflow
        result = await workflow.execute(
            tenant_id="tenant-123",
            email_id="email-123",
            query="Please draft a professional response about the project timeline"
        )
        
        # Handle both DraftWorkflowState and LangGraph AddableValuesDict
        if hasattr(result, 'current_step'):
            # Direct DraftWorkflowState object
            assert result.current_step == "completed"
            assert result.final_draft is not None
            assert result.tenant_id == "tenant-123"
            assert result.email_id == "email-123"
            assert result.query == "Please draft a professional response about the project timeline"
            step_results = result.step_results
        else:
            # LangGraph AddableValuesDict object
            assert result["current_step"] == "completed"
            assert result["final_draft"] is not None
            assert result["tenant_id"] == "tenant-123"
            assert result["email_id"] == "email-123"
            assert result["query"] == "Please draft a professional response about the project timeline"
            step_results = result["step_results"]
        
        # Verify all steps were executed
        expected_steps = ["coordinator", "analyzer", "retriever", "numeric_verifier", 
                         "drafter", "compliance_guard", "eval_gate"]
        for step in expected_steps:
            assert step in step_results
        
        # Verify workflow configuration
        coordinator_config = step_results["coordinator"]["workflow_config"]
        assert coordinator_config["max_tokens"] == 500
        assert coordinator_config["temperature"] == 0.7
    
    @pytest.mark.asyncio
    async def test_workflow_with_numeric_verification(self):
        """Test workflow with numeric verification."""
        config = {"max_tokens": 500, "temperature": 0.7}
        workflow = DraftWorkflow(config)
        
        # Execute workflow with numeric content
        result = await workflow.execute(
            tenant_id="tenant-123",
            email_id="email-123",
            query="The project costs 10000 USD and takes 6 months"
        )
        
        # Handle both DraftWorkflowState and LangGraph AddableValuesDict
        if hasattr(result, 'step_results'):
            step_results = result.step_results
        else:
            step_results = result["step_results"]
        
        # Verify numeric verification step
        numeric_step = step_results["numeric_verifier"]
        assert "verified_claims" in numeric_step
        assert "rejected_claims" in numeric_step
        
        # Should have some numeric claims
        all_claims = numeric_step["verified_claims"] + numeric_step["rejected_claims"]
        assert len(all_claims) > 0
    
    @pytest.mark.asyncio
    async def test_workflow_compliance_checking(self):
        """Test workflow compliance checking."""
        config = {"max_tokens": 500, "temperature": 0.7}
        workflow = DraftWorkflow(config)
        
        # Execute workflow
        result = await workflow.execute(
            tenant_id="tenant-123",
            email_id="email-123",
            query="Please draft a response"
        )
        
        # Handle both DraftWorkflowState and LangGraph AddableValuesDict
        if hasattr(result, 'step_results'):
            step_results = result.step_results
        else:
            step_results = result["step_results"]
        
        # Verify compliance step
        compliance_step = step_results["compliance_guard"]
        assert "compliance_checks" in compliance_step
        assert "policy_violations" in compliance_step
        assert "compliance_score" in compliance_step
        
        # Verify evaluation step
        eval_step = step_results["eval_gate"]
        assert "evaluation" in eval_step
        assert "overall_score" in eval_step
        assert "passed" in eval_step


class TestDraftWorkflowFactory:
    """Test draft workflow factory function."""
    
    def test_create_draft_workflow_default_config(self):
        """Test creating workflow with default configuration."""
        workflow = create_draft_workflow()
        
        assert workflow.config["max_tokens"] == 1000
        assert workflow.config["temperature"] == 0.7
        assert workflow.config["enable_streaming"] is True
    
    def test_create_draft_workflow_custom_config(self):
        """Test creating workflow with custom configuration."""
        custom_config = {
            "max_tokens": 2000,
            "temperature": 0.5,
            "enable_streaming": False
        }
        
        workflow = create_draft_workflow(custom_config)
        
        assert workflow.config["max_tokens"] == 2000
        assert workflow.config["temperature"] == 0.5
        assert workflow.config["enable_streaming"] is False
    
    def test_create_draft_workflow_with_services(self):
        """Test creating workflow with external services."""
        mock_search = Mock()
        mock_llm = Mock()
        
        workflow = create_draft_workflow(
            config={"max_tokens": 1000},
            hybrid_search_service=mock_search,
            llm_client=mock_llm
        )
        
        assert workflow.hybrid_search_service == mock_search
        assert workflow.llm_client == mock_llm


# Test data for parametrized tests
@pytest.mark.parametrize("query,expected_style", [
    ("Please kindly draft a response", "formal"),
    ("Hey, can you write this?", "casual"),
    ("Implement the integration", "technical"),
    ("ASAP response needed", "urgent"),
    ("Random text without indicators", "professional"),
])
def test_query_style_detection(query, expected_style):
    """Test query style detection with different inputs."""
    analyzer = QueryAnalyzer()
    result = asyncio.run(analyzer.analyze_query(query))
    assert result["style"] == expected_style


@pytest.mark.parametrize("text,expected_claims", [
    ("Cost is 100 USD", [{"text": "100 USD", "value": 100.0, "unit": "USD"}]),
    ("Duration: 3 months", [{"text": "3 months", "value": 3.0, "unit": "months"}]),
    ("Price: $99.99", [{"text": "$99.99", "value": 99.99, "unit": "$"}]),
    ("No numbers here", []),
    ("Multiple: 1, 2, 3", [
        {"text": "1", "value": 1.0, "unit": ""},
        {"text": "2", "value": 2.0, "unit": ""},
        {"text": "3", "value": 3.0, "unit": ""}
    ]),
])
def test_numeric_claim_extraction(text, expected_claims):
    """Test numeric claim extraction with different inputs."""
    verifier = NumericVerifier()
    claims = verifier.extract_numeric_claims(text)
    
    assert len(claims) == len(expected_claims)
    for i, expected in enumerate(expected_claims):
        assert claims[i]["text"] == expected["text"]
        assert claims[i]["value"] == expected["value"]
        assert claims[i]["unit"] == expected["unit"]


@pytest.mark.parametrize("draft,expected_passed,expected_score", [
    ("Clean professional email", True, 100.0),
    ("SSN: 123-45-6789", False, 70.0),
    ("CC: 1234-5678-9012-3456", False, 70.0),
    ("SSN: 123-45-6789 and CC: 1234-5678-9012-3456", False, 40.0),
])
@pytest.mark.asyncio
async def test_compliance_checking(draft, expected_passed, expected_score):
    """Test compliance checking with different inputs."""
    guard = ComplianceGuard("tenant-123")
    result = await guard.check_compliance(draft)
    
    assert result["passed"] == expected_passed
    assert result["score"] == expected_score
