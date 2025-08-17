"""Evaluation gate service with fail-closed policy enforcement."""

import re
from typing import Dict, Any, List, Optional
import structlog
from pydantic import BaseModel

logger = structlog.get_logger(__name__)


class EvalScores(BaseModel):
    """Evaluation scores for different criteria."""
    grounding: float
    completeness: float
    tone: float
    policy: float


class EvalResult(BaseModel):
    """Evaluation result with scores and pass/fail status."""
    scores: EvalScores
    passed: bool
    reasons: List[str]
    overall_score: float


class EvalGate:
    """Evaluation gate with fail-closed policy enforcement."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled = config.get("eval.enabled", True)
        self.grounding_threshold = config.get("eval.thresholds.grounding", 0.70)
        self.completeness_threshold = config.get("eval.thresholds.completeness", 0.65)
        self.tone_threshold = config.get("eval.tone_threshold", 0.8)
        self.policy_threshold = config.get("eval.policy_threshold", 0.9)
        self.overall_threshold = config.get("eval.overall_threshold", 0.7)
        
        # Policy violation patterns
        self.policy_patterns = {
            "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
            "credit_card": r"\b\d{4}-\d{4}-\d{4}-\d{4}\b",
            "phone": r"\b\d{3}-\d{3}-\d{4}\b",
            "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "url": r"https?://[^\s]+",
            "ip_address": r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"
        }
        
        # Tone indicators
        self.tone_indicators = {
            "professional": ["sincerely", "regards", "best regards", "thank you", "appreciate"],
            "casual": ["hey", "hi", "thanks", "cool", "awesome", "great"],
            "formal": ["respectfully", "yours truly", "faithfully", "cordially"],
            "urgent": ["asap", "urgent", "immediately", "critical", "emergency"]
        }
    
    async def evaluate_draft(
        self, 
        draft_text: str, 
        used_citations: List[Dict[str, Any]], 
        retrieval_results: List[Dict[str, Any]]
    ) -> EvalResult:
        """Evaluate draft with fail-closed policy."""
        try:
            if not self.enabled:
                # If evaluation is disabled, always pass
                return EvalResult(
                    scores=EvalScores(
                        grounding=1.0,
                        completeness=1.0,
                        tone=1.0,
                        policy=1.0
                    ),
                    passed=True,
                    reasons=["Evaluation disabled"],
                    overall_score=1.0
                )
            
            # Calculate individual scores
            grounding_score = self._calculate_grounding_score(draft_text, used_citations, retrieval_results)
            completeness_score = self._calculate_completeness_score(draft_text)
            tone_score = self._calculate_tone_score(draft_text)
            policy_score = self._calculate_policy_score(draft_text)
            
            # Calculate overall score (weighted average)
            overall_score = (
                grounding_score * 0.4 +
                completeness_score * 0.3 +
                tone_score * 0.2 +
                policy_score * 0.1
            )
            
            # Determine pass/fail status
            passed = self._determine_pass_status(
                grounding_score, completeness_score, tone_score, policy_score, overall_score
            )
            
            # Generate reasons for the result
            reasons = self._generate_reasons(
                grounding_score, completeness_score, tone_score, policy_score, overall_score
            )
            
            return EvalResult(
                scores=EvalScores(
                    grounding=grounding_score,
                    completeness=completeness_score,
                    tone=tone_score,
                    policy=policy_score
                ),
                passed=passed,
                reasons=reasons,
                overall_score=overall_score
            )
            
        except Exception as e:
            logger.error("Evaluation failed", error=str(e))
            # Fail-closed: if evaluation fails, reject the draft
            return EvalResult(
                scores=EvalScores(
                    grounding=0.0,
                    completeness=0.0,
                    tone=0.0,
                    policy=0.0
                ),
                passed=False,
                reasons=[f"Evaluation error: {str(e)}"],
                overall_score=0.0
            )
    
    def _calculate_grounding_score(
        self, 
        draft_text: str, 
        used_citations: List[Dict[str, Any]], 
        retrieval_results: List[Dict[str, Any]]
    ) -> float:
        """Calculate grounding score based on citation usage."""
        if not used_citations:
            return 0.0  # No citations = no grounding
        
        if not retrieval_results:
            return 0.5  # Some citations but no retrieval context
        
        # Calculate citation coverage
        total_citations = len(retrieval_results)
        used_citation_count = len(used_citations)
        
        if total_citations == 0:
            return 0.0
        
        citation_coverage = used_citation_count / total_citations
        
        # Check if citations are actually referenced in the text
        citation_references = 0
        for citation in used_citations:
            # Simple check: see if citation content appears in draft
            if hasattr(citation, 'content_preview'):
                content_preview = citation.content_preview[:50]  # First 50 chars
                if content_preview.lower() in draft_text.lower():
                    citation_references += 1
        
        reference_score = citation_references / max(1, used_citation_count)
        
        # Combine coverage and reference scores
        grounding_score = (citation_coverage * 0.6) + (reference_score * 0.4)
        
        return min(1.0, max(0.0, grounding_score))
    
    def _calculate_completeness_score(self, draft_text: str) -> float:
        """Calculate completeness score based on content length and structure."""
        if not draft_text:
            return 0.0
        
        # Length-based scoring
        word_count = len(draft_text.split())
        if word_count < 10:
            length_score = 0.3
        elif word_count < 25:
            length_score = 0.6
        elif word_count < 50:
            length_score = 0.8
        else:
            length_score = 1.0
        
        # Structure-based scoring
        structure_score = 0.5  # Base score
        
        # Check for common email elements
        if re.search(r"dear|hello|hi", draft_text.lower()):
            structure_score += 0.1
        if re.search(r"sincerely|regards|thank you", draft_text.lower()):
            structure_score += 0.1
        if re.search(r"\n\n", draft_text):  # Paragraphs
            structure_score += 0.1
        if len(draft_text.split('.')) > 2:  # Multiple sentences
            structure_score += 0.1
        if len(draft_text.split('\n')) > 3:  # Multiple lines
            structure_score += 0.1
        
        # Cap structure score at 1.0
        structure_score = min(1.0, structure_score)
        
        # Combine scores
        completeness_score = (length_score * 0.6) + (structure_score * 0.4)
        
        return min(1.0, max(0.0, completeness_score))
    
    def _calculate_tone_score(self, draft_text: str) -> float:
        """Calculate tone score based on professional communication standards."""
        if not draft_text:
            return 0.0
        
        draft_lower = draft_text.lower()
        
        # Check for professional tone indicators
        professional_indicators = 0
        casual_indicators = 0
        formal_indicators = 0
        
        for tone, indicators in self.tone_indicators.items():
            for indicator in indicators:
                if indicator in draft_lower:
                    if tone == "professional":
                        professional_indicators += 1
                    elif tone == "casual":
                        casual_indicators += 1
                    elif tone == "formal":
                        formal_indicators += 1
        
        # Calculate tone balance
        total_indicators = professional_indicators + casual_indicators + formal_indicators
        
        if total_indicators == 0:
            return 0.7  # Neutral score for no indicators
        
        # Prefer professional tone
        if professional_indicators > 0:
            tone_score = 0.9
        elif formal_indicators > 0:
            tone_score = 0.8
        elif casual_indicators > 0:
            tone_score = 0.6
        else:
            tone_score = 0.7
        
        # Penalize excessive casual language
        if casual_indicators > professional_indicators + formal_indicators:
            tone_score *= 0.8
        
        return min(1.0, max(0.0, tone_score))
    
    def _calculate_policy_score(self, draft_text: str) -> float:
        """Calculate policy compliance score."""
        if not draft_text:
            return 0.0
        
        # Check for policy violations
        violations = []
        
        for policy_type, pattern in self.policy_patterns.items():
            matches = re.findall(pattern, draft_text)
            if matches:
                violations.append(f"{policy_type}: {len(matches)} matches")
        
        # Calculate base score
        if not violations:
            policy_score = 1.0
        elif len(violations) == 1:
            policy_score = 0.7
        elif len(violations) == 2:
            policy_score = 0.4
        else:
            policy_score = 0.1
        
        # Additional checks
        if "password" in draft_text.lower() or "secret" in draft_text.lower():
            policy_score *= 0.8
        
        if "internal" in draft_text.lower() and "confidential" in draft_text.lower():
            policy_score *= 0.9
        
        return min(1.0, max(0.0, policy_score))
    
    def _determine_pass_status(
        self, 
        grounding_score: float, 
        completeness_score: float, 
        tone_score: float, 
        policy_score: float, 
        overall_score: float
    ) -> bool:
        """Determine if draft passes evaluation."""
        # Fail-closed policy: must meet all thresholds
        if grounding_score < self.grounding_threshold:
            return False
        
        if completeness_score < self.completeness_threshold:
            return False
        
        if tone_score < self.tone_threshold:
            return False
        
        if policy_score < self.policy_threshold:
            return False
        
        if overall_score < self.overall_threshold:
            return False
        
        return True
    
    def _generate_reasons(
        self, 
        grounding_score: float, 
        completeness_score: float, 
        tone_score: float, 
        policy_score: float, 
        overall_score: float
    ) -> List[str]:
        """Generate reasons for evaluation result."""
        reasons = []
        
        if grounding_score < self.grounding_threshold:
            reasons.append(f"Grounding score {grounding_score:.2f} below threshold {self.grounding_threshold}")
        
        if completeness_score < self.completeness_threshold:
            reasons.append(f"Completeness score {completeness_score:.2f} below threshold {self.completeness_threshold}")
        
        if tone_score < self.tone_threshold:
            reasons.append(f"Tone score {tone_score:.2f} below threshold {self.tone_threshold}")
        
        if policy_score < self.policy_threshold:
            reasons.append(f"Policy score {policy_score:.2f} below threshold {self.policy_threshold}")
        
        if overall_score < self.overall_threshold:
            reasons.append(f"Overall score {overall_score:.2f} below threshold {self.overall_threshold}")
        
        if not reasons:
            reasons.append("All evaluation criteria passed")
        
        return reasons
