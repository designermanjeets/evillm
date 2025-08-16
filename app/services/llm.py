"""LLM client service with config-driven provider selection and streaming support."""

import asyncio
import json
from typing import List, Dict, Any, Optional, AsyncGenerator
import structlog
from pydantic import BaseModel

from app.config.manager import ConfigManager

logger = structlog.get_logger(__name__)


class LLMMessage(BaseModel):
    """LLM message with role and content."""
    role: str  # "system", "user", "assistant"
    content: str


class LLMResponse(BaseModel):
    """LLM response with content and metadata."""
    content: str
    tokens_used: int
    cost_estimate: float
    model_used: str


class LLMClient:
    """Config-driven LLM client with streaming support."""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.provider = config.get("models.llm.provider", "stub")
        self.model = config.get("models.llm.model", "gpt-4")
        self.max_tokens = config.get("models.llm.max_tokens", 1000)
        self.temperature = config.get("models.llm.temperature", 0.7)
        
        # Initialize provider-specific client
        self._client = self._init_client()
    
    def _init_client(self):
        """Initialize provider-specific LLM client."""
        if self.provider == "openai":
            return self._init_openai_client()
        elif self.provider == "anthropic":
            return self._init_anthropic_client()
        elif self.provider == "local":
            return self._init_local_client()
        else:
            return self._init_stub_client()
    
    def _init_openai_client(self):
        """Initialize OpenAI client."""
        try:
            import openai
            api_key = self.config.get("models.llm.openai.api_key")
            if not api_key:
                logger.warning("OpenAI API key not configured, falling back to stub")
                return self._init_stub_client()
            
            client = openai.AsyncOpenAI(api_key=api_key)
            return {"type": "openai", "client": client}
        except ImportError:
            logger.warning("OpenAI client not available, falling back to stub")
            return self._init_stub_client()
    
    def _init_anthropic_client(self):
        """Initialize Anthropic client."""
        try:
            import anthropic
            api_key = self.config.get("models.llm.anthropic.api_key")
            if not api_key:
                logger.warning("Anthropic API key not configured, falling back to stub")
                return self._init_stub_client()
            
            client = anthropic.AsyncAnthropic(api_key=api_key)
            return {"type": "anthropic", "client": client}
        except ImportError:
            logger.warning("Anthropic client not available, falling back to stub")
            return self._init_stub_client()
    
    def _init_local_client(self):
        """Initialize local LLM client."""
        try:
            # This would integrate with local models like Ollama, vLLM, etc.
            logger.info("Local LLM client initialized")
            return {"type": "local", "client": None}
        except Exception as e:
            logger.warning(f"Local LLM client failed: {e}, falling back to stub")
            return self._init_stub_client()
    
    def _init_stub_client(self):
        """Initialize stub client for testing."""
        logger.info("Using stub LLM client")
        return {"type": "stub", "client": None}
    
    async def stream_chat(
        self, 
        messages: List[LLMMessage], 
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Stream chat completion with real-time token generation."""
        try:
            if self._client["type"] == "openai":
                async for token in self._stream_openai(messages, tools, **kwargs):
                    yield token
            elif self._client["type"] == "anthropic":
                async for token in self._stream_anthropic(messages, tools, **kwargs):
                    yield token
            elif self._client["type"] == "local":
                async for token in self._stream_local(messages, tools, **kwargs):
                    yield token
            else:
                async for token in self._stream_stub(messages, tools, **kwargs):
                    yield token
                    
        except Exception as e:
            logger.error("LLM streaming failed", error=str(e))
            yield f"Error: {str(e)}"
    
    async def _stream_openai(self, messages: List[LLMMessage], tools: Optional[List[Dict[str, Any]]] = None, **kwargs):
        """Stream OpenAI completion."""
        try:
            client = self._client["client"]
            
            # Convert messages to OpenAI format
            openai_messages = [{"role": msg.role, "content": msg.content} for msg in messages]
            
            # Prepare tools if provided
            openai_tools = None
            if tools:
                openai_tools = tools
            
            # Stream completion
            stream = await client.chat.completions.create(
                model=self.model,
                messages=openai_messages,
                tools=openai_tools,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stream=True,
                **kwargs
            )
            
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error("OpenAI streaming failed", error=str(e))
            yield f"OpenAI Error: {str(e)}"
    
    async def _stream_anthropic(self, messages: List[LLMMessage], tools: Optional[List[Dict[str, Any]]] = None, **kwargs):
        """Stream Anthropic completion."""
        try:
            client = self._client["client"]
            
            # Convert messages to Anthropic format
            anthropic_messages = []
            for msg in messages:
                if msg.role == "system":
                    anthropic_messages.append({"role": "user", "content": f"System: {msg.content}"})
                else:
                    anthropic_messages.append({"role": msg.role, "content": msg.content})
            
            # Stream completion
            stream = await client.messages.create(
                model=self.model,
                messages=anthropic_messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stream=True,
                **kwargs
            )
            
            async for chunk in stream:
                if chunk.delta.content:
                    for content_block in chunk.delta.content:
                        if content_block.type == "text":
                            yield content_block.text
                            
        except Exception as e:
            logger.error("Anthropic streaming failed", error=str(e))
            yield f"Anthropic Error: {str(e)}"
    
    async def _stream_local(self, messages: List[LLMMessage], tools: Optional[List[Dict[str, Any]]] = None, **kwargs):
        """Stream local LLM completion."""
        # This would integrate with local models
        # For now, fall back to stub
        async for token in self._stream_stub(messages, tools, **kwargs):
            yield token
    
    async def _stream_stub(self, messages: List[LLMMessage], tools: Optional[List[Dict[str, Any]]] = None, **kwargs):
        """Stream stub completion for testing."""
        # Get the last user message
        user_message = next((msg.content for msg in reversed(messages) if msg.role == "user"), "No query")
        
        # Generate stub response
        stub_response = f"Stub LLM response to: {user_message}"
        
        # Stream tokens with small delays to simulate real streaming
        for i, char in enumerate(stub_response):
            yield char
            if i % 10 == 0:  # Add small delay every 10 characters
                await asyncio.sleep(0.01)
    
    async def chat_completion(
        self, 
        messages: List[LLMMessage], 
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> LLMResponse:
        """Non-streaming chat completion."""
        try:
            # Collect streamed tokens
            content = ""
            async for token in self.stream_chat(messages, tools, **kwargs):
                content += token
            
            # Calculate token count (rough estimate)
            tokens_used = len(content.split())
            
            # Estimate cost based on provider and model
            cost_estimate = self._estimate_cost(tokens_used)
            
            return LLMResponse(
                content=content,
                tokens_used=tokens_used,
                cost_estimate=cost_estimate,
                model_used=self.model
            )
            
        except Exception as e:
            logger.error("LLM chat completion failed", error=str(e))
            return LLMResponse(
                content=f"Error: {str(e)}",
                tokens_used=0,
                cost_estimate=0.0,
                model_used=self.model
            )
    
    def _estimate_cost(self, tokens: int) -> float:
        """Estimate cost based on provider and model."""
        # Rough cost estimates (these would be more accurate in production)
        if self.provider == "openai":
            if "gpt-4" in self.model:
                return tokens * 0.00003  # $0.03 per 1K tokens
            else:
                return tokens * 0.000002  # $0.002 per 1K tokens
        elif self.provider == "anthropic":
            if "claude-3" in self.model:
                return tokens * 0.000015  # $0.015 per 1K tokens
            else:
                return tokens * 0.000008  # $0.008 per 1K tokens
        else:
            return 0.0  # Local models have no API cost
    
    async def analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze query for intent, style, and requirements."""
        try:
            system_message = LLMMessage(
                role="system",
                content="""Analyze the following query and extract:
                - Intent: What the user wants to accomplish
                - Style: Formal, casual, technical, urgent
                - Tone: Professional, friendly, authoritative
                - Key points: Main requirements or constraints
                - Constraints: Any limitations or special requirements
                
                Return a JSON object with these fields."""
            )
            
            user_message = LLMMessage(role="user", content=query)
            
            response = await self.chat_completion([system_message, user_message])
            
            try:
                # Try to parse JSON response
                analysis = json.loads(response.content)
                return analysis
            except json.JSONDecodeError:
                # Fallback to rule-based analysis
                return self._rule_based_analysis(query)
                
        except Exception as e:
            logger.error("Query analysis failed", error=str(e))
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
            "constraints": []
        }
