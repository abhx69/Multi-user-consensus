"""
LLM Provider abstraction layer.

This module provides a unified interface for different LLM backends:
- OllamaProvider: For local inference using Ollama (default)
- OpenAIProvider: For OpenAI API or compatible endpoints

The abstraction allows easy switching between providers without changing agent code.

Example usage:
    from gaprio.agent.llm_provider import get_llm_provider
    
    provider = get_llm_provider()  # Uses settings to determine provider
    response = await provider.generate("Hello, how are you?")
"""

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from typing import Any

import httpx

try:
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError
except ImportError:
    boto3 = None  # type: ignore
    BotoCoreError = None
    ClientError = None


from gaprio.config import settings

logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.
    
    All providers must implement the generate() method which takes a prompt
    and returns a string response. Additional parameters can be passed via kwargs.
    """
    
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> str:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: The user prompt/message to send
            system_prompt: Optional system prompt to set context
            temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative)
            max_tokens: Maximum tokens in the response
            **kwargs: Provider-specific parameters
            
        Returns:
            The generated text response
        """
        pass
    
    @abstractmethod
    async def generate_with_tools(
        self,
        prompt: str,
        tools: list[dict[str, Any]],
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Generate a response with tool/function calling capabilities.
        
        Args:
            prompt: The user prompt/message
            tools: List of tool definitions in OpenAI-compatible format
            system_prompt: Optional system prompt
            **kwargs: Provider-specific parameters
            
        Returns:
            Dictionary containing response and any tool calls
        """
        pass


class OllamaProvider(LLMProvider):
    """
    Ollama LLM provider for local inference.
    
    Uses the Ollama HTTP API to generate responses. Supports both basic
    generation and tool/function calling (if the model supports it).
    
    Configuration via settings:
        - ollama_base_url: API endpoint (default: http://localhost:11434)
        - ollama_model: Model name (default: llama3:instruct)
    """
    
    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
    ):
        """
        Initialize the Ollama provider.
        
        Args:
            base_url: Override the default Ollama API URL
            model: Override the default model name
        """
        self.base_url = base_url or settings.ollama_base_url
        self.model = model or settings.ollama_model
        self._client = httpx.AsyncClient(timeout=120.0)  # LLMs can be slow
        
    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> str:
        """Generate a response using Ollama's /api/generate endpoint."""
        
        # Build the full prompt with system context if provided
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\nUser: {prompt}\n\nAssistant:"
        
        payload = {
            "model": self.model,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        
        try:
            response = await self._client.post(
                f"{self.base_url}/api/generate",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            return data.get("response", "")
            
        except httpx.HTTPError as e:
            logger.error(f"Ollama API error: {e}")
            raise RuntimeError(f"Failed to generate response: {e}") from e
    
    async def generate_with_tools(
        self,
        prompt: str,
        tools: list[dict[str, Any]],
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Generate a response with tool calling using Ollama's chat API.
        
        Note: Tool calling support depends on the model. llama3:instruct
        supports basic function calling through structured prompts.
        """
        
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "tools": tools if tools else None,
        }
        
        try:
            response = await self._client.post(
                f"{self.base_url}/api/chat",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            
            message = data.get("message", {})
            
            return {
                "content": message.get("content", ""),
                "tool_calls": message.get("tool_calls", []),
                "raw_response": data,
            }
            
        except httpx.HTTPError as e:
            logger.error(f"Ollama API error: {e}")
            raise RuntimeError(f"Failed to generate response: {e}") from e
    
    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()


class OpenAIProvider(LLMProvider):
    """
    OpenAI-compatible LLM provider.
    
    Works with:
    - OpenAI API directly
    - Azure OpenAI
    - Any OpenAI-compatible API (vLLM, LocalAI, etc.)
    
    Configuration via settings:
        - openai_api_key: API key for authentication
        - openai_model: Model name (default: gpt-4)
    """
    
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str = "https://api.openai.com/v1",
    ):
        """
        Initialize the OpenAI provider.
        
        Args:
            api_key: Override the default API key
            model: Override the default model name
            base_url: API base URL (for compatible endpoints)
        """
        self.api_key = api_key or settings.openai_api_key
        self.model = model or settings.openai_model
        self.base_url = base_url
        self._client = httpx.AsyncClient(
            timeout=120.0,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
    
    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> str:
        """Generate a response using OpenAI's chat completions API."""
        
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        try:
            response = await self._client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            
            return data["choices"][0]["message"]["content"]
            
        except httpx.HTTPError as e:
            logger.error(f"OpenAI API error: {e}")
            raise RuntimeError(f"Failed to generate response: {e}") from e
    
    async def generate_with_tools(
        self,
        prompt: str,
        tools: list[dict[str, Any]],
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Generate a response with tool calling using OpenAI's function calling."""
        
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": self.model,
            "messages": messages,
            "tools": tools if tools else None,
        }
        
        try:
            response = await self._client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            
            message = data["choices"][0]["message"]
            
            # Parse tool calls if present
            tool_calls = []
            if "tool_calls" in message:
                for tc in message["tool_calls"]:
                    tool_calls.append({
                        "id": tc["id"],
                        "name": tc["function"]["name"],
                        "arguments": json.loads(tc["function"]["arguments"]),
                    })
            
            return {
                "content": message.get("content", ""),
                "tool_calls": tool_calls,
                "raw_response": data,
            }
            
        except httpx.HTTPError as e:
            logger.error(f"OpenAI API error: {e}")
            raise RuntimeError(f"Failed to generate response: {e}") from e
    
    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()


class OpenRouterProvider(OpenAIProvider):
    """
    OpenRouter LLM provider.
    
    Works with OpenRouter's OpenAI-compatible API.
    
    Configuration via settings:
        - openrouter_api_key: API key for authentication
        - openrouter_model: Model name
    """
    
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
    ):
        """Initialize the OpenRouter provider."""
        super().__init__(
            api_key=api_key or settings.openrouter_api_key,
            model=model or settings.openrouter_model,
            base_url=base_url or settings.openrouter_base_url,
        )
        
        # Optional headers for OpenRouter
        self._client.headers.update({
            "HTTP-Referer": "https://github.com/abhx69/Gaprio-core-agent",
            "X-Title": "Gaprio Agent",
        })


class BedrockProvider(LLMProvider):
    """
    Amazon Bedrock LLM provider.
    
    Works with Bedrock's converse API, supporting models like Claude 3.
    
    Configuration via settings:
        - bedrock_model_id: Model ID (e.g. anthropic.claude-3-haiku-20240307-v1:0)
        - aws_access_key_id: AWS access key
        - aws_secret_access_key: AWS secret key
        - aws_region: AWS region (e.g., us-east-1)
    """
    
    def __init__(
        self,
        model_id: str | None = None,
        region: str | None = None,
    ):
        """Initialize the Bedrock provider."""
        if boto3 is None:
            raise ImportError(
                "boto3 is required for BedrockProvider. Install with `pip install boto3`."
            )

        self.model_id = model_id or settings.bedrock_model_id
        
        # Determine region and credentials
        self.region = region or settings.aws_region
        kwargs = {"region_name": self.region}
        
        if settings.aws_access_key_id and settings.aws_secret_access_key:
            kwargs["aws_access_key_id"] = settings.aws_access_key_id
            kwargs["aws_secret_access_key"] = settings.aws_secret_access_key
            
        self._client = boto3.client("bedrock-runtime", **kwargs)

    async def _converse_async(self, **kwargs) -> dict:
        """Run the synchronous boto3 call in a thread pool."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self._client.converse(**kwargs))
    
    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> str:
        """Generate a response using Bedrock's converse API."""
        
        messages = [{"role": "user", "content": [{"text": prompt}]}]
        system_prompts = []
        if system_prompt:
            system_prompts.append({"text": system_prompt})
            
        inference_config = {
            "temperature": temperature,
            "maxTokens": max_tokens,
        }
        
        try:
            response = await self._converse_async(
                modelId=self.model_id,
                messages=messages,
                system=system_prompts,
                inferenceConfig=inference_config,
            )
            
            output_message = response.get("output", {}).get("message", {})
            content_blocks = output_message.get("content", [])
            
            # The converse api can return multiple content blocks
            response_text = ""
            for block in content_blocks:
                if "text" in block:
                    response_text += block["text"]
                    
            return response_text
            
        except (BotoCoreError, ClientError) as e:
            logger.error(f"Bedrock API error: {e}")
            raise RuntimeError(f"Failed to generate response using Bedrock: {e}") from e
    
    async def generate_with_tools(
        self,
        prompt: str,
        tools: list[dict[str, Any]],
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Generate a response with tool calling using Bedrock's converse API."""
        
        messages = [{"role": "user", "content": [{"text": prompt}]}]
        system_prompts = []
        if system_prompt:
            system_prompts.append({"text": system_prompt})
            
        # Convert OpenAI tool format to Bedrock tool format
        bedrock_tools = []
        for tool in tools:
            if "type" in tool and tool["type"] == "function":
                # OpenAI format: {"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}
                func = tool["function"]
                
                # Create a schema matching bedrock expectations
                b_tool_spec = {
                    "name": func["name"],
                }
                if "description" in func:
                    b_tool_spec["description"] = func["description"]
                if "parameters" in func:
                    b_tool_spec["inputSchema"] = {"json": func["parameters"]}
                else:
                    # Provide an empty schema if none exists
                    b_tool_spec["inputSchema"] = {"json": {"type": "object", "properties": {}}}
                    
                bedrock_tools.append({
                    "toolSpec": b_tool_spec
                })
        
        tool_config = {}
        if bedrock_tools:
            tool_config["tools"] = bedrock_tools
            
        args = {
            "modelId": self.model_id,
            "messages": messages,
            "system": system_prompts,
        }
        if tool_config:
            args["toolConfig"] = tool_config
            
        try:
            response = await self._converse_async(**args)
            output_message = response.get("output", {}).get("message", {})
            content_blocks = output_message.get("content", [])
            
            response_text = ""
            tool_calls = []
            
            for block in content_blocks:
                if "text" in block:
                    response_text += block["text"]
                elif "toolUse" in block:
                    tu = block["toolUse"]
                    # Convert back to common OpenAI-like format for the agent to use
                    tool_calls.append({
                        "id": tu["toolUseId"],
                        "name": tu["name"],
                        "arguments": tu["input"],
                    })
                    
            return {
                "content": response_text,
                "tool_calls": tool_calls,
                "raw_response": response,
            }
            
        except (BotoCoreError, ClientError) as e:
            logger.error(f"Bedrock API error: {e}")
            raise RuntimeError(f"Failed to generate response with tools using Bedrock: {e}") from e

    async def close(self) -> None:
        """Close method for Bedrock (Boto3 handles connection pooling automatically)."""
        pass


def get_llm_provider() -> LLMProvider:
    """
    Factory function to get the configured LLM provider.
    
    Returns the appropriate provider based on settings.llm_provider:
    - "ollama" -> OllamaProvider (default)
    - "openai" -> OpenAIProvider
    - "openrouter" -> OpenRouterProvider
    - "bedrock" -> BedrockProvider
    
    Returns:
        Configured LLM provider instance
    """
    if settings.llm_provider == "openai":
        return OpenAIProvider()
    elif settings.llm_provider == "openrouter":
        return OpenRouterProvider()
    elif settings.llm_provider == "bedrock":
        return BedrockProvider()
    else:
        return OllamaProvider()


# =============================================================================
# Test function - run with: python -m gaprio.agent.llm_provider
# =============================================================================
async def _test_provider():
    """Quick test of the configured LLM provider."""
    import asyncio
    
    provider = get_llm_provider()
    print(f"Testing {provider.__class__.__name__}...")
    
    try:
        response = await provider.generate(
            "Say 'Hello, I am working!' in exactly those words.",
            system_prompt="You are a helpful assistant.",
        )
        print(f"Response: {response}")
    finally:
        await provider.close()


if __name__ == "__main__":
    import asyncio
    asyncio.run(_test_provider())
