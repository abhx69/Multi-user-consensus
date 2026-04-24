"""
LLM Router — Task-specific model selection.

Routes each AI task to the optimal LLM provider based on
task type, cost, and capability requirements.
"""

import logging
from enum import Enum
from gaprio.agent.llm_provider import (
    LLMProvider, get_llm_provider,
    OllamaProvider, OpenRouterProvider, BedrockProvider
)
from gaprio.config import settings

logger = logging.getLogger(__name__)


class TaskType(str, Enum):
    WORK_DETECTION = "work_detection"
    CONSENSUS = "consensus"
    ORCHESTRATION = "orchestration"
    MEMORY = "memory"
    TOOL_SELECTION = "tool_selection"
    GENERAL = "general"


# Model routing table — maps task → (provider_class, model_override)
ROUTING_TABLE: dict[TaskType, dict] = {
    TaskType.WORK_DETECTION: {
        "openrouter": {"model": "anthropic/claude-3.5-sonnet"},
        "bedrock":    {"model": "anthropic.claude-3-5-sonnet-20241022-v2:0"},
        "ollama":     {"model": "llama3:instruct"},
    },
    TaskType.CONSENSUS: {
        "openrouter": {"model": "anthropic/claude-3.5-sonnet"},
        "bedrock":    {"model": "anthropic.claude-3-5-sonnet-20241022-v2:0"},
        "ollama":     {"model": "llama3:instruct"},
    },
    TaskType.ORCHESTRATION: {
        "openrouter": {"model": "openai/gpt-4o"},
        "bedrock":    {"model": "anthropic.claude-3-5-sonnet-20241022-v2:0"},
        "ollama":     {"model": "llama3:instruct"},
    },
    TaskType.MEMORY: {
        "openrouter": {"model": "meta-llama/llama-3-8b-instruct"},
        "bedrock":    {"model": "meta.llama3-8b-instruct-v1:0"},
        "ollama":     {"model": "llama3:instruct"},
    },
    TaskType.TOOL_SELECTION: {
        "openrouter": {"model": "openai/gpt-4o"},
        "bedrock":    {"model": "anthropic.claude-3-5-sonnet-20241022-v2:0"},
        "ollama":     {"model": "llama3:instruct"},
    },
    TaskType.GENERAL: {
        "openrouter": {"model": "anthropic/claude-3.5-sonnet"},
        "bedrock":    {"model": "anthropic.claude-3-5-sonnet-20241022-v2:0"},
        "ollama":     {"model": "llama3:instruct"},
    },
}


class LLMRouter:
    """Routes LLM calls to the optimal provider per task type."""

    def __init__(self):
        self._provider_name = getattr(settings, "llm_provider", "ollama").lower()
        self._providers: dict[str, LLMProvider] = {}
        logger.info(f"LLMRouter initialized with base provider: {self._provider_name}")

    def _get_or_create_provider(self, model_override: str | None = None) -> LLMProvider:
        """Get or create a provider instance, optionally with model override."""
        cache_key = f"{self._provider_name}:{model_override or 'default'}"
        if cache_key not in self._providers:
            provider = get_llm_provider()
            if model_override and hasattr(provider, 'model'):
                provider.model = model_override
            self._providers[cache_key] = provider
        return self._providers[cache_key]

    def get_provider(self, task: TaskType) -> LLMProvider:
        """Get the optimal LLM provider for a specific task type."""
        route = ROUTING_TABLE.get(task, ROUTING_TABLE[TaskType.GENERAL])
        config = route.get(self._provider_name, route.get("ollama", {}))
        model = config.get("model")
        logger.info(f"Routing {task.value} → {self._provider_name}/{model}")
        return self._get_or_create_provider(model)

    async def generate(self, task: TaskType, prompt: str, **kwargs) -> str:
        """Generate a response using the optimal provider for the task."""
        provider = self.get_provider(task)
        return await provider.generate(prompt=prompt, **kwargs)