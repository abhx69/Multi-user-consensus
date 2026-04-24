"""
Agent module - Core intelligence layer of Gaprio.

This module contains:
- agent.py: Main Agent class orchestrating tools, memory, and LLM
- llm_provider.py: Abstraction layer for LLM providers (Ollama/OpenAI)
- prompts.py: System prompts and templates for the agent
"""

from gaprio.agent.agent import Agent
from gaprio.agent.llm_provider import LLMProvider, OllamaProvider, OpenAIProvider

__all__ = ["Agent", "LLMProvider", "OllamaProvider", "OpenAIProvider"]
