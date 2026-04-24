"""
Memory module - Multi-layered memory system for Gaprio.

Memory Layers:
- Short-term: Conversation context (current chat window)
- Working memory: Ad-hoc notes within session (not persisted unless explicitly saved)
- Long-term: MEMORY.md (curated), memory/YYYY-MM-DD.md (daily logs)
- Profile files: USER.md, SOUL.md, TOOLS.md
- Task state: heartbeat-state.json for scheduled tasks and reminders
"""

from gaprio.memory.memory_manager import MemoryManager
from gaprio.memory.conversation import ConversationMemory
from gaprio.memory.file_memory import FileMemory
from gaprio.memory.heartbeat import HeartbeatManager
from gaprio.memory.preference_extractor import PreferenceExtractor

__all__ = [
    "MemoryManager",
    "ConversationMemory", 
    "FileMemory",
    "HeartbeatManager",
    "PreferenceExtractor",
]

