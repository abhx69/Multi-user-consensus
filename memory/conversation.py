"""
Conversation memory for short-term context.

Manages the immediate conversation history within the current chat session.
This is the "short-term memory" layer that provides context from recent
messages in the ongoing conversation.

Key features:
- Maintains a sliding window of conversation messages
- Respects token/message limits
- Provides formatted context for the agent
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

from gaprio.config import settings

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """
    A single message in the conversation.
    
    Attributes:
        role: Who sent the message (user, assistant, or system)
        content: The message text
        timestamp: When the message was sent
        metadata: Additional info (user_id, channel_id, etc.)
    """
    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert to dictionary format for LLM."""
        return {
            "role": self.role,
            "content": self.content,
        }


class ConversationMemory:
    """
    Short-term memory for the current conversation.
    
    Maintains a sliding window of recent messages, providing context
    for the agent. Older messages are dropped when the limit is reached.
    
    Usage:
        memory = ConversationMemory()
        memory.add_message("user", "Hello!")
        memory.add_message("assistant", "Hi there!")
        
        context = memory.get_context()
        # Returns formatted conversation history
    """
    
    def __init__(
        self,
        max_messages: int | None = None,
        session_id: str | None = None,
    ):
        """
        Initialize conversation memory.
        
        Args:
            max_messages: Maximum messages to keep (default from settings)
            session_id: Optional session identifier for tracking
        """
        self.max_messages = max_messages or settings.max_conversation_history
        self.session_id = session_id
        self.messages: list[Message] = []
        self.created_at = datetime.now()
        
        logger.debug(f"ConversationMemory created: {session_id or 'unnamed'}")
    
    def add_message(
        self,
        role: Literal["user", "assistant", "system"],
        content: str,
        **metadata,
    ) -> None:
        """
        Add a message to the conversation.
        
        Args:
            role: The speaker (user, assistant, or system)
            content: The message content
            **metadata: Additional metadata (user_id, channel_id, etc.)
        """
        message = Message(
            role=role,
            content=content,
            timestamp=datetime.now(),
            metadata=metadata,
        )
        self.messages.append(message)
        
        # Trim if we exceed the limit
        if len(self.messages) > self.max_messages:
            # Keep a system message if there is one
            if self.messages[0].role == "system":
                self.messages = [self.messages[0]] + self.messages[-(self.max_messages - 1):]
            else:
                self.messages = self.messages[-self.max_messages:]
        
        logger.debug(f"Added {role} message, total: {len(self.messages)}")
    
    def add_user_message(self, content: str, **metadata) -> None:
        """Convenience method to add a user message."""
        self.add_message("user", content, **metadata)
    
    def add_assistant_message(self, content: str, **metadata) -> None:
        """Convenience method to add an assistant message."""
        self.add_message("assistant", content, **metadata)
    
    def set_system_message(self, content: str) -> None:
        """
        Set or update the system message.
        
        The system message is always kept at the beginning of the conversation
        and persists even when older messages are trimmed.
        """
        # Remove existing system message if present
        self.messages = [m for m in self.messages if m.role != "system"]
        
        # Add new system message at the beginning
        system_msg = Message(role="system", content=content)
        self.messages.insert(0, system_msg)
    
    def get_messages(self) -> list[dict]:
        """
        Get messages in LLM-compatible format.
        
        Returns:
            List of dicts with role and content keys
        """
        return [m.to_dict() for m in self.messages]
    
    def get_context(self, include_system: bool = True) -> str:
        """
        Get formatted conversation context as a string.
        
        Args:
            include_system: Whether to include system message
            
        Returns:
            Formatted conversation history
        """
        lines = []
        for msg in self.messages:
            if not include_system and msg.role == "system":
                continue
            
            prefix = {
                "user": "User",
                "assistant": "Assistant",
                "system": "System",
            }[msg.role]
            
            lines.append(f"{prefix}: {msg.content}")
        
        return "\n\n".join(lines)
    
    def get_last_n_messages(self, n: int) -> list[Message]:
        """Get the last n messages."""
        return self.messages[-n:] if n > 0 else []
    
    def get_last_user_message(self) -> Message | None:
        """Get the most recent user message."""
        for msg in reversed(self.messages):
            if msg.role == "user":
                return msg
        return None
    
    def clear(self) -> None:
        """Clear all messages except system message."""
        self.messages = [m for m in self.messages if m.role == "system"]
        logger.debug("Conversation cleared")
    
    def reset(self) -> None:
        """Completely reset the conversation."""
        self.messages = []
        logger.debug("Conversation reset")
    
    @property
    def message_count(self) -> int:
        """Get the number of messages."""
        return len(self.messages)
    
    @property
    def user_message_count(self) -> int:
        """Get the number of user messages."""
        return sum(1 for m in self.messages if m.role == "user")
    
    def to_dict(self) -> dict:
        """Serialize conversation for storage."""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "messages": [
                {
                    "role": m.role,
                    "content": m.content,
                    "timestamp": m.timestamp.isoformat(),
                    "metadata": m.metadata,
                }
                for m in self.messages
            ],
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "ConversationMemory":
        """Deserialize conversation from storage."""
        memory = cls(session_id=data.get("session_id"))
        memory.created_at = datetime.fromisoformat(data["created_at"])
        
        for msg_data in data.get("messages", []):
            memory.messages.append(Message(
                role=msg_data["role"],
                content=msg_data["content"],
                timestamp=datetime.fromisoformat(msg_data["timestamp"]),
                metadata=msg_data.get("metadata", {}),
            ))
        
        return memory
