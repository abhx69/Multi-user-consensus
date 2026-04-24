"""
Unified Memory Manager for Gaprio.

This is the main interface for all memory operations, coordinating:
- Short-term: ConversationMemory for current chat context
- Working memory: Session-specific notes (not persisted automatically)
- Long-term: FileMemory for persistent storage
- Task state: HeartbeatManager for scheduled items

The MemoryManager provides the "mandatory recall step" - before the agent
answers questions about past work, preferences, or todos, it searches
the appropriate memory sources.
"""

import logging
from datetime import datetime
from typing import Any

from gaprio.memory.conversation import ConversationMemory
from gaprio.memory.file_memory import FileMemory
from gaprio.memory.heartbeat import HeartbeatManager
from gaprio.memory.preference_extractor import PreferenceExtractor

logger = logging.getLogger(__name__)


class MemoryManager:
    """
    Unified memory access layer for the agent.
    
    Coordinates all memory systems and provides a single interface for:
    - Adding memories (deciding where they should go)
    - Recalling relevant information (searching across sources)
    - Managing conversation context
    - Handling scheduled tasks and reminders
    
    Memory Hierarchy:
    1. Working memory (session notes, not persisted)
    2. Short-term (conversation history)
    3. Long-term (MEMORY.md, daily logs)
    4. Profile (USER.md, SOUL.md, TOOLS.md)
    
    Usage:
        memory = MemoryManager()
        
        # Start a conversation
        conv = memory.get_or_create_conversation("session_123")
        conv.add_user_message("Hello!")
        
        # Recall information for a query
        context = await memory.recall("What did we discuss yesterday?")
        
        # Add important info to long-term memory
        memory.remember("User prefers bullet points", importance="high")
    """
    
    def __init__(self):
        """Initialize all memory subsystems."""
        self.file_memory = FileMemory()
        self.heartbeat = HeartbeatManager()
        self._conversations: dict[str, ConversationMemory] = {}
        self._working_memory: dict[str, Any] = {}
        
        logger.info("MemoryManager initialized")
    
    # =========================================================================
    # Conversation Management (Short-term)
    # =========================================================================
    
    def get_or_create_conversation(self, session_id: str) -> ConversationMemory:
        """
        Get or create a conversation memory for a session.
        
        Args:
            session_id: Unique session identifier (e.g., channel_id:user_id)
            
        Returns:
            ConversationMemory instance for the session
        """
        if session_id not in self._conversations:
            self._conversations[session_id] = ConversationMemory(session_id=session_id)
            logger.debug(f"Created new conversation: {session_id}")
        
        return self._conversations[session_id]
    
    def clear_conversation(self, session_id: str) -> None:
        """Clear a conversation's history."""
        if session_id in self._conversations:
            self._conversations[session_id].clear()
    
    def get_conversation_context(self, session_id: str) -> str:
        """Get formatted conversation history for a session."""
        if session_id in self._conversations:
            return self._conversations[session_id].get_context()
        return ""
    
    # =========================================================================
    # Working Memory (Session-specific, not persisted)
    # =========================================================================
    
    def set_working_note(self, key: str, value: Any) -> None:
        """
        Store a note in working memory.
        
        Working memory is session-specific and not persisted unless
        explicitly written to files.
        
        Args:
            key: Identifier for the note
            value: Content to store
        """
        self._working_memory[key] = {
            "value": value,
            "timestamp": datetime.now().isoformat(),
        }
    
    def get_working_note(self, key: str) -> Any:
        """
        Retrieve a note from working memory.
        
        Args:
            key: Identifier for the note
            
        Returns:
            Stored value, or None if not found
        """
        note = self._working_memory.get(key)
        return note["value"] if note else None
    
    def clear_working_memory(self) -> None:
        """Clear all working memory notes."""
        self._working_memory.clear()
    
    def persist_working_note(self, key: str, to_memory: bool = True) -> None:
        """
        Persist a working note to long-term storage.
        
        Args:
            key: Note to persist
            to_memory: If True, save to MEMORY.md; if False, to daily log
        """
        note = self._working_memory.get(key)
        if not note:
            return
        
        content = f"{key}: {note['value']}"
        
        if to_memory:
            self.file_memory.add_to_memory(content, category="From Working Memory")
        else:
            self.file_memory.log_event(content, event_type="note")
    
    # =========================================================================
    # Recall (Mandatory Recall Step)
    # =========================================================================
    
    async def recall(
        self,
        query: str,
        user_id: str | None = None,
        include_conversation: bool = True,
        include_memory: bool = True,
        include_logs: bool = True,
        include_profile: bool = True,
    ) -> str:
        """
        Recall relevant information for a query.
        
        This is the mandatory recall step that searches across memory
        sources to find relevant context before the agent responds.
        
        Args:
            query: The search query (usually the user's message)
            user_id: Optional user ID for context
            include_conversation: Search current conversation
            include_memory: Search MEMORY.md
            include_logs: Search daily logs
            include_profile: Include profile context
            
        Returns:
            Formatted context string with relevant information
        """
        sections = []
        
        # Search curated memory (MEMORY.md)
        if include_memory:
            memory_matches = self.file_memory.search_memory(query)
            if memory_matches:
                sections.append(
                    "**From Long-term Memory:**\n" +
                    "\n".join(f"- {m}" for m in memory_matches[:5])
                )
        
        # Search recent logs
        if include_logs:
            log_matches = self.file_memory.search_logs(query, days=7)
            if log_matches:
                sections.append(
                    "**From Recent Activity:**\n" +
                    "\n".join(
                        f"- [{m['date']}] {m['line']}"
                        for m in log_matches[:5]
                    )
                )
        
        # Include profile context
        if include_profile:
            # User preferences
            user_profile = self.file_memory.read_profile("user")
            if user_profile:
                # Extract the profile (up to 1200 chars to avoid context overflow but catch all prefs)
                sections.append(
                    "**User Preferences:**\n" +
                    user_profile[:1200]
                )
        
        # Check for pending reminders
        pending = self.heartbeat.get_pending_reminders()
        if pending:
            sections.append(
                "**Pending Reminders:**\n" +
                "\n".join(
                    f"- {r['message']} (at {r['trigger_at']})"
                    for r in pending[:3]
                )
            )
        
        if not sections:
            return "No relevant memory found."
        
        return "\n\n".join(sections)
    
    # =========================================================================
    # Remember (Adding to memory)
    # =========================================================================
    
    def remember(
        self,
        content: str,
        importance: str = "normal",
        category: str | None = None,
    ) -> None:
        """
        Add information to long-term memory.
        
        Automatically decides where to store based on importance:
        - high: MEMORY.md (curated, persistent)
        - normal: Daily log
        - low: Working memory only
        
        Args:
            content: Information to remember
            importance: "high", "normal", or "low"
            category: Optional category for organization
        """
        if importance == "high":
            self.file_memory.add_to_memory(content, category=category)
        elif importance == "normal":
            self.file_memory.log_event(content, event_type="note")
        else:  # low
            self.set_working_note(f"note_{datetime.now().timestamp()}", content)
    
    def log_interaction(
        self,
        user_id: str,
        channel_id: str,
        message: str,
        response: str,
    ) -> None:
        """
        Log a conversation interaction to the daily log.
        
        Args:
            user_id: Slack user ID
            channel_id: Slack channel ID
            message: User's message
            response: Agent's response
        """
        self.file_memory.log_interaction(
            user_message=message,
            assistant_response=response,
            user_id=user_id,
            channel_id=channel_id,
        )
    
    def extract_and_save_preferences(
        self,
        message: str,
        user_id: str | None = None,
    ) -> list[dict]:
        """
        Extract preferences from a user message and save to USER.md.
        
        Uses PreferenceExtractor to detect preference statements like:
        - "My name is Hanu"
        - "I prefer short summaries"
        - "Always ask before posting"
        
        Args:
            message: The user's message to analyze
            user_id: Optional user ID for logging
            
        Returns:
            List of extracted preferences (as dicts with category and value)
        """
        extractor = PreferenceExtractor()
        
        # Quick check first
        if not extractor.has_preference_indicators(message):
            return []
        
        # Extract preferences
        preferences = extractor.extract_preferences(message)
        saved_prefs = []
        
        for pref in preferences:
            try:
                # Save to USER.md
                updated = self.file_memory.update_user_preference(
                    category=pref.category,
                    value=pref.value,
                )
                
                if updated:
                    # Log the preference learning event
                    self.file_memory.log_event(
                        f"Learned {pref.category}: {pref.value}",
                        event_type="preference",
                    )
                    
                    saved_prefs.append({
                        "category": pref.category,
                        "value": pref.value,
                    })
                    
                    logger.info(f"Saved preference: {pref.category} = {pref.value}")
                    
            except Exception as e:
                logger.warning(f"Failed to save preference {pref.category}: {e}")
        
        return saved_prefs
    
    # =========================================================================
    # Reminders and Scheduled Tasks
    # =========================================================================
    
    def schedule_reminder(
        self,
        message: str,
        trigger_at: datetime,
        channel: str | None = None,
        user_id: str | None = None,
    ) -> str:
        """
        Schedule a reminder.
        
        Args:
            message: Reminder text
            trigger_at: When to send the reminder
            channel: Target channel (defaults to DM)
            user_id: User who created the reminder
            
        Returns:
            Reminder ID
        """
        reminder_id = f"reminder_{datetime.now().timestamp()}"
        
        self.heartbeat.add_reminder(
            reminder_id=reminder_id,
            message=message,
            trigger_at=trigger_at,
            channel=channel,
            user_id=user_id,
        )
        
        return reminder_id
    
    def schedule_recurring_message(
        self,
        channel: str,
        message: str,
        time: str,
        name: str | None = None,
    ) -> str:
        """
        Schedule a recurring message.
        
        Args:
            channel: Target channel
            message: Message to post
            time: Time to post (e.g., "10:00")
            name: Optional task name
            
        Returns:
            Task ID
        """
        task_id = f"recurring_{datetime.now().timestamp()}"
        
        self.heartbeat.add_recurring_task(
            task_id=task_id,
            name=name or f"Message to {channel}",
            schedule=time,
            action="post_message",
            action_params={"message": message},
            channel=channel,
        )
        
        return task_id
    
    def get_due_items(self) -> dict:
        """Get all due reminders and tasks."""
        return self.heartbeat.get_due_items()
    
    # =========================================================================
    # Full Context
    # =========================================================================
    
    def get_full_context(self, session_id: str | None = None) -> str:
        """
        Get combined context from all memory sources.
        
        Args:
            session_id: Optional session ID to include conversation
            
        Returns:
            Formatted context string
        """
        sections = []
        
        # Conversation context
        if session_id and session_id in self._conversations:
            conv_context = self._conversations[session_id].get_context()
            if conv_context:
                sections.append(f"## Conversation\n{conv_context}")
        
        # File-based context
        file_context = self.file_memory.get_full_context()
        if file_context:
            sections.append(file_context)
        
        # Working memory
        if self._working_memory:
            notes = "\n".join(
                f"- {k}: {v['value']}"
                for k, v in self._working_memory.items()
            )
            sections.append(f"## Session Notes\n{notes}")
        
        return "\n\n".join(sections)
