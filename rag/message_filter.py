"""
Smart Message Filtering Middleware for RAG.

Provides intelligent filtering and preprocessing of Slack messages:
- Noise removal (system messages, reactions, empty messages)
- Date/time filtering
- Thread chunking and grouping
- De-duplication
- Permission checking
- Structured output formatting for LLM

IMPORTANT: Gaprio's own messages (decisions, suggestions) are PRESERVED
as they represent valuable team outcomes.
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from gaprio.config import settings

logger = logging.getLogger(__name__)


@dataclass
class FilteredResult:
    """Result of message filtering with structured output."""
    
    messages: list[dict] = field(default_factory=list)
    threads: list[dict] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    
    @property
    def count(self) -> int:
        """Total message count."""
        return len(self.messages)
    
    @property
    def thread_count(self) -> int:
        """Number of threads."""
        return len(self.threads)
    
    def to_structured_text(self, channel_name: str = "channel") -> str:
        """
        Format messages as structured text for LLM consumption.
        
        Returns:
            Formatted string optimized for LLM understanding
        """
        lines = []
        
        # Header
        lines.append(f"## Messages from #{channel_name}")
        lines.append(f"Total: {self.count} messages")
        if self.metadata.get("filtered_count"):
            lines.append(f"(Filtered {self.metadata['filtered_count']} noise messages)")
        lines.append("")
        
        # Group by threads
        if self.threads:
            for thread in self.threads:
                thread_msgs = thread.get("messages", [])
                if not thread_msgs:
                    continue
                    
                # Thread header
                first_msg = thread_msgs[0]
                lines.append(f"### Thread: {first_msg.get('text', '')[:50]}...")
                
                for msg in thread_msgs[:10]:  # Limit per thread
                    user = msg.get("user", "unknown")
                    text = msg.get("text", "")
                    lines.append(f"- @{user}: {text}")
                
                if len(thread_msgs) > 10:
                    lines.append(f"  [+{len(thread_msgs) - 10} more messages]")
                lines.append("")
        else:
            # No threads, just list messages
            for msg in self.messages[:50]:  # Limit total
                user = msg.get("user", "unknown")
                text = msg.get("text", "")
                ts = msg.get("ts", "")
                lines.append(f"- @{user}: {text}")
            
            if len(self.messages) > 50:
                lines.append(f"[+{len(self.messages) - 50} more messages]")
        
        return "\n".join(lines)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "messages": self.messages,
            "threads": self.threads,
            "metadata": self.metadata,
            "count": self.count,
            "thread_count": self.thread_count,
        }


@dataclass
class FilterOptions:
    """Options for message filtering."""
    
    remove_bots: bool = True
    keep_gaprio: bool = True  # IMPORTANT: Keep Gaprio's decisions!
    remove_system: bool = True
    remove_reactions_only: bool = True
    remove_empty: bool = True
    deduplicate: bool = True
    chunk_threads: bool = True
    max_thread_messages: int = 20
    start_date: datetime | None = None
    end_date: datetime | None = None


class MessageFilter:
    """
    Smart message filtering middleware.
    
    Processes raw Slack messages through a pipeline of filters
    to produce clean, structured output for the LLM.
    
    Usage:
        filter = MessageFilter()
        result = filter.process(messages, options=FilterOptions())
        structured_text = result.to_structured_text()
    """
    
    # System message subtypes to filter
    SYSTEM_SUBTYPES = {
        "channel_join",
        "channel_leave", 
        "channel_topic",
        "channel_purpose",
        "channel_name",
        "channel_archive",
        "channel_unarchive",
        "group_join",
        "group_leave",
        "group_topic",
        "group_purpose",
        "group_name",
        "pinned_item",
        "unpinned_item",
        "file_share",  # Often noise, but can be kept if needed
        "me_message",
    }
    
    # Bot IDs/names to preserve (Gaprio's own messages)
    PRESERVED_BOTS = {
        "gaprio",
        "gaprio-agent",
        "gaprio_bot",
    }
    
    def __init__(self, gaprio_bot_id: str | None = None):
        """
        Initialize the filter.
        
        Args:
            gaprio_bot_id: Bot user ID for Gaprio (to preserve its messages)
        """
        self.gaprio_bot_id = gaprio_bot_id
    
    def process(
        self,
        messages: list[dict],
        options: FilterOptions | None = None,
    ) -> FilteredResult:
        """
        Process messages through the filtering pipeline.
        
        Pipeline order:
        1. Remove empty messages
        2. Remove system messages
        3. Remove bot messages (except Gaprio)
        4. Filter by date
        5. De-duplicate
        6. Chunk threads
        
        Args:
            messages: Raw Slack messages
            options: Filtering options
            
        Returns:
            FilteredResult with cleaned messages
        """
        if options is None:
            options = FilterOptions()
        
        original_count = len(messages)
        filtered = list(messages)
        
        # Step 1: Remove empty messages
        if options.remove_empty:
            filtered = self.filter_empty(filtered)
        
        # Step 2: Remove system messages
        if options.remove_system:
            filtered = self.filter_system(filtered)
        
        # Step 3: Remove bot messages (but keep Gaprio!)
        if options.remove_bots:
            filtered = self.filter_bots(filtered, keep_gaprio=options.keep_gaprio)
        
        # Step 4: Filter by date
        if options.start_date or options.end_date:
            filtered = self.filter_by_date(
                filtered, 
                start_date=options.start_date,
                end_date=options.end_date,
            )
        
        # Step 5: De-duplicate
        if options.deduplicate:
            filtered = self.deduplicate(filtered)
        
        # Step 6: Chunk threads
        threads = []
        if options.chunk_threads:
            threads = self.chunk_threads(filtered, options.max_thread_messages)
        
        # Build result
        result = FilteredResult(
            messages=filtered,
            threads=threads,
            metadata={
                "original_count": original_count,
                "filtered_count": original_count - len(filtered),
                "filters_applied": self._get_applied_filters(options),
            }
        )
        
        logger.info(
            f"Filtered {original_count} -> {len(filtered)} messages "
            f"({original_count - len(filtered)} removed)"
        )
        
        return result
    
    def filter_empty(self, messages: list[dict]) -> list[dict]:
        """Remove messages with empty or whitespace-only text."""
        return [
            msg for msg in messages
            if msg.get("text", "").strip()
        ]
    
    def filter_system(self, messages: list[dict]) -> list[dict]:
        """Remove system messages (joins, leaves, topic changes, etc.)."""
        return [
            msg for msg in messages
            if msg.get("subtype") not in self.SYSTEM_SUBTYPES
        ]
    
    def filter_bots(
        self, 
        messages: list[dict], 
        keep_gaprio: bool = True,
    ) -> list[dict]:
        """
        Remove bot messages except Gaprio.
        
        IMPORTANT: Gaprio's messages are preserved because they often
        contain decisions and outcomes that the team has accepted.
        """
        filtered = []
        
        for msg in messages:
            # Check if it's a bot message
            is_bot = msg.get("subtype") == "bot_message" or msg.get("bot_id")
            
            if not is_bot:
                # Human message - always keep
                filtered.append(msg)
                continue
            
            if not keep_gaprio:
                # Don't keep any bots
                continue
            
            # Check if it's Gaprio's message
            bot_id = msg.get("bot_id", "")
            username = msg.get("username", "").lower()
            
            is_gaprio = (
                bot_id == self.gaprio_bot_id or
                username in self.PRESERVED_BOTS or
                any(name in username for name in self.PRESERVED_BOTS)
            )
            
            if is_gaprio:
                # Preserve Gaprio's decisions!
                msg["_is_gaprio_decision"] = True  # Mark for special handling
                filtered.append(msg)
            # else: skip other bots
        
        return filtered
    
    def filter_by_date(
        self,
        messages: list[dict],
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> list[dict]:
        """Filter messages to a specific date range."""
        filtered = []
        
        for msg in messages:
            ts = msg.get("ts", "")
            if not ts:
                continue
            
            try:
                msg_time = datetime.fromtimestamp(float(ts))
            except (ValueError, TypeError):
                continue
            
            if start_date and msg_time < start_date:
                continue
            if end_date and msg_time > end_date:
                continue
            
            filtered.append(msg)
        
        return filtered
    
    def deduplicate(self, messages: list[dict]) -> list[dict]:
        """
        Remove duplicate messages.
        
        Handles:
        - Exact duplicates (same text and user)
        - Edited messages (keeps most recent)
        """
        seen = {}  # (user, text_hash) -> message
        
        for msg in messages:
            user = msg.get("user", "unknown")
            text = msg.get("text", "").strip()
            
            # Create a normalized key
            # Remove mentions and normalize whitespace
            normalized = re.sub(r"<@\w+>", "", text)
            normalized = " ".join(normalized.split()).lower()
            
            key = (user, hash(normalized))
            
            # Check for edited messages - keep most recent
            existing = seen.get(key)
            if existing:
                existing_ts = float(existing.get("ts", 0))
                current_ts = float(msg.get("ts", 0))
                if current_ts > existing_ts:
                    seen[key] = msg  # Replace with newer
            else:
                seen[key] = msg
        
        # Return in original order
        result = []
        seen_keys = set()
        for msg in messages:
            user = msg.get("user", "unknown")
            text = msg.get("text", "").strip()
            normalized = re.sub(r"<@\w+>", "", text)
            normalized = " ".join(normalized.split()).lower()
            key = (user, hash(normalized))
            
            if key not in seen_keys and key in seen:
                result.append(seen[key])
                seen_keys.add(key)
        
        return result
    
    def chunk_threads(
        self, 
        messages: list[dict],
        max_per_thread: int = 20,
    ) -> list[dict]:
        """
        Group messages by thread and chunk long threads.
        
        Returns:
            List of thread dicts with grouped messages
        """
        threads = {}  # thread_ts -> list of messages
        standalone = []  # Messages not in threads
        
        for msg in messages:
            thread_ts = msg.get("thread_ts")
            
            if thread_ts:
                if thread_ts not in threads:
                    threads[thread_ts] = []
                threads[thread_ts].append(msg)
            else:
                standalone.append(msg)
        
        # Build thread objects
        result = []
        
        # First, standalone messages as a "general" thread
        if standalone:
            result.append({
                "thread_ts": None,
                "is_standalone": True,
                "messages": standalone[:max_per_thread],
                "total_count": len(standalone),
                "truncated": len(standalone) > max_per_thread,
            })
        
        # Then actual threads
        for thread_ts, msgs in sorted(threads.items(), key=lambda x: x[0]):
            # Sort by timestamp
            msgs.sort(key=lambda m: float(m.get("ts", 0)))
            
            result.append({
                "thread_ts": thread_ts,
                "is_standalone": False,
                "messages": msgs[:max_per_thread],
                "total_count": len(msgs),
                "truncated": len(msgs) > max_per_thread,
            })
        
        return result
    
    def check_permissions(
        self,
        user_id: str,
        channel_id: str,
        slack_client: Any = None,
    ) -> bool:
        """
        Check if user has permission to access channel messages.
        
        For private channels, verifies user is a member.
        
        Args:
            user_id: Slack user ID
            channel_id: Slack channel ID
            slack_client: Slack client for API calls
            
        Returns:
            True if user has access
        """
        if not slack_client:
            # Can't verify without client, assume allowed
            return True
        
        try:
            # Get channel info
            response = slack_client.conversations_info(channel=channel_id)
            if not response.get("ok"):
                return False
            
            channel = response.get("channel", {})
            
            # Public channels - everyone with access to workspace
            if not channel.get("is_private"):
                return True
            
            # Private channel - check membership
            members_response = slack_client.conversations_members(channel=channel_id)
            if members_response.get("ok"):
                members = members_response.get("members", [])
                return user_id in members
            
            return False
            
        except Exception as e:
            logger.warning(f"Permission check failed: {e}")
            return True  # Fail open for now
    
    def _get_applied_filters(self, options: FilterOptions) -> list[str]:
        """Get list of filters that were applied."""
        filters = []
        if options.remove_empty:
            filters.append("empty")
        if options.remove_system:
            filters.append("system")
        if options.remove_bots:
            filters.append("bots")
        if options.deduplicate:
            filters.append("duplicates")
        if options.chunk_threads:
            filters.append("threads")
        if options.start_date or options.end_date:
            filters.append("date_range")
        return filters


# Convenience functions for common use cases

def filter_for_summary(
    messages: list[dict],
    hours: int = 24,
    gaprio_bot_id: str | None = None,
) -> FilteredResult:
    """
    Filter messages for summarization.
    
    Optimized for producing clean summaries:
    - Removes noise
    - Groups threads
    - Keeps decisions (including Gaprio's)
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(hours=hours)
    
    filter = MessageFilter(gaprio_bot_id=gaprio_bot_id)
    return filter.process(
        messages,
        options=FilterOptions(
            remove_bots=True,
            keep_gaprio=True,
            remove_system=True,
            deduplicate=True,
            chunk_threads=True,
            start_date=start_date,
            end_date=end_date,
        )
    )


def filter_for_indexing(messages: list[dict]) -> FilteredResult:
    """
    Filter messages for RAG indexing.
    
    Optimized for vector store:
    - Removes noise
    - Keeps more messages (less aggressive)
    - No thread chunking (index individually)
    """
    filter = MessageFilter()
    return filter.process(
        messages,
        options=FilterOptions(
            remove_bots=True,
            keep_gaprio=True,
            remove_system=True,
            deduplicate=True,
            chunk_threads=False,  # Index individually
        )
    )
