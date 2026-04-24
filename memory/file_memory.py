"""
File-backed long-term memory for Gaprio.

This module manages persistent memory stored in markdown files:
- MEMORY.md: Curated, important information (decisions, preferences, facts)
- memory/YYYY-MM-DD.md: Daily logs of events and interactions
- USER.md: User preferences and personal information
- SOUL.md: Agent behavior configuration
- TOOLS.md: Environment-specific notes

The file-based approach provides:
- Human-readable storage
- Easy manual editing and review
- Git-friendly version control
- No database dependencies
"""

import logging
import re
from datetime import datetime, date
from pathlib import Path
from typing import Literal

from gaprio.config import settings

logger = logging.getLogger(__name__)


class FileMemory:
    """
    File-backed long-term memory manager.
    
    Handles reading and writing to the various memory files:
    - Curated memory (MEMORY.md)
    - Daily logs (memory/YYYY-MM-DD.md)
    - Profile files (USER.md, SOUL.md, TOOLS.md)
    
    Usage:
        file_memory = FileMemory()
        
        # Write to curated memory
        file_memory.add_to_memory("User prefers concise responses")
        
        # Log an event
        file_memory.log_event("Summarized #general channel")
        
        # Read memory
        content = file_memory.read_memory()
    """
    
    def __init__(self, data_dir: Path | None = None):
        """
        Initialize file memory with data directory.
        
        Args:
            data_dir: Directory for memory files (default from settings)
        """
        self.data_dir = Path(data_dir or settings.data_dir)
        self._ensure_directories()
        self._ensure_files()
        
        logger.info(f"FileMemory initialized at {self.data_dir}")
    
    def _ensure_directories(self) -> None:
        """Create necessary directories."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "memory").mkdir(exist_ok=True)
    
    def _ensure_files(self) -> None:
        """Create memory files if they don't exist."""
        # Create MEMORY.md with header
        memory_file = self.data_dir / "MEMORY.md"
        if not memory_file.exists():
            memory_file.write_text(
                "# Long-Term Memory\n\n"
                "This file contains curated, important information.\n"
                "Edit manually to add or remove entries.\n\n"
                "---\n\n"
            )
        
        # Create USER.md
        user_file = self.data_dir / "USER.md"
        if not user_file.exists():
            user_file.write_text(
                "# User Profile\n\n"
                "User preferences and personal information.\n\n"
                "## Preferences\n\n"
                "- Response style: Concise and helpful\n"
                "- Tone: Professional but friendly\n\n"
                "## Information\n\n"
                "- Name: (not set)\n"
                "- Timezone: (not set)\n"
            )
        
        # Create SOUL.md
        soul_file = self.data_dir / "SOUL.md"
        if not soul_file.exists():
            soul_file.write_text(
                "# Agent Behavior\n\n"
                "Configuration for how the agent should behave.\n\n"
                "## Personality\n\n"
                "- Be helpful and proactive\n"
                "- Acknowledge mistakes\n"
                "- Ask for clarification when unsure\n\n"
                "## Boundaries\n\n"
                "- Don't share private channel content without permission\n"
                "- Don't take destructive actions without confirmation\n"
                "- Keep sensitive information local\n"
            )
        
        # Create TOOLS.md
        tools_file = self.data_dir / "TOOLS.md"
        if not tools_file.exists():
            tools_file.write_text(
                "# Environment Notes\n\n"
                "Tool-specific configuration and environment details.\n\n"
                "## Accounts\n\n"
                "- Slack: Connected\n"
                "- GitHub: (configure in .env)\n"
                "- Notion: (configure in .env)\n\n"
                "## Notes\n\n"
                "Add environment-specific notes here.\n"
            )
    
    # =========================================================================
    # Curated Memory (MEMORY.md)
    # =========================================================================
    
    def read_memory(self) -> str:
        """
        Read the curated memory file.
        
        Returns:
            Contents of MEMORY.md
        """
        memory_file = self.data_dir / "MEMORY.md"
        return memory_file.read_text(encoding="utf-8") if memory_file.exists() else ""
    
    def add_to_memory(
        self,
        content: str,
        category: str | None = None,
    ) -> None:
        """
        Add an entry to the curated memory.
        
        Args:
            content: The information to remember
            category: Optional category header
        """
        memory_file = self.data_dir / "MEMORY.md"
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        
        entry = f"\n## {category}\n" if category else ""
        entry += f"- [{timestamp}] {content}\n"
        
        with open(memory_file, "a", encoding="utf-8") as f:
            f.write(entry)
        
        logger.info(f"Added to memory: {content[:50]}...")
    
    def search_memory(self, query: str) -> list[str]:
        """
        Search the curated memory for relevant entries.
        
        Args:
            query: Search term
            
        Returns:
            List of matching lines
        """
        memory_content = self.read_memory()
        query_lower = query.lower()
        
        matches = []
        for line in memory_content.split("\n"):
            if query_lower in line.lower():
                matches.append(line.strip())
        
        return matches
    
    # =========================================================================
    # Daily Logs (memory/YYYY-MM-DD.md)
    # =========================================================================
    
    def _get_daily_log_path(self, log_date: date | None = None) -> Path:
        """Get the path for a daily log file."""
        log_date = log_date or date.today()
        return self.data_dir / "memory" / f"{log_date.isoformat()}.md"
    
    def read_daily_log(self, log_date: date | None = None) -> str:
        """
        Read a daily log file.
        
        Args:
            log_date: Date of the log (default: today)
            
        Returns:
            Contents of the daily log
        """
        log_path = self._get_daily_log_path(log_date)
        return log_path.read_text(encoding="utf-8") if log_path.exists() else ""
    
    def log_event(
        self,
        event: str,
        event_type: str = "event",
        log_date: date | None = None,
    ) -> None:
        """
        Log an event to the daily log.
        
        Args:
            event: Description of the event
            event_type: Type of event (event, interaction, note, etc.)
            log_date: Date for the log (default: today)
        """
        log_path = self._get_daily_log_path(log_date)
        
        # Create file with header if it doesn't exist
        if not log_path.exists():
            log_path.write_text(
                f"# Daily Log - {(log_date or date.today()).isoformat()}\n\n"
            )
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        entry = f"- [{timestamp}] **{event_type}**: {event}\n"
        
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(entry)
        
        logger.debug(f"Logged event: {event_type} - {event[:50]}...")
    
    def log_interaction(
        self,
        user_message: str,
        assistant_response: str,
        user_id: str | None = None,
        channel_id: str | None = None,
    ) -> None:
        """
        Log a conversation interaction.
        
        Args:
            user_message: What the user said
            assistant_response: What the assistant responded
            user_id: Slack user ID
            channel_id: Slack channel ID
        """
        context = []
        if user_id:
            context.append(f"user={user_id}")
        if channel_id:
            context.append(f"channel={channel_id}")
        
        context_str = f" ({', '.join(context)})" if context else ""
        
        # Log a summary rather than full messages to save space
        user_preview = user_message[:100] + "..." if len(user_message) > 100 else user_message
        response_preview = assistant_response[:100] + "..." if len(assistant_response) > 100 else assistant_response
        
        self.log_event(
            f"User{context_str}: {user_preview} -> Response: {response_preview}",
            event_type="interaction",
        )
    
    def get_recent_logs(self, days: int = 7) -> dict[str, str]:
        """
        Get logs from the last N days.
        
        Args:
            days: Number of days to look back
            
        Returns:
            Dict mapping date strings to log contents
        """
        logs = {}
        today = date.today()
        
        for i in range(days):
            log_date = date.fromordinal(today.toordinal() - i)
            log_path = self._get_daily_log_path(log_date)
            if log_path.exists():
                logs[log_date.isoformat()] = log_path.read_text(encoding="utf-8")
        
        return logs
    
    def search_logs(
        self,
        query: str,
        days: int = 7,
    ) -> list[dict]:
        """
        Search recent logs for a query.
        
        Args:
            query: Search term
            days: Number of days to search
            
        Returns:
            List of matches with date and line
        """
        logs = self.get_recent_logs(days)
        query_lower = query.lower()
        
        matches = []
        for log_date, content in logs.items():
            for line in content.split("\n"):
                if query_lower in line.lower():
                    matches.append({
                        "date": log_date,
                        "line": line.strip(),
                    })
        
        return matches
    
    # =========================================================================
    # Profile Files (USER.md, SOUL.md, TOOLS.md)
    # =========================================================================
    
    def read_profile(
        self,
        profile_type: Literal["user", "soul", "tools"],
    ) -> str:
        """
        Read a profile file.
        
        Args:
            profile_type: Which profile to read
            
        Returns:
            Contents of the profile file
        """
        filename = {
            "user": "USER.md",
            "soul": "SOUL.md",
            "tools": "TOOLS.md",
        }[profile_type]
        
        profile_path = self.data_dir / filename
        return profile_path.read_text(encoding="utf-8") if profile_path.exists() else ""
    
    def update_profile(
        self,
        profile_type: Literal["user", "soul", "tools"],
        section: str,
        content: str,
    ) -> None:
        """
        Update a section in a profile file.
        
        This is a simple append operation. For more complex updates,
        manually edit the file.
        
        Args:
            profile_type: Which profile to update
            section: Section name (used as header)
            content: Content to add
        """
        filename = {
            "user": "USER.md",
            "soul": "SOUL.md",
            "tools": "TOOLS.md",
        }[profile_type]
        
        profile_path = self.data_dir / filename
        
        timestamp = datetime.now().strftime("%Y-%m-%d")
        entry = f"\n## {section} (updated {timestamp})\n\n{content}\n"
        
        with open(profile_path, "a", encoding="utf-8") as f:
            f.write(entry)
        
        logger.info(f"Updated {filename}: {section}")
    
    def update_user_preference(
        self,
        category: str,
        value: str,
    ) -> bool:
        """
        Update a specific user preference in USER.md.
        
        Intelligently updates existing preferences or adds new ones.
        Category mappings:
        - name -> "Name:" line
        - timezone -> "Timezone:" line  
        - response_style -> "Response style:" line
        - primary_channels -> "Primary channels:" line
        - Other categories -> Added to Preferences section
        
        Args:
            category: Preference category (name, timezone, response_style, etc.)
            value: The preference value
            
        Returns:
            True if preference was updated
        """
        user_file = self.data_dir / "USER.md"
        if not user_file.exists():
            self._ensure_files()
        
        content = user_file.read_text(encoding="utf-8")
        updated = False
        
        # Map categories to line patterns
        line_patterns = {
            "name": (r"- Name:.*", f"- Name: {value}"),
            "timezone": (r"- Timezone:.*", f"- Timezone: {value}"),
            "response_style": (r"- Response style:.*", f"- Response style: {value}"),
            "primary_channels": (r"- Primary channels:.*", f"- Primary channels: {value}"),
        }
        
        if category in line_patterns:
            pattern, replacement = line_patterns[category]
            if re.search(pattern, content):
                content = re.sub(pattern, replacement, content)
                updated = True
            else:
                # Add to Information section if it exists
                if "## Information" in content:
                    content = content.replace(
                        "## Information\n",
                        f"## Information\n\n{replacement}\n"
                    )
                    updated = True
        else:
            # For other categories, add to Preferences section
            timestamp = datetime.now().strftime("%Y-%m-%d")
            new_entry = f"- {category.replace('_', ' ').title()}: {value} (learned {timestamp})"
            
            if "## Preferences" in content:
                # Find the end of Preferences section (next ## or end)
                pref_match = re.search(r"(## Preferences\n)(.*?)(?=\n## |\Z)", content, re.DOTALL)
                if pref_match:
                    section_content = pref_match.group(2)
                    # Check if this preference type already exists
                    category_pattern = rf"- {category.replace('_', ' ').title()}:.*"
                    if re.search(category_pattern, section_content, re.IGNORECASE):
                        content = re.sub(category_pattern, new_entry, content, flags=re.IGNORECASE)
                    else:
                        # Add new preference
                        content = content.replace(
                            pref_match.group(0),
                            pref_match.group(1) + section_content.rstrip() + f"\n{new_entry}\n"
                        )
                    updated = True
            else:
                # Add Preferences section if missing
                content += f"\n## Preferences\n\n{new_entry}\n"
                updated = True
        
        if updated:
            user_file.write_text(content, encoding="utf-8")
            logger.info(f"Updated user preference: {category} = {value}")
        
        return updated
    
    # =========================================================================
    # Combined Context
    # =========================================================================
    
    def get_full_context(self) -> str:
        """
        Get combined context from all memory sources.
        
        Returns a formatted string with:
        - User profile highlights
        - Recent memory entries
        - Recent log entries
        """
        sections = []
        
        # User profile summary
        user_profile = self.read_profile("user")
        if user_profile:
            sections.append(f"### User Profile\n{user_profile[:500]}")
        
        # Curated memory
        memory = self.read_memory()
        if memory:
            # Get last 1000 chars to avoid context overflow
            sections.append(f"### Long-Term Memory\n{memory[-1000:]}")
        
        # Recent logs (last 3 days, truncated)
        logs = self.get_recent_logs(days=3)
        if logs:
            log_preview = "\n---\n".join(
                f"**{date}**:\n{content[:300]}"
                for date, content in list(logs.items())[:2]
            )
            sections.append(f"### Recent Activity\n{log_preview}")
        
        return "\n\n".join(sections)
