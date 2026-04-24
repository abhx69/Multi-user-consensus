"""
Slack MCP Server.

Provides tools for interacting with Slack:
- Read messages from channels
- Post messages to channels
- Schedule messages
- Get channel/user info
- Manage reminders
"""

import logging
from datetime import datetime, timedelta
from typing import Any

from gaprio.config import settings
from gaprio.mcp.base_server import BaseMCPServer, MCPTool, ToolResult

logger = logging.getLogger(__name__)


class SlackMCPServer(BaseMCPServer):
    """
    MCP server for Slack operations.
    
    Provides tools for the agent to interact with Slack:
    - Reading channel messages
    - Posting messages
    - Scheduling messages
    - Managing channel information
    
    Usage:
        server = SlackMCPServer(slack_client)
        result = await server.execute("slack_read_messages", channel="C123", hours=24)
    """
    
    def __init__(self, slack_client: Any = None):
        """
        Initialize the Slack MCP server.
        
        Args:
            slack_client: Slack WebClient instance
        """
        self.slack_client = slack_client
        self._channel_cache: dict[str, str] = {}  # name -> id cache
        super().__init__("slack")
    
    def set_client(self, client: Any) -> None:
        """Set the Slack client (for late binding)."""
        self.slack_client = client
        # Pre-populate cache with known channels to avoid API lookup issues
        self._channel_cache: dict[str, str] = {
            "gaprio-sm": "C0ADH4KF0LS",
            "social": "C09FGFS1XMM", 
            "gaprio-project-collab": "C09FUSZH8BX",
            "all-gaprio": "C09FGFS171D",
        }
    
    async def _resolve_channel(self, channel: str) -> str:
        """
        Resolve a channel name, channel ID, or user ID to a valid channel ID.
        
        Args:
            channel: Channel name (with or without #), channel ID, or user ID
            
        Returns:
            Channel ID or user ID for DMs
        """
        if not channel:
            logger.warning("Empty channel provided")
            return ""
        
        # Handle Slack mention format: <#C0ADH4KF0LS> or <#C0ADH4KF0LS|channel-name>
        if channel.startswith("<#") and ">" in channel:
            # Extract the ID from <#ID> or <#ID|name>
            inner = channel[2:channel.index(">")]
            if "|" in inner:
                channel = inner.split("|")[0]  # Get just the ID
            else:
                channel = inner
            logger.info(f"Extracted channel ID from mention: {channel}")
            return channel
        
        # Remove # prefix if present
        if channel.startswith("#"):
            channel = channel[1:]
        
        # Remove @ prefix if present (for user mentions)
        if channel.startswith("@"):
            channel = channel[1:]
        
        # If it looks like a channel ID (starts with C or G), return as-is
        if channel.startswith(("C", "G")) and len(channel) >= 9:
            return channel
        
        # If it looks like a DM channel ID (starts with D), return as-is
        if channel.startswith("D") and len(channel) >= 9:
            return channel
        
        # If it looks like a user ID (starts with U or W), return as-is for DMs
        if channel.startswith(("U", "W")) and len(channel) >= 9:
            return channel
        
        # Check cache
        if channel in self._channel_cache:
            return self._channel_cache[channel]
        
        # Look up channel by name
        try:
            # First try public and private channels
            response = await self._call_api(
                "conversations.list",
                types="public_channel,private_channel",
                exclude_archived=True,
                limit=1000,
            )
            
            if response.get("ok"):
                for ch in response.get("channels", []):
                    ch_name = ch.get("name", "")
                    ch_id = ch.get("id", "")
                    self._channel_cache[ch_name] = ch_id
                    if ch_name.lower() == channel.lower():
                        return ch_id
            
            # Also try IMs (direct messages) - wrap in try/except in case of scope issues
            try:
                im_response = await self._call_api(
                    "conversations.list",
                    types="im",  # Only use 'im', not 'mpim' to avoid scope issues
                    limit=100,
                )
                
                if im_response.get("ok"):
                    for im in im_response.get("channels", []):
                        im_id = im.get("id", "")
                        user_id = im.get("user", "")
                        if user_id:
                            self._channel_cache[f"dm_{user_id}"] = im_id
            except Exception as im_error:
                logger.debug(f"Could not list IMs (may be OK): {im_error}")
            
            # If we still haven't found it, raise an error with available options
            available_channels = list(self._channel_cache.keys())
            error_msg = f"Could not resolve channel '{channel}'. Available channels: {', '.join(available_channels[:10])}"
            if len(available_channels) > 10:
                error_msg += "..."
            
            logger.warning(error_msg)
            raise ValueError(error_msg)
            
        except Exception as e:
            logger.error(f"Error resolving channel: {e}")
            raise  # Re-raise the exception to be caught by the tool handler
    
    def _register_tools(self) -> None:
        """Register all Slack tools."""
        
        # Read messages
        self.add_tool(MCPTool(
            name="slack_read_messages",
            description=(
                "Read messages from a Slack channel. "
                "Returns messages from the specified time period."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "channel": {
                        "type": "string",
                        "description": "Channel ID or name (e.g., C123456 or #general)",
                    },
                    "hours": {
                        "type": "integer",
                        "description": "How many hours back to read (default: 24)",
                        "default": 24,
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum messages to return (default: 100)",
                        "default": 100,
                    },
                },
                "required": ["channel"],
            },
            handler=self._read_messages,
        ))
        
        # Post message
        self.add_tool(MCPTool(
            name="slack_post_message",
            description=(
                "Post a message to a Slack channel. "
                "Supports basic formatting with Slack markdown."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "channel": {
                        "type": "string",
                        "description": "Channel ID or name to post to",
                    },
                    "text": {
                        "type": "string",
                        "description": "Message text to post",
                    },
                    "thread_ts": {
                        "type": "string",
                        "description": "Thread timestamp to reply in (optional)",
                    },
                },
                "required": ["channel", "text"],
            },
            handler=self._post_message,
        ))
        
        # Schedule message
        self.add_tool(MCPTool(
            name="slack_schedule_message",
            description=(
                "Schedule a message to be posted at a specific time. "
                "Useful for reminders and timed announcements."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "channel": {
                        "type": "string",
                        "description": "Channel ID to post to",
                    },
                    "text": {
                        "type": "string",
                        "description": "Message text",
                    },
                    "post_at": {
                        "type": "string",
                        "description": "When to post (ISO format or relative like '+5m', '+1h')",
                    },
                },
                "required": ["channel", "text", "post_at"],
            },
            handler=self._schedule_message,
        ))
        
        # Get channel info
        self.add_tool(MCPTool(
            name="slack_get_channel_info",
            description="Get information about a Slack channel.",
            parameters={
                "type": "object",
                "properties": {
                    "channel": {
                        "type": "string",
                        "description": "Channel ID",
                    },
                },
                "required": ["channel"],
            },
            handler=self._get_channel_info,
        ))
        
        # List channels
        self.add_tool(MCPTool(
            name="slack_list_channels",
            description="List channels the bot is a member of.",
            parameters={
                "type": "object",
                "properties": {
                    "types": {
                        "type": "string",
                        "description": "Channel types (public_channel, private_channel, mpim, im)",
                        "default": "public_channel,private_channel",
                    },
                },
            },
            handler=self._list_channels,
        ))
        self.add_tool(MCPTool(
            name="slack_get_thread_replies",
            description=(
                "Get all replies in a Slack thread. "
                "Critical for reading full group discussions and multi-user consensus."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "channel": {
                        "type": "string",
                        "description": "Channel ID",
                    },
                    "thread_ts": {
                        "type": "string",
                        "description": "Thread timestamp of the parent message",
                    },
                },
                "required": ["channel", "thread_ts"],
            },
            handler=self._get_thread_replies,
        ))
        
        self.add_tool(MCPTool(
            name="slack_search_messages",
            description="Search for messages across all Slack channels.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query",
                    },
                    "count": {
                        "type": "integer",
                        "description": "Maximum results to return (default: 10)",
                        "default": 10,
                    },
                },
                "required": ["query"],
            },
            handler=self._search_messages,
        ))
    
    def _ensure_client(self) -> None:
        """Ensure Slack client is configured."""
        if not self.slack_client:
            raise RuntimeError("Slack client not configured")
    
    async def _call_api(self, method: str, **kwargs) -> dict:
        """Call Slack API method."""
        self._ensure_client()
        
        client_method = getattr(self.slack_client, method.replace(".", "_"), None)
        
        if client_method is None:
            if hasattr(self.slack_client, "api_call"):
                return self.slack_client.api_call(method, **kwargs)
            raise ValueError(f"Cannot call Slack API: {method}")
        
        import asyncio
        if asyncio.iscoroutinefunction(client_method):
            return await client_method(**kwargs)
        else:
            return client_method(**kwargs)
    
    async def _read_messages(
        self,
        channel: str,
        hours: int = 24,
        limit: int = 100,
    ) -> ToolResult:
        """
        Read messages from a channel with smart filtering.
        
        Applies MessageFilter to:
        - Remove noise (bots, system messages)
        - Preserve Gaprio decisions
        - De-duplicate messages
        - Group threads
        
        Args:
            channel: Channel ID
            hours: Hours back to read
            limit: Max messages
            
        Returns:
            ToolResult with filtered, structured messages
        """
        try:
            from gaprio.rag.message_filter import MessageFilter, FilterOptions
            from datetime import datetime, timedelta
            
            # Resolve channel name to ID
            channel_id = await self._resolve_channel(channel)
            
            # Calculate oldest timestamp
            end_date = datetime.now()
            start_date = end_date - timedelta(hours=hours)
            oldest_ts = str(start_date.timestamp())
            
            response = await self._call_api(
                "conversations.history",
                channel=channel_id,
                oldest=oldest_ts,
                limit=limit,
            )
            
            if not response.get("ok"):
                return ToolResult(
                    success=False,
                    error=response.get("error", "Unknown error"),
                )
            
            messages = response.get("messages", [])
            
            # Apply smart filtering
            filter = MessageFilter()
            result = filter.process(
                messages,
                options=FilterOptions(
                    remove_bots=True,
                    keep_gaprio=True,  # Preserve Gaprio's decisions!
                    remove_system=True,
                    deduplicate=True,
                    chunk_threads=True,
                    start_date=start_date,
                    end_date=end_date,
                )
            )
            
            # Get channel name for structured output
            channel_name = channel
            try:
                ch_info = await self._call_api("conversations.info", channel=channel_id)
                if ch_info.get("ok"):
                    channel_name = ch_info.get("channel", {}).get("name", channel)
            except Exception:
                pass
            
            return ToolResult(
                success=True,
                data={
                    "channel": channel_id,
                    "channel_name": channel_name,
                    "messages": result.messages,
                    "threads": result.threads,
                    "count": result.count,
                    "thread_count": result.thread_count,
                    "hours": hours,
                    "metadata": result.metadata,
                    "structured_text": result.to_structured_text(channel_name),
                },
            )
            
        except Exception as e:
            logger.error(f"Error reading messages: {e}")
            return ToolResult(success=False, error=str(e))
    
    async def _post_message(
        self,
        channel: str,
        text: str,
        thread_ts: str | None = None,
    ) -> ToolResult:
        """
        Post a message to a channel.
        
        Args:
            channel: Channel ID
            text: Message text
            thread_ts: Optional thread to reply in
            
        Returns:
            ToolResult with post confirmation
        """
        try:
            # Resolve channel name to ID
            channel = await self._resolve_channel(channel)
            
            # Ensure text is provided
            if not text:
                return ToolResult(
                    success=False,
                    error="Text is required for posting a message",
                )
            
            kwargs = {
                "channel": channel,
                "text": text,
            }
            if thread_ts:
                kwargs["thread_ts"] = thread_ts
            
            response = await self._call_api("chat.postMessage", **kwargs)
            
            if not response.get("ok"):
                return ToolResult(
                    success=False,
                    error=response.get("error", "Unknown error"),
                )
            
            return ToolResult(
                success=True,
                data={
                    "channel": channel,
                    "ts": response.get("ts"),
                    "message": "Message posted successfully",
                },
            )
            
        except Exception as e:
            logger.error(f"Error posting message: {e}")
            return ToolResult(success=False, error=str(e))
    
    async def _schedule_message(
        self,
        channel: str,
        text: str,
        post_at: str,
    ) -> ToolResult:
        """
        Schedule a message for later.
        
        Args:
            channel: Channel ID
            text: Message text
            post_at: When to post (ISO or relative)
            
        Returns:
            ToolResult with schedule confirmation
        """
        try:
            # Parse post_at
            post_time = self._parse_time(post_at)
            post_timestamp = int(post_time.timestamp())
            
            response = await self._call_api(
                "chat.scheduleMessage",
                channel=channel,
                text=text,
                post_at=post_timestamp,
            )
            
            if not response.get("ok"):
                return ToolResult(
                    success=False,
                    error=response.get("error", "Unknown error"),
                )
            
            return ToolResult(
                success=True,
                data={
                    "channel": channel,
                    "scheduled_message_id": response.get("scheduled_message_id"),
                    "post_at": post_time.isoformat(),
                    "message": f"Message scheduled for {post_time.strftime('%Y-%m-%d %H:%M')}",
                },
            )
            
        except Exception as e:
            logger.error(f"Error scheduling message: {e}")
            return ToolResult(success=False, error=str(e))
    
    def _parse_time(self, time_str: str) -> datetime:
        """
        Parse a time string (ISO or relative).
        
        Examples:
            +5m -> 5 minutes from now
            +1h -> 1 hour from now
            10:00 -> 10:00 today
            2024-01-15T10:00:00 -> absolute time
        """
        now = datetime.now()
        
        # Relative time
        if time_str.startswith("+"):
            time_str = time_str[1:]
            
            if time_str.endswith("m"):
                minutes = int(time_str[:-1])
                return now + timedelta(minutes=minutes)
            elif time_str.endswith("h"):
                hours = int(time_str[:-1])
                return now + timedelta(hours=hours)
            elif time_str.endswith("d"):
                days = int(time_str[:-1])
                return now + timedelta(days=days)
        
        # Time of day (HH:MM)
        if ":" in time_str and len(time_str) <= 5:
            hour, minute = map(int, time_str.split(":"))
            result = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if result <= now:
                result += timedelta(days=1)
            return result
        
        # ISO format
        try:
            return datetime.fromisoformat(time_str)
        except ValueError:
            pass
        
        raise ValueError(f"Cannot parse time: {time_str}")
    
    async def _get_channel_info(self, channel: str) -> ToolResult:
        """Get information about a channel."""
        try:
            response = await self._call_api(
                "conversations.info",
                channel=channel,
            )
            
            if not response.get("ok"):
                return ToolResult(
                    success=False,
                    error=response.get("error"),
                )
            
            ch = response.get("channel", {})
            
            return ToolResult(
                success=True,
                data={
                    "id": ch.get("id"),
                    "name": ch.get("name"),
                    "topic": ch.get("topic", {}).get("value"),
                    "purpose": ch.get("purpose", {}).get("value"),
                    "member_count": ch.get("num_members"),
                    "is_private": ch.get("is_private", False),
                },
            )
            
        except Exception as e:
            logger.error(f"Error getting channel info: {e}")
            return ToolResult(success=False, error=str(e))
    
    async def _list_channels(
        self,
        types: str = "public_channel,private_channel",
    ) -> ToolResult:
        """List channels the bot is in."""
        try:
            response = await self._call_api(
                "conversations.list",
                types=types,
                exclude_archived=True,
            )
            
            if not response.get("ok"):
                return ToolResult(
                    success=False,
                    error=response.get("error"),
                )
            
            channels = [
                {
                    "id": ch.get("id"),
                    "name": ch.get("name"),
                    "is_member": ch.get("is_member", False),
                }
                for ch in response.get("channels", [])
                if ch.get("is_member", False)
            ]
            
            return ToolResult(
                success=True,
                data={
                    "channels": channels,
                    "count": len(channels),
                },
            )
            
        except Exception as e:
            logger.error(f"Error listing channels: {e}")
            return ToolResult(success=False, error=str(e))
    async def _get_thread_replies(
        self,
        channel: str,
        thread_ts: str,
    ) -> ToolResult:
        """
        Fetch all replies in a Slack thread.
        
        Critical for multi-user consensus: the full context of a group
        discussion is in the thread, not the main channel message.
        
        Args:
            channel: Channel ID
            thread_ts: Thread timestamp of the parent message
            
        Returns:
            ToolResult with all thread messages
        """
        try:
            channel_id = await self._resolve_channel(channel)
            
            response = await self._call_api(
                "conversations.replies",
                channel=channel_id,
                ts=thread_ts,
                limit=100,
            )
            
            if not response.get("ok"):
                return ToolResult(
                    success=False,
                    error=response.get("error", "Unknown error"),
                )
            
            messages = response.get("messages", [])
            formatted = [
                {
                    "user": m.get("user", "unknown"),
                    "text": m.get("text", ""),
                    "ts": m.get("ts", ""),
                }
                for m in messages
            ]
            
            return ToolResult(
                success=True,
                data={
                    "channel": channel_id,
                    "thread_ts": thread_ts,
                    "messages": formatted,
                    "count": len(formatted),
                },
            )
            
        except Exception as e:
            logger.error(f"Error getting thread replies: {e}")
            return ToolResult(success=False, error=str(e))
    
    async def _search_messages(
        self,
        query: str,
        count: int = 10,
    ) -> ToolResult:
        """
        Search for messages across all Slack channels.
        
        Uses Slack's search API which requires a user token (not bot token)
        with `search:read` scope.
        
        Args:
            query: Search query
            count: Max results to return
            
        Returns:
            ToolResult with matching messages
        """
        try:
            response = await self._call_api(
                "search.messages",
                query=query,
                count=count,
            )
            
            if not response.get("ok"):
                return ToolResult(
                    success=False,
                    error=response.get("error", "Unknown error"),
                )
            
            matches = response.get("messages", {}).get("matches", [])
            formatted = [
                {
                    "text": m.get("text", ""),
                    "user": m.get("user", ""),
                    "channel": m.get("channel", {}).get("name", ""),
                    "ts": m.get("ts", ""),
                    "permalink": m.get("permalink", ""),
                }
                for m in matches
            ]
            
            return ToolResult(
                success=True,
                data={
                    "results": formatted,
                    "count": len(formatted),
                    "query": query,
                },
            )
            
        except Exception as e:
            logger.error(f"Error searching messages: {e}")
            return ToolResult(success=False, error=str(e))    
