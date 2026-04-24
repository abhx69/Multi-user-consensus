"""
Slack event handlers.

Handles incoming Slack events:
- App mentions (@bot)
- Direct messages
- Slash commands

Each handler extracts context and passes requests to the agent for processing.
"""

import logging
from typing import Any, Callable

from slack_bolt.async_app import AsyncApp

from gaprio.agent.agent import Agent, AgentContext
from gaprio.memory.memory_manager import MemoryManager
from gaprio.rag.indexer import SlackIndexer
from gaprio.mcp.base_server import MCPRegistry

logger = logging.getLogger(__name__)


class SlackHandlers:
    """
    Manages Slack event handlers and agent integration.
    
    Coordinates between Slack events and the agent system,
    handling context building and response posting.
    """
    
    def __init__(
        self,
        agent: Agent,
        memory: MemoryManager,
        indexer: SlackIndexer | None = None,
        mcp_registry: MCPRegistry | None = None,
    ):
        """
        Initialize handlers.
        
        Args:
            agent: Agent instance for processing requests
            memory: Memory manager for context
            indexer: Optional indexer for RAG updates
            mcp_registry: Optional MCP registry for tools
        """
        self.agent = agent
        self.memory = memory
        self.indexer = indexer
        self.mcp_registry = mcp_registry
    
    async def handle_app_mention(
        self,
        event: dict,
        say: Callable,
        client: Any,
    ) -> None:
        """
        Handle @bot mentions in channels.
        
        Args:
            event: Slack event data
            say: Function to post a response
            client: Slack client
        """
        user_id = event.get("user", "")
        channel_id = event.get("channel", "")
        text = event.get("text", "")
        thread_ts = event.get("thread_ts") or event.get("ts")
        
        # Remove the bot mention from the text
        # Format is usually "<@BOTID> message"
        message = self._clean_mention(text)
        
        logger.info(f"App mention from {user_id} in {channel_id}: {message[:50]}...")
        
        # Build context
        session_id = f"{channel_id}:{user_id}"
        conversation = self.memory.get_or_create_conversation(session_id)
        conversation.add_user_message(message, user_id=user_id, channel_id=channel_id)
        
        context = AgentContext(
            user_id=user_id,
            channel_id=channel_id,
            message=message,
            conversation_history=conversation.get_messages(),
            metadata={
                "thread_ts": thread_ts,
                "event_type": "app_mention",
            },
        )
        
        # Show typing indicator
        # Note: Slack doesn't have a built-in typing indicator for bots
        # but we can acknowledge quickly
        
        try:
            # Process with agent
            response = await self.agent.process(context)
            
            # Add response to conversation
            conversation.add_assistant_message(response.text)
            
            # Post reply
            await say(
                text=response.text,
                thread_ts=thread_ts,
            )
            
            logger.info(f"Responded in {channel_id}")
            
        except Exception as e:
            logger.error(f"Error handling mention: {e}")
            await say(
                text=f"Sorry, I encountered an error: {str(e)[:100]}",
                thread_ts=thread_ts,
            )
    
    async def handle_direct_message(
        self,
        event: dict,
        say: Callable,
        client: Any,
    ) -> None:
        """
        Handle direct messages to the bot.
        
        Args:
            event: Slack event data
            say: Function to post a response
            client: Slack client
        """
        user_id = event.get("user", "")
        channel_id = event.get("channel", "")
        text = event.get("text", "")
        
        # Ignore bot's own messages
        if event.get("bot_id"):
            return
        
        logger.info(f"DM from {user_id}: {text[:50]}...")
        
        # Build context
        session_id = f"dm:{user_id}"
        conversation = self.memory.get_or_create_conversation(session_id)
        conversation.add_user_message(text, user_id=user_id)
        
        context = AgentContext(
            user_id=user_id,
            channel_id=channel_id,
            message=text,
            conversation_history=conversation.get_messages(),
            metadata={
                "event_type": "direct_message",
            },
        )
        
        try:
            # Process with agent
            response = await self.agent.process(context)
            
            # Add response to conversation
            conversation.add_assistant_message(response.text)
            
            # Reply
            await say(text=response.text)
            
        except Exception as e:
            logger.error(f"Error handling DM: {e}")
            await say(text=f"Sorry, I encountered an error: {str(e)[:100]}")
    
    def _clean_mention(self, text: str) -> str:
        """Remove bot mention from message text."""
        import re
        # Remove <@USERID> patterns
        cleaned = re.sub(r"<@[A-Z0-9]+>\s*", "", text).strip()
        return cleaned


def register_handlers(
    app: AsyncApp,
    agent: Agent,
    memory: MemoryManager,
    indexer: SlackIndexer | None = None,
    mcp_registry: MCPRegistry | None = None,
) -> SlackHandlers:
    """
    Register all event handlers with the Slack app.
    
    Args:
        app: Slack Bolt App
        agent: Agent for processing
        memory: Memory manager
        indexer: Optional RAG indexer
        mcp_registry: Optional MCP registry
        
    Returns:
        SlackHandlers instance
    """
    handlers = SlackHandlers(
        agent=agent,
        memory=memory,
        indexer=indexer,
        mcp_registry=mcp_registry,
    )
    
    # ========================
    # Diagnostic middleware: logs ALL incoming events
    # Uses print() to guarantee output shows in terminal
    # ========================
    @app.middleware
    async def log_all_events(body, next):
        event = body.get("event", {})
        event_type = event.get("type", body.get("type", "unknown"))
        channel = event.get("channel", "N/A")
        channel_type = event.get("channel_type", "N/A")
        user = event.get("user", "N/A")
        subtype = event.get("subtype", "none")
        print(
            f"🔔 [Slack Event] type={event_type} subtype={subtype} "
            f"channel={channel} channel_type={channel_type} user={user}",
            flush=True
        )
        logger.info(
            f"🔔 [Slack Event] type={event_type} subtype={subtype} "
            f"channel={channel} channel_type={channel_type} user={user}"
        )
        await next()
    
    logger.info("📋 Registering Slack event handlers...")
    
    # App mentions (in channels)
    @app.event("app_mention")
    async def handle_mention(event, say, client):
        await handlers.handle_app_mention(event, say, client)
    
    # Direct messages AND channel monitoring
    @app.event("message")
    async def handle_message(event, say, client):
        # Ignore message subtypes (edits, deletes, etc.)
        if event.get("subtype"):
            return
        
        # Ignore bot messages
        if event.get("bot_id"):
            return
        
        channel_type = event.get("channel_type", "")
        
        # Handle DMs normally
        if channel_type == "im":
            await handlers.handle_direct_message(event, say, client)
            return
        
        # Channel messages → monitoring pipeline (non-blocking)
        channel_id = event.get("channel", "")
        text = event.get("text", "")
        user_id = event.get("user", "")
        
        logger.info(f"📡 [Monitor] Channel message event: ch={channel_id} user={user_id} len={len(text)} text={text[:80]}")
        
        if text and len(text.strip()) >= 10:
            import asyncio
            asyncio.create_task(
                _process_monitored_message(channel_id, text, user_id)
            )
        else:
            logger.debug(f"📡 [Monitor] Skipping short/empty message in {channel_id}")
    
    # Slash command: /gaprio
    @app.command("/gaprio")
    async def handle_gaprio_command(ack, command, say):
        await ack()
        
        text = command.get("text", "").strip()
        user_id = command.get("user_id", "")
        channel_id = command.get("channel_id", "")
        
        if not text:
            await say(
                text="Usage: `/gaprio <your request>`\n"
                     "Example: `/gaprio summarize #general for the past 24 hours`"
            )
            return
        
        # Build context
        session_id = f"cmd:{user_id}:{channel_id}"
        conversation = memory.get_or_create_conversation(session_id)
        conversation.add_user_message(text, user_id=user_id)
        
        context = AgentContext(
            user_id=user_id,
            channel_id=channel_id,
            message=text,
            conversation_history=conversation.get_messages(),
            metadata={"event_type": "slash_command"},
        )
        
        try:
            response = await agent.process(context)
            conversation.add_assistant_message(response.text)
            await say(text=response.text)
        except Exception as e:
            logger.error(f"Error in slash command: {e}")
            await say(text=f"Error: {str(e)[:100]}")
    
    # Health check - responds to home tab open
    @app.event("app_home_opened")
    async def handle_home_opened(event, client):
        user_id = event.get("user")
        
        try:
            # Update the home tab with a simple view
            await client.views_publish(
                user_id=user_id,
                view={
                    "type": "home",
                    "blocks": [
                        {
                            "type": "section",
                            "text": {
                                "type": "mrkdwn",
                                "text": "*Welcome to Gaprio! 👋*\n\n"
                                        "I'm your AI assistant. Here's what I can do:\n\n"
                                        "• Summarize channel conversations\n"
                                        "• Schedule reminders and messages\n"
                                        "• Create GitHub issues from discussions\n"
                                        "• Create Notion pages for documentation\n\n"
                                        "*How to use:*\n"
                                        "• Mention me in a channel: `@Gaprio summarize today`\n"
                                        "• Send me a DM\n"
                                        "• Use `/gaprio <request>`"
                            }
                        },
                        {
                            "type": "divider"
                        },
                        {
                            "type": "context",
                            "elements": [
                                {
                                    "type": "mrkdwn",
                                    "text": "Built with ❤️ using Gaprio Agent Bot"
                                }
                            ]
                        }
                    ]
                }
            )
        except Exception as e:
            logger.error(f"Error updating home tab: {e}")
    
    logger.info("Slack handlers registered")
    return handlers


async def _process_monitored_message(channel_id: str, text: str, sender_user_id: str) -> None:
    """
    Process a channel message through the monitoring pipeline.
    
    1. Check if any user is monitoring this channel (via Express backend DB)
    2. If yes, send the context to Express → Python agent /analyze-context
    3. Express stores the resulting suggested actions
    """
    import httpx
    
    EXPRESS_URL = "http://localhost:5000"
    
    logger.info(f"📡 [Monitor] Processing message: ch={channel_id} sender={sender_user_id} text={text[:80]}")
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Step 1: Check which users monitor this channel
            # We call Express backend which queries monitored_channels table
            resp = await client.get(
                f"{EXPRESS_URL}/api/monitoring/channels/check/{channel_id}"
            )
            
            if resp.status_code != 200:
                logger.warning(f"📡 [Monitor] Check endpoint returned {resp.status_code}")
                return
            
            data = resp.json()
            user_ids = data.get("user_ids", [])
            
            if not user_ids:
                logger.debug(f"📡 [Monitor] No users monitoring channel {channel_id}")
                return
            
            logger.info(f"📡 [Monitor] Channel {channel_id}: {len(user_ids)} monitoring user(s) = {user_ids}")
            
            # Step 2: For each monitoring user, send context to the agent
            for uid in user_ids:
                try:
                    analyze_resp = await client.post(
                        f"{EXPRESS_URL}/api/monitoring/internal/analyze",
                        json={
                            "userId": uid,
                            "platform": "slack",
                            "channelId": channel_id,
                            "context": text,
                            "metadata": {"sender": sender_user_id}
                        },
                        timeout=60.0
                    )
                    logger.info(f"📡 [Monitor] Analyze response for user {uid}: status={analyze_resp.status_code} body={analyze_resp.text[:300]}")
                except Exception as inner_err:
                    logger.warning(f"📡 [Monitor] Failed to analyze for user {uid}: {inner_err}")
                    
    except httpx.ConnectError:
        logger.warning("📡 [Monitor] Express backend not reachable (ConnectError)")
    except Exception as e:
        logger.error(f"📡 [Monitor] Pipeline error: {e}")

