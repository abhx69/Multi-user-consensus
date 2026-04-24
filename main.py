"""
Main entry point for Gaprio Agent Bot.

Initializes all components and starts the Slack bot:
- Agent with LLM provider
- Memory system
- RAG system
- MCP tools
- Slack Bolt app

Run with: python -m gaprio.main
"""

import asyncio
import logging
import sys
from pathlib import Path

from gaprio.config import settings
from gaprio.agent.agent import Agent, ToolDefinition
from gaprio.agent.llm_provider import get_llm_provider
from gaprio.memory.memory_manager import MemoryManager
from gaprio.rag.indexer import SlackIndexer
from gaprio.rag.retriever import Retriever
from gaprio.mcp.base_server import MCPRegistry
from gaprio.mcp.slack_server import SlackMCPServer
from gaprio.mcp.github_server import GitHubMCPServer
from gaprio.mcp.notion_server import NotionMCPServer
from gaprio.mcp.external_server import ExternalMCPServer
from gaprio.mcp.dynamic_server import DynamicToolServer
from gaprio.slack.app import create_slack_app, create_socket_mode_handler
from gaprio.slack.handlers import register_handlers

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)

logger = logging.getLogger(__name__)


def setup_directories() -> None:
    """Create necessary directories."""
    settings.ensure_directories()
    logger.info(f"Data directory: {settings.data_dir}")


def create_mcp_registry(slack_client=None) -> MCPRegistry:
    """
    Create and configure the MCP registry with all tool servers.
    
    Args:
        slack_client: Optional Slack client for the Slack MCP server
        
    Returns:
        Configured MCPRegistry
    """
    registry = MCPRegistry()
    
    # Slack MCP Server
    slack_server = SlackMCPServer(slack_client)
    registry.register(slack_server)
    
    # GitHub MCP Server (if configured)
    if settings.github_token:
        github_server = GitHubMCPServer()
        registry.register(github_server)
        logger.info("GitHub MCP server enabled")
    else:
        logger.warning("GitHub token not configured - GitHub tools disabled")
    
    # Notion MCP Server (if configured)
    if settings.notion_token:
        notion_server = NotionMCPServer()
        registry.register(notion_server)
        logger.info("Notion MCP server enabled")
    else:
        logger.warning("Notion token not configured - Notion tools disabled")
    
    # Asana MCP Server (if configured)
    if settings.asana_refresh_token or settings.asana_access_token:
        from gaprio.mcp.asana_server import AsanaMCPServer
        asana_server = AsanaMCPServer()
        registry.register(asana_server)
        logger.info("Asana MCP server enabled")
    else:
        logger.warning("Asana token not configured - Asana tools disabled")
    
    # Google Workspace MCP Server (if configured via .env token OR OAuth client for DB tokens)
    if settings.google_refresh_token or settings.google_client_id:
        from gaprio.mcp.google_server import GoogleMCPServer
        google_server = GoogleMCPServer()
        registry.register(google_server)
        logger.info("Google Workspace MCP server enabled")
    else:
        logger.warning("Google not configured - Google Workspace tools disabled")
    
    # Dynamic MCP Server
    dynamic_server = DynamicToolServer()
    registry.register(dynamic_server)
    logger.info("Dynamic MCP server enabled")
    # Knowledge Graph MCP Server
    from gaprio.mcp.knowledge_graph_server import KnowledgeGraphMCPServer
    kg_server = KnowledgeGraphMCPServer()
    registry.register(kg_server)
    logger.info("Knowledge Graph MCP server enabled")
    
    return registry


def register_agent_tools(agent: Agent, mcp_registry: MCPRegistry) -> None:
    """
    Register all MCP tools with the agent.
    
    Args:
        agent: Agent instance
        mcp_registry: MCP registry with tool servers
    """
    for tool in mcp_registry.get_all_tools():
        agent.register_tool(ToolDefinition(
            name=tool.name,
            description=tool.description,
            parameters=tool.parameters,
            handler=tool.handler,
        ))
    
    logger.info(f"Registered {len(agent.tools)} tools with agent")


async def setup_rag(slack_client) -> tuple[SlackIndexer, Retriever]:
    """
    Set up the RAG system.
    
    Args:
        slack_client: Slack client for fetching messages
        
    Returns:
        Tuple of (indexer, retriever)
    """
    indexer = SlackIndexer(slack_client=slack_client)
    retriever = Retriever()
    
    # Initial indexing (in background)
    # We'll trigger this after the bot starts
    
    return indexer, retriever


async def run_scheduled_tasks(
    memory: MemoryManager,
    slack_client,
    indexer: SlackIndexer | None = None,
) -> None:
    """
    Run scheduled tasks (reminders, recurring messages, RAG indexing).
    
    This runs in a background loop checking for due items.
    
    Args:
        memory: Memory manager with heartbeat
        slack_client: Slack client for posting
        indexer: Optional SlackIndexer for RAG indexing
    """
    from apscheduler.schedulers.asyncio import AsyncIOScheduler
    
    scheduler = AsyncIOScheduler()
    
    async def check_due_items():
        """Check for and execute due items."""
        due = memory.get_due_items()
        
        # Handle reminders
        for reminder in due.get("reminders", []):
            try:
                channel = reminder.get("channel") or reminder.get("user_id")
                if channel:
                    await slack_client.chat_postMessage(
                        channel=channel,
                        text=f"⏰ Reminder: {reminder['message']}",
                    )
                memory.heartbeat.mark_reminder_complete(reminder["id"])
                logger.info(f"Sent reminder: {reminder['id']}")
            except Exception as e:
                logger.error(f"Failed to send reminder: {e}")
        
        # Handle recurring tasks
        for task in due.get("tasks", []):
            try:
                if task.get("action") == "post_message":
                    channel = task.get("channel")
                    message = task.get("action_params", {}).get("message", "")
                    if channel and message:
                        await slack_client.chat_postMessage(
                            channel=channel,
                            text=message,
                        )
                        memory.heartbeat.update_recurring_task_run(task["id"])
                        logger.info(f"Executed recurring task: {task['name']}")
            except Exception as e:
                logger.error(f"Failed to execute task: {e}")
    
    async def run_rag_indexing():
        """Run periodic RAG indexing of Slack channels."""
        if not indexer:
            return
        try:
            logger.info("Starting scheduled RAG indexing...")
            stats = await indexer.index_all_channels()
            logger.info(f"RAG indexing complete: {stats}")
        except Exception as e:
            logger.error(f"RAG indexing failed: {e}")
    
    # Check reminders every minute
    scheduler.add_job(check_due_items, "interval", minutes=1)
    
    # Run RAG indexing based on configured frequency
    if indexer:
        scheduler.add_job(
            run_rag_indexing, 
            "interval", 
            hours=settings.rag_index_frequency_hours,
            id="rag_indexing"
        )
        logger.info(f"RAG indexing scheduled every {settings.rag_index_frequency_hours} hours")
    
    scheduler.start()
    
    logger.info("Scheduled task runner started")


async def register_external_mcp_servers(registry: MCPRegistry) -> None:
    """
    Connect and register external MCP servers configured in Settings.
    
    Parses the command string and initializes an ExternalMCPServer bridge
    for each configured server. 
    """
    if not settings.external_mcp_servers:
        return

    logger.info(f"Connecting to {len(settings.external_mcp_servers)} external MCP servers...")
    
    for name, cmd_str in settings.external_mcp_servers.items():
        try:
            # Parse command string (simple split for now)
            # e.g. "npx -y @modelcontextprotocol/server-slack"
            parts = cmd_str.split()
            if not parts:
                continue
                
            command = parts[0]
            args = parts[1:]
            
            logger.debug(f"Loading external server '{name}': {command} {args}")
            
            # Create bridge
            # Pass current environment to subprocess
            import os
            server = ExternalMCPServer(name, command, args, env=os.environ.copy())
            
            # Initialize (connects process)
            await server.initialize()
            
            # Register in registry
            registry.register(server)
            logger.info(f"Successfully registered external server: {name}")
            
        except Exception as e:
            logger.error(f"Failed to load external MCP server '{name}': {e}")


async def async_main() -> None:
    """
    Async main entry point.
    
    Initializes all systems and starts the Slack bot.
    """
    logger.info("=" * 50)
    logger.info("Starting Gaprio Agent Bot")
    logger.info("=" * 50)
    
    # Setup
    setup_directories()
    
    # Check required configuration
    if not settings.slack_bot_token:
        logger.error("SLACK_BOT_TOKEN is required. Set it in .env file.")
        sys.exit(1)
    
    if not settings.slack_app_token:
        logger.error("SLACK_APP_TOKEN is required for Socket Mode. Set it in .env file.")
        sys.exit(1)
    
    # Create Slack app
    app = create_slack_app()
    slack_client = app.client
    
    # Create components
    logger.info("Initializing components...")
    
    # LLM Provider
    llm_provider = get_llm_provider()
    logger.info(f"LLM Provider: {llm_provider.__class__.__name__}")
    
    # Memory
    memory = MemoryManager()
    
    # Retriever (lazy initialization of vector store)
    retriever = Retriever()
    
    # RAG Indexer
    indexer = SlackIndexer(slack_client=slack_client)
    
    # Agent
    agent = Agent(
        llm_provider=llm_provider,
        memory_manager=memory,
        retriever=retriever,
    )
    
    # MCP Registry
    mcp_registry = create_mcp_registry(slack_client)
    
    # Register external MCP servers
    await register_external_mcp_servers(mcp_registry)
    
    # Update Slack MCP server with client
    slack_server = mcp_registry.get_server("slack")
    if slack_server:
        slack_server.set_client(slack_client)
    
    # Register tools with agent
    register_agent_tools(agent, mcp_registry)
    
    # Register Slack handlers
    register_handlers(
        app=app,
        agent=agent,
        memory=memory,
        indexer=indexer,
        mcp_registry=mcp_registry,
    )
    
    # Create Socket Mode handler
    handler = create_socket_mode_handler(app)
    
    logger.info("All components initialized")
    logger.info("")
    logger.info("Bot is starting...")
    logger.info("Send a message to the bot in Slack to test!")
    logger.info("")
    
    # Also print to stdout directly in case logger output isn't captured
    print("=" * 50)
    print("🤖 GAPRIO BOT STARTING")
    print("=" * 50)
    print(f"📡 Socket Mode connecting with token: {settings.slack_app_token[:20]}...")
    print(f"🔧 Registered {len(agent.tools)} tools")
    print("=" * 50)
    
    # Start scheduled tasks (reminders, RAG indexing)
    await run_scheduled_tasks(memory, slack_client, indexer)
    
    # Trigger initial RAG indexing in background
    async def initial_indexing():
        await asyncio.sleep(5)  # Wait for bot to fully connect
        logger.info("Running initial RAG indexing...")
        try:
            stats = await indexer.index_all_channels()
            logger.info(f"Initial RAG indexing complete: {stats}")
        except Exception as e:
            logger.error(f"Initial RAG indexing failed: {e}")
    
    asyncio.create_task(initial_indexing())
    
    # Start the bot (async)
    try:
        print("🚀 Starting Socket Mode handler... (if this is the last line, the handler is running)")
        await handler.start_async()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.error(f"Error: {e}")
        print(f"❌ FATAL ERROR: {e}")
        sys.exit(1)


def main() -> None:
    """Sync entry point wrapper for console script."""
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
