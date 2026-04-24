import asyncio
import logging
import json
from contextlib import AsyncExitStack
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from gaprio.config import settings
from gaprio.mcp.base_server import BaseMCPServer, MCPTool, ToolResult

logger = logging.getLogger(__name__)


class ExternalMCPServer(BaseMCPServer):
    """
    Bridge for external MCP servers (standard JSON-RPC).
    
    Connects to an external process (e.g., Node.js server) via stdio
    and exposes its tools as Gaprio internal tools.
    """
    
    def __init__(
        self, 
        name: str, 
        command: str, 
        args: list[str], 
        env: dict[str, str] | None = None
    ):
        """
        Initialize the external server bridge.
        
        Args:
            name: Server name (e.g., "slack-ext")
            command: Command to run (e.g., "npx")
            args: Arguments (e.g., ["-y", "@modelcontextprotocol/server-slack"])
            env: Environment variables
        """
        # We don't call super().__init__ because it calls _register_tools synchronously
        # and we need async initialization. Instead we manually set up.
        self.name = name
        self.tools: dict[str, MCPTool] = {}
        
        self.command = command
        self.args = args
        self.env = env
        
        self.session: ClientSession | None = None
        self._exit_stack = AsyncExitStack()
        
        logger.info(f"External MCP Server '{name}' bridge created")
    
    def _register_tools(self) -> None:
        """
        Placeholder - tools are registered dynamically in initialize().
        """
        pass
    
    async def initialize(self) -> None:
        """
        Connect to the external server and discover tools.
        
        This must be called before using the server.
        """
        server_params = StdioServerParameters(
            command=self.command,
            args=self.args,
            env=self.env,
        )
        
        try:
            # Connect via stdio
            read, write = await self._exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            
            # Create session
            self.session = await self._exit_stack.enter_async_context(
                ClientSession(read, write)
            )
            
            # Initialize session
            await self.session.initialize()
            
            # List tools
            tools_result = await self.session.list_tools()
            
            # Register tools
            for tool in tools_result.tools:
                # Map external tool to internal MCPTool
                internal_tool = MCPTool(
                    name=tool.name,
                    description=tool.description or "",
                    parameters=tool.inputSchema or {},
                    handler=self._create_handler(tool.name),
                    category=self.name,
                )
                self.add_tool(internal_tool)
                
            logger.info(f"Initialized external server '{self.name}' with {len(self.tools)} tools")
            
        except Exception as e:
            logger.error(f"Failed to initialize external server '{self.name}': {e}")
            # Clean up if failed
            await self.shutdown()
            raise
    
    def _create_handler(self, tool_name: str):
        """Create a closure for handling tool execution."""
        async def handler(**kwargs) -> ToolResult:
            if not self.session:
                return ToolResult(
                    success=False,
                    error="Server not connected",
                )
            
            try:
                # Call tool on external server
                result = await self.session.call_tool(tool_name, arguments=kwargs)
                
                # Format result
                # Standard MCP result is a list of content (text/image)
                content_text = ""
                for content in result.content:
                    if content.type == "text":
                        content_text += content.text
                    elif content.type == "image":
                        content_text += f"[Image: {content.mimeType}]"
                        
                return ToolResult(
                    success=not result.isError,
                    data=content_text,
                    metadata={"source": self.name}
                )
                
            except Exception as e:
                logger.error(f"External tool '{tool_name}' failed: {e}")
                return ToolResult(
                    success=False,
                    error=str(e),
                )
        
        return handler

    async def shutdown(self) -> None:
        """Clean up resources."""
        await self._exit_stack.aclose()
        self.session = None
