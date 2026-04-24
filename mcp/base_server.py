"""
Base MCP Server implementation.

Provides the foundation for all MCP (Model Context Protocol) tool servers.
Each server exposes a set of tools that the agent can invoke to perform
actions on external systems (Slack, GitHub, Notion, etc.).

The MCP pattern provides:
- Consistent tool interface for the agent
- Type-safe parameter handling
- Error handling and logging
- Easy addition of new tool servers
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable

logger = logging.getLogger(__name__)


@dataclass
class MCPTool:
    """
    Definition of an MCP tool.
    
    Attributes:
        name: Unique identifier (e.g., "slack_read_messages")
        description: What the tool does (for LLM to understand)
        parameters: JSON Schema for tool parameters
        handler: Async function that executes the tool
        category: Tool category for grouping
    """
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., Awaitable[Any]]
    category: str = "general"
    
    def to_openai_format(self) -> dict[str, Any]:
        """Convert to OpenAI function calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            }
        }
    
    def to_agent_format(self) -> dict[str, Any]:
        """Convert to agent tool format."""
        from gaprio.agent.agent import ToolDefinition
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
            handler=self.handler,
        )


@dataclass
class ToolResult:
    """
    Result from a tool execution.
    
    Attributes:
        success: Whether the tool executed successfully
        data: The result data (if successful)
        error: Error message (if failed)
        metadata: Additional info about the execution
    """
    success: bool
    data: Any = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "metadata": self.metadata,
        }


class BaseMCPServer(ABC):
    """
    Abstract base class for MCP tool servers.
    
    Each server manages a collection of related tools (e.g., all Slack tools).
    The server handles:
    - Tool registration
    - Tool execution
    - Result formatting
    
    Subclasses must implement:
    - _register_tools(): Define available tools
    - Any tool handler methods
    
    Usage:
        class MyServer(BaseMCPServer):
            def _register_tools(self):
                self.add_tool(MCPTool(
                    name="my_tool",
                    description="Does something",
                    parameters={...},
                    handler=self._handle_my_tool,
                ))
            
            async def _handle_my_tool(self, **params) -> ToolResult:
                # Implementation
                return ToolResult(success=True, data="result")
    """
    
    def __init__(self, name: str):
        """
        Initialize the MCP server.
        
        Args:
            name: Server name (e.g., "slack", "github")
        """
        self.name = name
        self.tools: dict[str, MCPTool] = {}
        self._register_tools()
        
        logger.info(f"MCP Server '{name}' initialized with {len(self.tools)} tools")
    
    @abstractmethod
    def _register_tools(self) -> None:
        """
        Register all tools provided by this server.
        
        Subclasses must implement this to define their tools.
        """
        pass
    
    def add_tool(self, tool: MCPTool) -> None:
        """
        Add a tool to the server.
        
        Args:
            tool: Tool definition to add
        """
        tool.category = self.name
        self.tools[tool.name] = tool
        logger.debug(f"Registered tool: {tool.name}")
    
    def get_tool(self, name: str) -> MCPTool | None:
        """
        Get a tool by name.
        
        Args:
            name: Tool name
            
        Returns:
            Tool definition, or None if not found
        """
        return self.tools.get(name)
    
    def list_tools(self) -> list[MCPTool]:
        """Get all tools from this server."""
        return list(self.tools.values())
    
    def get_tools_for_agent(self) -> list[dict[str, Any]]:
        """Get all tools in agent-compatible format."""
        return [tool.to_openai_format() for tool in self.tools.values()]
    
    async def execute(self, tool_name: str, **params) -> ToolResult:
        """
        Execute a tool by name.
        
        Args:
            tool_name: Name of the tool to execute
            **params: Tool parameters
            
        Returns:
            ToolResult with execution results
        """
        tool = self.get_tool(tool_name)
        
        if not tool:
            return ToolResult(
                success=False,
                error=f"Unknown tool: {tool_name}",
            )
        
        try:
            logger.info(f"Executing tool: {tool_name}")
            result = await tool.handler(**params)
            
            # Ensure result is a ToolResult
            if not isinstance(result, ToolResult):
                result = ToolResult(success=True, data=result)
            
            return result
            
        except Exception as e:
            logger.error(f"Tool {tool_name} failed: {e}")
            return ToolResult(
                success=False,
                error=str(e),
                metadata={"tool": tool_name, "params": params},
            )
    
    def get_descriptions(self) -> str:
        """
        Get human-readable descriptions of all tools.
        
        Returns:
            Formatted string describing each tool
        """
        lines = [f"## {self.name.upper()} Tools\n"]
        for tool in self.tools.values():
            lines.append(f"- **{tool.name}**: {tool.description}")
        return "\n".join(lines)


class MCPRegistry:
    """
    Registry for all MCP servers.
    
    Provides a unified interface for discovering and executing
    tools across all registered servers.
    
    Usage:
        registry = MCPRegistry()
        registry.register(SlackMCPServer())
        registry.register(GitHubMCPServer())
        
        # Get all tools
        tools = registry.get_all_tools()
        
        # Execute a tool
        result = await registry.execute("slack_read_messages", channel="C123")
    """
    
    def __init__(self):
        """Initialize the registry."""
        self.servers: dict[str, BaseMCPServer] = {}
    
    def register(self, server: BaseMCPServer) -> None:
        """
        Register an MCP server.
        
        Args:
            server: Server to register
        """
        self.servers[server.name] = server
        logger.info(f"Registered MCP server: {server.name}")
    
    def get_server(self, name: str) -> BaseMCPServer | None:
        """Get a server by name."""
        return self.servers.get(name)
    
    def get_all_tools(self) -> list[MCPTool]:
        """Get all tools from all servers."""
        tools = []
        for server in self.servers.values():
            tools.extend(server.list_tools())
        return tools
    
    def get_tools_for_agent(self) -> list[dict[str, Any]]:
        """Get all tools in agent-compatible format."""
        tools = []
        for server in self.servers.values():
            tools.extend(server.get_tools_for_agent())
        return tools
    
    async def execute(self, tool_name: str, **params) -> ToolResult:
        """
        Execute a tool by name (searches all servers).
        
        Args:
            tool_name: Tool to execute
            **params: Tool parameters
            
        Returns:
            ToolResult
        """
        for server in self.servers.values():
            tool = server.get_tool(tool_name)
            if tool:
                return await server.execute(tool_name, **params)
        
        return ToolResult(
            success=False,
            error=f"Unknown tool: {tool_name}",
        )
    
    def get_tool_by_name(self, tool_name: str) -> MCPTool | None:
        """Find a tool across all servers."""
        for server in self.servers.values():
            tool = server.get_tool(tool_name)
            if tool:
                return tool
        return None
