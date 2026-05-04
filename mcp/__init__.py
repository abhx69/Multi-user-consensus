"""
MCP module - Model Context Protocol tools for external integrations.

Each MCP server exposes tools that the agent can invoke:
- SlackServer: Read/post messages, schedule messages, channel management
- AsanaServer: Create/manage tasks, projects, comments
- GoogleServer: Gmail, Calendar, Drive integration
"""

from gaprio.mcp.base_server import BaseMCPServer, MCPTool, ToolResult, MCPRegistry
from gaprio.mcp.slack_server import SlackMCPServer
from gaprio.mcp.asana_server import AsanaMCPServer
from gaprio.mcp.google_server import GoogleMCPServer
from gaprio.mcp.external_server import ExternalMCPServer
from gaprio.mcp.dynamic_server import DynamicToolServer

__all__ = [
    "BaseMCPServer",
    "MCPTool",
    "ToolResult",
    "MCPRegistry",
    "SlackMCPServer",
    "AsanaMCPServer",
    "GoogleMCPServer",
    "ExternalMCPServer",
    "DynamicToolServer",
]
