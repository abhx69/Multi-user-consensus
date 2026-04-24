import logging
import importlib
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

from gaprio.mcp.base_server import BaseMCPServer, MCPTool, ToolResult

logger = logging.getLogger(__name__)


class DynamicToolServer(BaseMCPServer):
    """
    Server for managing dynamically created tools.
    
    Allows the agent to create and register new Python tools at runtime.
    Tools are saved to `src/gaprio/mcp/dynamic/` and loaded on startup.
    """
    
    def __init__(self):
        self.dynamic_dir = Path(__file__).parent / "dynamic"
        self.dynamic_dir.mkdir(exist_ok=True)
        (self.dynamic_dir / "__init__.py").touch(exist_ok=True)
        
        super().__init__("dynamic")
        
        # Load existing dynamic tools after init
        self._load_existing_tools()
        
    def _register_tools(self) -> None:
        """Register the builder tool."""
        self.add_tool(MCPTool(
            name="create_tool",
            description=(
                "Create a new python tool dynamically. "
                "Use this when no existing tool can perform the requested task. "
                "The code must define a 'handler' async function."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name of the tool (snake_case)"
                    },
                    "description": {
                        "type": "string",
                        "description": "Description of what the tool does"
                    },
                    "code": {
                        "type": "string",
                        "description": "Complete Python code for the tool. Must include imports and an async handler(**kwargs) -> ToolResult function."
                    },
                    "requirements": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of PyPI packages required (e.g. ['requests', 'pandas'])"
                    }
                },
                "required": ["name", "description", "code"]
            },
            handler=self.create_tool_handler,
            category="dynamic"
        ))
    
    def _load_existing_tools(self) -> None:
        """Load all python files in the dynamic directory as tools."""
        for file_path in self.dynamic_dir.glob("*.py"):
            if file_path.name == "__init__.py":
                continue
                
            tool_name = file_path.stem
            try:
                self._load_tool_module(tool_name)
            except Exception as e:
                logger.error(f"Failed to load dynamic tool '{tool_name}': {e}")

    def _load_tool_module(self, name: str) -> None:
        """Import module and register tool."""
        module_name = f"gaprio.mcp.dynamic.{name}"
        
        # Force reload if already imported
        if module_name in sys.modules:
            module = importlib.reload(sys.modules[module_name])
        else:
            module = importlib.import_module(module_name)
            
        handler = getattr(module, "handler", None)
        if not handler:
            raise ValueError(f"Module {name} missing 'handler' function")
            
        # Register the tool
        # We assume description is in module docstring or we use default
        description = module.__doc__ or "Dynamic tool"
        
        # We try to infer parameters from handler signature or metadata variable
        # For now, we use flexible params
        parameters = getattr(module, "tool_parameters", {"type": "object"})
        
        self.add_tool(MCPTool(
            name=name,
            description=description,
            parameters=parameters,
            handler=handler,
            category="dynamic"
        ))
        logger.info(f"Loaded dynamic tool: {name}")

    async def create_tool_handler(
        self, 
        name: str, 
        code: str, 
        description: str,
        requirements: list[str] | None = None
    ) -> ToolResult:
        """
        Handler for create_tool.
        
        Saves code and registers the tool.
        """
        try:
            # Basic validation
            if not name.isidentifier():
                return ToolResult(False, error="Tool name must be a valid python identifier")
                
            # Save file
            file_path = self.dynamic_dir / f"{name}.py"
            
            # Add metadata comments
            full_code = f'"""{description}"""\n\n'
            if requirements:
                full_code += f"# Requirements: {', '.join(requirements)}\n"
            
            # Simple parameter definition injection if missing
            if "tool_parameters =" not in code:
                 full_code += 'tool_parameters = {"type": "object", "properties": {}, "additionalProperties": True}\n\n'
            
            full_code += code
            
            file_path.write_text(full_code, encoding="utf-8")
            
            # Load attempts
            self._load_tool_module(name)
            
            return ToolResult(
                success=True,
                data=f"Tool '{name}' created and registered successfully. You can now use it.",
                metadata={"path": str(file_path)}
            )
            
        except Exception as e:
            logger.error(f"Failed to create tool '{name}': {e}")
            return ToolResult(False, error=str(e))
