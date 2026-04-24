"""
FastAPI REST API for Gaprio Agent.

Provides HTTP endpoints to interact with the Gaprio Agent
from custom frontends, mobile apps, or other services.

Run with: gaprio-api or python -m gaprio.api_main
"""

import logging
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from gaprio.agent.agent import Agent, AgentContext, AgentResponse
from gaprio.agent.llm_provider import get_llm_provider
from gaprio.memory.memory_manager import MemoryManager
from gaprio.mcp.base_server import MCPRegistry
from gaprio.main import create_mcp_registry, register_agent_tools

logger = logging.getLogger(__name__)

# =============================================================================
# Global State
# =============================================================================

_agent: Agent | None = None
_mcp_registry: MCPRegistry | None = None
_sessions: dict[str, list[dict[str, str]]] = {}  # session_id -> conversation history


# =============================================================================
# Request/Response Models
# =============================================================================

class ChatRequest(BaseModel):
    """Request body for /api/chat endpoint."""
    message: str = Field(..., description="User message to the agent")
    user_id: str = Field(default="api-user", description="User identifier")
    session_id: str | None = Field(default=None, description="Session ID for conversation continuity")


class ChatResponse(BaseModel):
    """Response body for /api/chat endpoint."""
    response: str = Field(..., description="Agent's response text")
    session_id: str = Field(..., description="Session ID for this conversation")
    tool_calls: list[dict[str, Any]] = Field(default_factory=list, description="Tools executed")
    data: dict[str, Any] | None = Field(default=None, description="Structured data for UI components")
    success: bool = Field(default=True, description="Whether the request was successful")
    timestamp: str = Field(..., description="Response timestamp")


class HealthResponse(BaseModel):
    """Response body for /api/health endpoint."""
    status: str
    agent_ready: bool
    tools_count: int
    uptime: str


class ToolInfo(BaseModel):
    """Information about a single tool."""
    name: str
    description: str
    parameters: dict[str, Any]


class ToolsResponse(BaseModel):
    """Response body for /api/tools endpoint."""
    tools: list[ToolInfo]
    count: int


# =============================================================================
# Lifecycle Management
# =============================================================================

_start_time: datetime = datetime.now()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle - startup and shutdown."""
    global _agent, _mcp_registry, _start_time
    
    logger.info("Starting Gaprio API Server...")
    
    # Initialize components
    llm = get_llm_provider()
    memory = MemoryManager()
    _agent = Agent(llm_provider=llm, memory_manager=memory)
    
    # Create Slack client from bot token if available
    slack_client = None
    from gaprio.config import settings
    if settings.slack_bot_token:
        try:
            from slack_sdk import WebClient
            slack_client = WebClient(token=settings.slack_bot_token)
            logger.info("Slack client initialized for API mode")
        except ImportError:
            logger.warning("slack_sdk not installed - Slack tools will not work")
        except Exception as e:
            logger.warning(f"Failed to create Slack client: {e}")
    else:
        logger.warning("SLACK_BOT_TOKEN not set - Slack tools will not work")
    
    # Create MCP registry with Slack client
    _mcp_registry = create_mcp_registry(slack_client=slack_client)
    
    # Register tools with agent
    register_agent_tools(_agent, _mcp_registry)
    
    _start_time = datetime.now()
    logger.info(f"Gaprio API ready with {len(_agent.tools)} tools")
    
    yield
    
    # Cleanup
    logger.info("Shutting down Gaprio API Server...")
    if _agent:
        await _agent.close()
    # Close MySQL pool for DB token bridge
    from gaprio.db_tokens import close_pool
    await close_pool()


# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title="Gaprio Agent API",
    description="REST API for interacting with the Gaprio AI Agent",
    version="0.1.0",
    lifespan=lifespan,
)

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for local development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# API Endpoints
# =============================================================================

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Send a message to the Gaprio Agent and get a response.
    
    This is the main endpoint for interacting with the agent.
    Use session_id to maintain conversation context across requests.
    """
    global _agent, _sessions
    
    if not _agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    # Generate or use existing session ID
    session_id = request.session_id or str(uuid.uuid4())
    
    # Get or create conversation history for this session
    if session_id not in _sessions:
        _sessions[session_id] = []
    
    conversation_history = _sessions[session_id]
    
    # Create agent context
    context = AgentContext(
        user_id=request.user_id,
        channel_id=f"api-{session_id}",
        message=request.message,
        conversation_history=conversation_history,
    )
    
    try:
        # Process the request
        response: AgentResponse = await _agent.process(context)
        
        # Update conversation history
        conversation_history.append({"role": "user", "content": request.message})
        conversation_history.append({"role": "assistant", "content": response.text})
        
        # Keep only last 20 messages to prevent memory issues
        if len(conversation_history) > 20:
            _sessions[session_id] = conversation_history[-20:]
        
        return ChatResponse(
            response=response.text,
            session_id=session_id,
            tool_calls=response.tool_calls,
            data=response.data,  # Pass UI data to frontend
            success=True,
            timestamp=datetime.now().isoformat(),
        )
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Check the health status of the API server."""
    global _agent, _start_time
    
    uptime = datetime.now() - _start_time
    uptime_str = str(uptime).split(".")[0]  # Remove microseconds
    
    return HealthResponse(
        status="healthy",
        agent_ready=_agent is not None,
        tools_count=len(_agent.tools) if _agent else 0,
        uptime=uptime_str,
    )


@app.get("/api/tools", response_model=ToolsResponse)
async def list_tools() -> ToolsResponse:
    """List all available tools that the agent can use."""
    global _agent
    
    if not _agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    tools = [
        ToolInfo(
            name=name,
            description=tool.description,
            parameters=tool.parameters,
        )
        for name, tool in _agent.tools.items()
    ]
    
    return ToolsResponse(
        tools=tools,
        count=len(tools),
    )


@app.delete("/api/sessions/{session_id}")
async def clear_session(session_id: str) -> dict[str, str]:
    """Clear a conversation session."""
    global _sessions
    
    if session_id in _sessions:
        del _sessions[session_id]
        return {"status": "cleared", "session_id": session_id}
    
    return {"status": "not_found", "session_id": session_id}


# =============================================================================
# Express Backend Bridge (super-fishstick compatibility)
# =============================================================================
# These endpoints match the format that the Express backend's ai.controller.js
# expects when proxying requests from the Next.js frontend.

class AskAgentRequest(BaseModel):
    """Request body matching Express backend's ai.controller.js format."""
    user_id: str | int = Field(default="api-user", description="User identifier from Express backend")
    message: str = Field(..., description="User message")


@app.post("/ask-agent")
async def ask_agent(request: AskAgentRequest) -> dict[str, Any]:
    """
    Plan-only endpoint: determines WHAT tools to use but does NOT execute them.
    
    Returns { message, plan } where plan items have status="pending".
    The frontend shows these as action cards for user approval.
    Actual execution happens via /approve-action.
    """
    global _agent
    
    if not _agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    # Set user_id in context so MCP servers can fetch DB tokens
    from gaprio.db_tokens import set_current_user_id
    set_current_user_id(str(request.user_id))
    
    # Create agent context
    context = AgentContext(
        user_id=str(request.user_id),
        channel_id=f"dashboard-{request.user_id}",
        message=request.message,
    )
    
    # Plan only — do NOT execute tools
    result = await _agent.plan_only(context)
    
    # Transform tool_calls into the "plan" format the frontend expects
    plan = []
    for i, tc in enumerate(result.tool_calls):
        plan.append({
            "id": f"action-{i}",
            "tool": tc.get("name", tc.get("tool", "unknown")),
            "provider": "gaprio",
            "parameters": tc.get("parameters", tc.get("arguments", tc.get("params", {}))),
            "status": tc.get("status", "pending"),
            "success": None,  # Not executed yet
            "result": None,   # No result yet
        })
    
    return {
        "message": result.text,
        "plan": plan,
        "data": result.data,
    }


class ApproveActionRequest(BaseModel):
    """Request to execute a specific tool action."""
    user_id: str | int = Field(default="api-user")
    tool: str = Field(..., description="Tool name to execute (e.g. 'asana_create_task')")
    parameters: dict[str, Any] = Field(default_factory=dict, description="Tool parameters")


@app.post("/approve-action")
async def approve_action(request: ApproveActionRequest) -> dict[str, Any]:
    """
    Execute a specific tool that was previously planned.
    
    Called when the user clicks 'Execute' on an action card.
    Receives the tool name and parameters, runs the tool, and returns results.
    """
    global _agent
    
    if not _agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    tool_name = request.tool
    params = request.parameters
    
    # Set user_id in context so MCP servers can fetch DB tokens
    from gaprio.db_tokens import set_current_user_id
    set_current_user_id(str(request.user_id))
    
    # Check if tool exists
    if tool_name not in _agent.tools:
        return {
            "success": False,
            "error": f"Unknown tool: {tool_name}",
            "result": None,
        }
    
    # Execute the tool
    try:
        tool = _agent.tools[tool_name]
        result = await tool.handler(**params)
        
        # Extract structured data from result
        result_data = None
        if hasattr(result, "data"):
            result_data = result.data
        elif isinstance(result, dict):
            result_data = result
        
        logger.info(f"Tool {tool_name} executed successfully via approve-action")
        
        return {
            "success": True,
            "error": None,
            "result": result_data,
        }
    except Exception as e:
        logger.error(f"Tool {tool_name} failed via approve-action: {e}")
        return {
            "success": False,
            "error": str(e),
            "result": None,
        }


# =============================================================================
# Monitoring & Predictive Actions
# =============================================================================

class AnalyzeContextRequest(BaseModel):
    """Request body for /analyze-context endpoint."""
    user_id: str | int = Field(..., description="User identifier")
    platform: str = Field(..., description="Source platform: slack, asana, google")
    channel_id: str = Field(default="", description="Channel/context identifier")
    context: str = Field(..., description="The message or event text to analyze")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional context")


class ExecuteActionRequest(BaseModel):
    """Request body for /execute-action endpoint."""
    user_id: str | int = Field(..., description="User identifier")
    tool: str = Field(..., description="Tool name to execute")
    parameters: dict[str, Any] = Field(default_factory=dict, description="Tool parameters")


@app.post("/analyze-context")
async def analyze_context(request: AnalyzeContextRequest) -> dict[str, Any]:
    """
    Analyze platform context and suggest actions WITHOUT executing them.
    
    Uses the agent's LLM with a monitoring-specific prompt to generate
    tool call suggestions. The suggestions are returned as structured data
    for the frontend to display in the Suggested Actions panel.
    """
    global _agent
    
    if not _agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    import json
    import re
    from gaprio.agent.prompts import MONITORING_PROMPT
    from gaprio.db_tokens import set_current_user_id
    
    try:
        # Set user context for DB token bridge
        set_current_user_id(str(request.user_id))
        
        # Build tools description for the prompt
        tools_desc = "\n".join([
            f"- **{name}**: {tool.description}\n  Parameters: {tool.parameters}"
            for name, tool in _agent.tools.items()
        ])
        
        # Build the monitoring prompt
        prompt = MONITORING_PROMPT.format(
            tools_description=tools_desc,
            platform=request.platform,
            channel=request.channel_id or "unknown",
            context=request.context[:1000],  # Truncate very long contexts
        )
        
        # Call LLM for analysis (suggest-only, no tool execution)
        raw_response = await _agent.llm.generate(prompt)
        
        logger.info(f"📡 [Monitor] LLM raw response ({len(raw_response)} chars): {raw_response[:500]}")
        
        # Strip markdown code blocks if present (```json ... ```)
        cleaned = raw_response.strip()
        cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r'```\s*$', '', cleaned, flags=re.MULTILINE)
        cleaned = cleaned.strip()
        
        # Extract JSON object from the response
        json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group())
                suggestions = parsed.get("suggestions", [])
            except json.JSONDecodeError as jde:
                logger.warning(f"📡 [Monitor] JSON parse error: {jde}. Extracted text: {json_match.group()[:300]}")
                suggestions = []
        else:
            logger.warning(f"📡 [Monitor] No JSON object found in LLM response: {cleaned[:300]}")
            suggestions = []
        
        # Validate that suggested tools actually exist
        valid_suggestions = []
        for s in suggestions:
            tool_name = s.get("tool", "")
            if tool_name in _agent.tools:
                valid_suggestions.append({
                    "tool": tool_name,
                    "params": s.get("params", s.get("parameters", {})),
                    "description": s.get("description", f"Execute {tool_name}")
                })
            else:
                logger.warning(f"📡 [Monitor] LLM suggested non-existent tool '{tool_name}'")
        
        logger.info(f"📡 [Monitor] Analysis complete: {len(valid_suggestions)} valid suggestion(s) from {request.platform}")
        return {"suggestions": valid_suggestions}
        
    except json.JSONDecodeError as jde:
        logger.warning(f"📡 [Monitor] Failed to parse LLM response as JSON: {jde}")
        return {"suggestions": []}
    except AttributeError as ae:
        logger.error(f"📡 [Monitor] Agent attribute error: {ae}", exc_info=True)
        return {"suggestions": []}
    except Exception as e:
        logger.error(f"📡 [Monitor] Analysis error: {e}", exc_info=True)
        return {"suggestions": []}


@app.post("/execute-action")
async def execute_action(request: ExecuteActionRequest) -> dict[str, Any]:
    """
    Execute a specific tool action (called when user approves a suggested action).
    
    This directly invokes the tool handler with the provided parameters,
    bypassing the full agent processing pipeline.
    """
    global _agent
    
    if not _agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    tool_name = request.tool
    if tool_name not in _agent.tools:
        raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found")
    
    try:
        from gaprio.db_tokens import set_current_user_id
        set_current_user_id(str(request.user_id))
        
        tool = _agent.tools[tool_name]
        params = dict(request.parameters)  # Make a mutable copy
        
        logger.info(f"📡 [Execute] Tool={tool_name}, Params={params}")
        
        # Normalize common LLM parameter name mismatches
        # Generic aliases (safe for all tools)
        generic_aliases = {
            "channel": "channel_id",
            "message": "text",
        }
        # Asana-specific aliases (only apply to asana_* tools)
        asana_aliases = {
            "task_name": "name",
            "title": "name",
            "task_title": "name",
            "summary": "name",
            "subject": "name",
            "description": "notes",
            "task_description": "notes",
            "body": "notes",
            "content": "notes",
            "due_date": "due_on",
            "deadline": "due_on",
            "start": "due_on",
            "start_date": "due_on",
            "date": "due_on",
        }
        
        for alias, canonical in generic_aliases.items():
            if alias in params and canonical not in params:
                params[canonical] = params.pop(alias)
        
        if tool_name.startswith("asana_"):
            for alias, canonical in asana_aliases.items():
                if alias in params and canonical not in params:
                    params[canonical] = params.pop(alias)
        
        # Fallback: if tool is asana_create_task and 'name' is still missing,
        # construct from any available text param
        if tool_name == "asana_create_task" and "name" not in params:
            for fallback_key in ["notes", "text", "message"]:
                if fallback_key in params:
                    params["name"] = params[fallback_key][:100]
                    break
            else:
                # Last resort: use the context/message that triggered the suggestion
                params["name"] = "Task from monitoring"
        
        logger.info(f"📡 [Execute] Normalized params={params}")
        
        result = await tool.handler(**params)
        
        return {
            "success": True,
            "tool": tool_name,
            "result": str(result) if result else "Action completed successfully"
        }
    except Exception as e:
        logger.error(f"Tool execution error ({tool_name}): {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Google Workspace OAuth2 Integration
# =============================================================================

GOOGLE_SCOPES = [
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/calendar",
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/documents",
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/contacts.readonly",
]


@app.get("/api/integrations/google/status")
async def google_status() -> dict[str, Any]:
    """Check whether Google Workspace is connected."""
    from gaprio.config import settings as _s
    connected = bool(_s.google_refresh_token and _s.google_client_id)
    return {
        "connected": connected,
        "client_id_set": bool(_s.google_client_id),
        "client_secret_set": bool(_s.google_client_secret),
        "refresh_token_set": bool(_s.google_refresh_token),
    }


@app.get("/api/integrations/google/auth-url")
async def google_auth_url() -> dict[str, str]:
    """Generate the Google OAuth2 consent URL."""
    from gaprio.config import settings as _s
    if not _s.google_client_id or not _s.google_client_secret:
        raise HTTPException(
            status_code=400,
            detail="Set GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET in .env first",
        )

    from urllib.parse import urlencode

    params = urlencode({
        "client_id": _s.google_client_id,
        "redirect_uri": _s.google_redirect_uri,
        "response_type": "code",
        "scope": " ".join(GOOGLE_SCOPES),
        "access_type": "offline",
        "prompt": "consent",
    })
    url = f"https://accounts.google.com/o/oauth2/v2/auth?{params}"
    return {"auth_url": url}


@app.get("/api/integrations/google/callback")
async def google_callback(code: str | None = None, error: str | None = None):
    """Handle the OAuth2 callback — exchange code for tokens."""
    if error:
        return HTMLResponse(
            f"<h2>Authorization failed</h2><p>{error}</p>", status_code=400
        )
    if not code:
        raise HTTPException(status_code=400, detail="Missing authorization code")

    from gaprio.config import settings as _s
    import httpx

    # Exchange the code for tokens
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "https://oauth2.googleapis.com/token",
            data={
                "code": code,
                "client_id": _s.google_client_id,
                "client_secret": _s.google_client_secret,
                "redirect_uri": _s.google_redirect_uri,
                "grant_type": "authorization_code",
            },
        )

    if resp.status_code != 200:
        logger.error(f"Google token exchange failed: {resp.text}")
        return HTMLResponse(
            f"<h2>Token exchange failed</h2><pre>{resp.text}</pre>", status_code=400
        )

    tokens = resp.json()
    refresh_token = tokens.get("refresh_token", "")

    if not refresh_token:
        return HTMLResponse(
            "<h2>No refresh token received</h2>"
            "<p>Try revoking access at "
            "<a href='https://myaccount.google.com/permissions'>Google Permissions</a> "
            "and re-authorizing.</p>",
            status_code=400,
        )

    # Persist to .env
    _update_env_value("GOOGLE_REFRESH_TOKEN", refresh_token)

    # Hot-reload into the running server
    _s.google_refresh_token = refresh_token
    _activate_google_server()

    logger.info("Google Workspace connected successfully")
    return HTMLResponse(
        "<html><body style='font-family:Inter,sans-serif;background:#0A0A0F;color:#fff;"
        "display:flex;align-items:center;justify-content:center;height:100vh'>"
        "<div style='text-align:center'>"
        "<h1 style='color:#F97316'>✓ Google Workspace Connected</h1>"
        "<p style='color:#9CA3AF'>You can close this tab and return to Gaprio.</p>"
        "</div></body></html>"
    )


@app.post("/api/integrations/google/disconnect")
async def google_disconnect() -> dict[str, str]:
    """Disconnect Google Workspace by clearing the refresh token."""
    from gaprio.config import settings as _s
    _update_env_value("GOOGLE_REFRESH_TOKEN", "")
    _s.google_refresh_token = ""
    logger.info("Google Workspace disconnected")
    return {"status": "disconnected"}


# -- helpers ------------------------------------------------------------------

def _update_env_value(key: str, value: str) -> None:
    """Update a single key in the .env file (preserving all other content)."""
    from pathlib import Path
    env_path = Path(".env")
    if not env_path.exists():
        env_path.write_text(f"{key}={value}\n", encoding="utf-8")
        return

    lines = env_path.read_text(encoding="utf-8").splitlines(keepends=True)
    found = False
    for i, line in enumerate(lines):
        if line.startswith(f"{key}="):
            lines[i] = f"{key}={value}\n"
            found = True
            break
    if not found:
        lines.append(f"{key}={value}\n")
    env_path.write_text("".join(lines), encoding="utf-8")


def _activate_google_server() -> None:
    """Register the Google MCP server or update its token if already active."""
    global _agent, _mcp_registry
    if not _mcp_registry or not _agent:
        return

    # RELOAD settings from .env to get the latest token
    # The global 'settings' object is cached, so we must re-instantiate
    from gaprio.config import Settings
    fresh_settings = Settings()

    # If already registered, update the refresh token
    existing = _mcp_registry.get_server("google")
    if existing:
        existing._refresh_token = fresh_settings.google_refresh_token
        existing._access_token = None  # Force re-fetch
        logger.info("Updated Google server refresh token via hot-reload")
        return

    from gaprio.mcp.google_server import GoogleMCPServer
    from gaprio.agent.agent import ToolDefinition

    # Create new server (uses stale settings by default)
    server = GoogleMCPServer()
    
    # FORCE UPDATE with fresh settings
    server._refresh_token = fresh_settings.google_refresh_token
    
    _mcp_registry.register(server)
    for tool in server.list_tools():
        _agent.register_tool(ToolDefinition(
            name=tool.name,
            description=tool.description,
            parameters=tool.parameters,
            handler=tool.handler,
        ))
    logger.info(f"Hot-loaded Google Workspace server with {len(server.tools)} tools")



# =============================================================================
# Simple Chat UI (served at root)
# =============================================================================

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the chat UI at root."""
    from pathlib import Path
    html_path = Path(__file__).parent / "chat_ui.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Gaprio API is running</h1><p>Chat UI not found.</p>")

