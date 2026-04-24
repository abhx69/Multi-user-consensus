"""
Asana MCP Server implementation.

Provides tools for interacting with Asana:
- Task management (create, list, update, complete)
- Project management (list, get)
- Comment management (add)

Uses OAuth 2.0 authentication flow.
"""

import logging
import re
from datetime import datetime, timedelta
import traceback
from typing import Any
import httpx

from gaprio.config import settings
from gaprio.mcp.base_server import BaseMCPServer, MCPTool, ToolResult

logger = logging.getLogger(__name__)


def _parse_natural_date(date_str: str) -> str | None:
    """
    Parse natural language date expressions into YYYY-MM-DD format.
    
    Handles:
    - "today", "today 9pm", "today at 5:00"
    - "tomorrow", "tomorrow morning"
    - "2026-02-09" (pass through)
    - "next monday", "next week"
    
    Returns:
        Date in YYYY-MM-DD format, or None if parsing fails.
    """
    if not date_str:
        return None
    
    date_str = date_str.strip().lower()
    today = datetime.now()
    
    # Already in YYYY-MM-DD format - pass through
    if re.match(r"^\d{4}-\d{2}-\d{2}$", date_str):
        return date_str
    
    # "today" (with optional time like "today 9pm")
    if date_str.startswith("today"):
        return today.strftime("%Y-%m-%d")
    
    # "tomorrow" (with optional time)
    if date_str.startswith("tomorrow"):
        return (today + timedelta(days=1)).strftime("%Y-%m-%d")
    
    # "next week" 
    if "next week" in date_str:
        return (today + timedelta(weeks=1)).strftime("%Y-%m-%d")
    
    # Handle day names like "next monday", "monday", etc.
    days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    for i, day in enumerate(days):
        if day in date_str:
            days_ahead = i - today.weekday()
            if days_ahead <= 0:  # Target day already happened this week
                days_ahead += 7
            return (today + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
    
    # Try to parse common date formats
    formats_to_try = [
        "%Y-%m-%d",
        "%m/%d/%Y",
        "%d/%m/%Y",
        "%B %d, %Y",  # e.g., "February 9, 2026"
        "%b %d, %Y",  # e.g., "Feb 9, 2026"
        "%d %B %Y",   # e.g., "9 February 2026"
        "%d %b %Y",   # e.g., "9 Feb 2026"
    ]
    
    # Strip time parts for parsing
    date_only = re.sub(r'\s+\d{1,2}(:\d{2})?\s*(am|pm)?.*$', '', date_str, flags=re.IGNORECASE)
    
    for fmt in formats_to_try:
        try:
            parsed = datetime.strptime(date_only, fmt)
            return parsed.strftime("%Y-%m-%d")
        except ValueError:
            continue
    
    # If all parsing fails, return None and let caller handle
    logger.warning(f"Could not parse date: {date_str}")
    return None


class AsanaMCPServer(BaseMCPServer):
    """
    MCP Server for Asana tools.
    
    Provides task and project management capabilities.
    Uses Asana's REST API with OAuth access token.
    """
    
    BASE_URL = "https://app.asana.com/api/1.0"
    
    def __init__(self):
        """Initialize the Asana MCP server."""
        self._access_token = settings.asana_access_token
        self._refresh_token = settings.asana_refresh_token
        self._client_id = settings.asana_client_id
        self._client_secret = settings.asana_client_secret
        self._default_workspace = settings.asana_default_workspace
        
        super().__init__("asana")
    
    def _get_headers(self) -> dict[str, str]:
        """Get headers with authorization token."""
        return {
            "Authorization": f"Bearer {self._access_token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
    
    async def _refresh_access_token(self) -> bool:
        """
        Refresh the access token using the refresh token.
        
        Token resolution order:
        1. DB tokens (when user_id is in context, from Express backend's MySQL)
        2. .env tokens (single-user fallback)
        
        Returns:
            True if successful, False otherwise
        """
        # --- Try DB tokens first (multi-user, via Express backend) ---
        from gaprio.db_tokens import get_current_user_id, get_connection_tokens, update_connection_tokens
        user_id = get_current_user_id()
        
        if user_id:
            tokens = await get_connection_tokens(user_id, "asana")
            if tokens and tokens.get("refresh_token"):
                try:
                    async with httpx.AsyncClient() as client:
                        response = await client.post(
                            "https://app.asana.com/-/oauth_token",
                            data={
                                "grant_type": "refresh_token",
                                "client_id": self._client_id,
                                "client_secret": self._client_secret,
                                "refresh_token": tokens["refresh_token"],
                            },
                        )
                        
                        if response.status_code == 200:
                            data = response.json()
                            self._access_token = data["access_token"]
                            new_refresh = data.get("refresh_token", tokens["refresh_token"])
                            if "refresh_token" in data:
                                self._refresh_token = data["refresh_token"]
                            # Write refreshed tokens back to DB
                            from datetime import datetime, timedelta
                            expires_at = datetime.now() + timedelta(seconds=data.get("expires_in", 3600))
                            await update_connection_tokens(
                                user_id, "asana",
                                access_token=self._access_token,
                                refresh_token=new_refresh,
                                expires_at=expires_at,
                            )
                            logger.info("Asana access token refreshed (from DB tokens)")
                            return True
                        else:
                            logger.warning(f"Asana token refresh failed (DB): {response.text}")
                except Exception as e:
                    logger.warning(f"Asana token refresh error (DB): {e}")
        
        # --- Fallback: .env tokens (single-user mode) ---
        if not self._refresh_token or not self._client_id or not self._client_secret:
            logger.warning("Cannot refresh token: missing OAuth credentials")
            return False
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://app.asana.com/-/oauth_token",
                    data={
                        "grant_type": "refresh_token",
                        "client_id": self._client_id,
                        "client_secret": self._client_secret,
                        "refresh_token": self._refresh_token,
                    },
                )
                
                if response.status_code == 200:
                    data = response.json()
                    self._access_token = data["access_token"]
                    if "refresh_token" in data:
                        self._refresh_token = data["refresh_token"]
                    logger.info("Asana access token refreshed")
                    return True
                else:
                    logger.error(f"Token refresh failed: {response.text}")
                    return False
                    
        except Exception as e:
            logger.error(f"Token refresh error: {e}")
            return False
    
    async def _call_api(
        self,
        method: str,
        endpoint: str,
        data: dict | None = None,
        params: dict | None = None,
    ) -> dict[str, Any]:
        """
        Make an authenticated API call to Asana.
        
        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            endpoint: API endpoint (e.g., "/tasks")
            data: Request body data
            params: Query parameters
            
        Returns:
            API response as dict
        """
        url = f"{self.BASE_URL}{endpoint}"
        
        async with httpx.AsyncClient() as client:
            response = await client.request(
                method,
                url,
                headers=self._get_headers(),
                json={"data": data} if data else None,
                params=params,
            )
            
            # Handle token expiry
            if response.status_code == 401:
                logger.info("Access token expired, attempting refresh")
                if await self._refresh_access_token():
                    # Retry with new token
                    response = await client.request(
                        method,
                        url,
                        headers=self._get_headers(),
                        json={"data": data} if data else None,
                        params=params,
                    )
            
            response.raise_for_status()
            return response.json()
    
    def _register_tools(self) -> None:
        """Register all Asana tools."""
        
        # Task: Create
        self.add_tool(MCPTool(
            name="asana_create_task",
            description="Create a new task in Asana with optional assignee and due date. Always extract assignee name and deadline from the user's request and pass them as parameters.",
            parameters={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Task name/title",
                    },
                    "project": {
                        "type": "string",
                        "description": "Project GID or name to add task to (e.g., 'Gaprio-Agent')",
                    },
                    "notes": {
                        "type": "string",
                        "description": "Task description/notes",
                    },
                    "due_on": {
                        "type": "string",
                        "description": "Due date in YYYY-MM-DD format. REQUIRED if user mentions a deadline.",
                    },
                    "assignee": {
                        "type": "string",
                        "description": "Assignee name or email. REQUIRED if user mentions 'assign to' someone.",
                    },
                },
                "required": ["name"],
            },
            handler=self._create_task,
        ))
        
        # Task: List
        self.add_tool(MCPTool(
            name="asana_list_tasks",
            description="List tasks from a project or workspace. Returns task names, status, and due dates.",
            parameters={
                "type": "object",
                "properties": {
                    "project": {
                        "type": "string",
                        "description": "Project GID to list tasks from",
                    },
                    "assignee": {
                        "type": "string",
                        "description": "Filter by assignee (email or 'me')",
                    },
                    "completed": {
                        "type": "boolean",
                        "description": "Include completed tasks (default: false)",
                    },
                },
            },
            handler=self._list_tasks,
        ))
        
        # Task: Update
        self.add_tool(MCPTool(
            name="asana_update_task",
            description="Update a task's properties (name, notes, due date, etc.)",
            parameters={
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": "Task GID to update",
                    },
                    "name": {
                        "type": "string",
                        "description": "New task name",
                    },
                    "notes": {
                        "type": "string",
                        "description": "New task notes",
                    },
                    "due_on": {
                        "type": "string",
                        "description": "New due date (YYYY-MM-DD)",
                    },
                    "completed": {
                        "type": "boolean",
                        "description": "Mark task completed/incomplete",
                    },
                },
                "required": ["task_id"],
            },
            handler=self._update_task,
        ))
        
        # Task: Complete
        self.add_tool(MCPTool(
            name="asana_complete_task",
            description="Mark a task as completed",
            parameters={
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": "Task GID to complete",
                    },
                },
                "required": ["task_id"],
            },
            handler=self._complete_task,
        ))
        
        # Project: List
        self.add_tool(MCPTool(
            name="asana_list_projects",
            description="List all projects in a workspace",
            parameters={
                "type": "object",
                "properties": {
                    "workspace": {
                        "type": "string",
                        "description": "Workspace GID (optional, uses default if not provided)",
                    },
                },
            },
            handler=self._list_projects,
        ))
        
        # Project: Get
        self.add_tool(MCPTool(
            name="asana_get_project",
            description="Get details of a specific project",
            parameters={
                "type": "object",
                "properties": {
                    "project_id": {
                        "type": "string",
                        "description": "Project GID",
                    },
                },
                "required": ["project_id"],
            },
            handler=self._get_project,
        ))
        
        # Task: Add Comment
        self.add_tool(MCPTool(
            name="asana_add_comment",
            description="Add a comment to a task",
            parameters={
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": "Task GID",
                    },
                    "text": {
                        "type": "string",
                        "description": "Comment text",
                    },
                },
                "required": ["task_id", "text"],
            },
            handler=self._add_comment,
        ))
        self.add_tool(MCPTool(
            name="asana_search_tasks",
            description=(
                "Search for existing Asana tasks by query. "
                "Use BEFORE creating a new task to check for duplicates."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query",
                    },
                    "project_id": {
                        "type": "string",
                        "description": "Optional project GID to scope the search",
                    },
                },
                "required": ["query"],
            },
            handler=self._search_tasks,
        ))
        
        self.add_tool(MCPTool(
            name="asana_get_project_status",
            description=(
                "Get an overall health snapshot of an Asana project — "
                "total tasks, completed, in-progress, and overdue count."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "project_id": {
                        "type": "string",
                        "description": "Project GID",
                    },
                },
                "required": ["project_id"],
            },
            handler=self._get_project_status,
        ))
    
    # =========================================================================
    # Tool Handlers
    # =========================================================================
    
    async def _create_task(
        self,
        name: str,
        project: str | None = None,
        notes: str = "",
        due_on: str | None = None,
        assignee: str | None = None,
        **kwargs,
    ) -> ToolResult:
        """
        Create a new task in Asana.
        
        Returns:
            ToolResult with task GID and URL
        """
        # Handle LLM hallucination of 'due_date' parameter
        if due_on is None and "due_date" in kwargs:
            due_on = kwargs["due_date"]
        if not self._access_token and not self._refresh_token:
            return ToolResult(
                success=False,
                error="Asana not configured. Set ASANA_ACCESS_TOKEN or ASANA_REFRESH_TOKEN in .env",
            )
        
        # Try to refresh if we don't have an access token
        if not self._access_token:
            if not await self._refresh_access_token():
                return ToolResult(
                    success=False,
                    error="Failed to refresh Asana access token",
                )
        
        try:
            # Resolve project if provided by name
            project_gid = None
            if project:
                project_gid = await self._resolve_project(project)
            
            # Build task data
            task_data: dict[str, Any] = {"name": name}
            
            if project_gid:
                task_data["projects"] = [project_gid]
            elif self._default_workspace:
                task_data["workspace"] = self._default_workspace
            
            if notes:
                task_data["notes"] = notes
            if due_on:
                # Parse natural language dates to YYYY-MM-DD format
                parsed_date = _parse_natural_date(due_on)
                if parsed_date:
                    task_data["due_on"] = parsed_date
                else:
                    # Return error for unparseable dates
                    return ToolResult(
                        success=False,
                        error=f"Could not parse due date: '{due_on}'. Please use YYYY-MM-DD format or natural language like 'today', 'tomorrow', 'next Monday'.",
                    )
            if assignee:
                resolved_assignee = await self._resolve_user(assignee, project_gid)
                if resolved_assignee:
                    task_data["assignee"] = resolved_assignee
                else:
                    # Return error for unresolvable assignee
                    return ToolResult(
                        success=False,
                        error=f"Could not find user '{assignee}' in the workspace. Please use their full name, email address, or Asana user GID.",
                    )
            
            # Create task
            response = await self._call_api("POST", "/tasks", data=task_data)
            task = response.get("data", {})
            
            task_gid = task.get("gid", "")
            task_url = task.get("permalink_url", f"https://app.asana.com/0/0/{task_gid}")
            
            logger.info(f"Created Asana task: {name} ({task_gid})")
            
            return ToolResult(
                success=True,
                data={
                    "gid": task_gid,
                    "url": task_url,
                    "name": name,
                    "message": f"Created task: {name}",
                },
            )
            
        except httpx.HTTPStatusError as e:
            error_msg = f"API error: {e.response.status_code}"
            try:
                error_data = e.response.json()
                error_msg = error_data.get("errors", [{}])[0].get("message", error_msg)
            except Exception:
                pass
            logger.error(f"Asana create task failed: {error_msg}")
            return ToolResult(success=False, error=error_msg)
            
        except Exception as e:
            logger.error(f"Asana create task error: {e}")
            logger.error(traceback.format_exc())
            return ToolResult(success=False, error=str(e) or "Unknown error occurred (check logs)")
    
    async def _list_tasks(
        self,
        project: str | None = None,
        assignee: str | None = None,
        completed: bool = False,
        **kwargs,  # Accept any extra params to avoid errors from LLM
    ) -> ToolResult:
        """
        List tasks from a project or for an assignee.
        
        Returns:
            ToolResult with list of tasks
        """
        if not self._access_token and not self._refresh_token:
            return ToolResult(
                success=False,
                error="Asana not configured. Set ASANA_ACCESS_TOKEN or ASANA_REFRESH_TOKEN in .env",
            )
        
        # Try to refresh if we don't have an access token
        if not self._access_token:
            if not await self._refresh_access_token():
                return ToolResult(
                    success=False,
                    error="Failed to refresh Asana access token",
                )
        
        try:
            params: dict[str, Any] = {
                "opt_fields": "name,due_on,completed,assignee.name,permalink_url",
                "completed_since": "now" if not completed else None,
            }
            
            if project:
                project_gid = await self._resolve_project(project)
                if project_gid:
                    params["project"] = project_gid
            elif assignee:
                params["assignee"] = assignee
                params["workspace"] = self._default_workspace
            else:
                # Default: my tasks
                params["assignee"] = "me"
                params["workspace"] = self._default_workspace
            
            response = await self._call_api("GET", "/tasks", params=params)
            tasks = response.get("data", [])
            
            # Format tasks
            formatted_tasks = []
            for task in tasks:
                formatted_tasks.append({
                    "gid": task.get("gid"),
                    "name": task.get("name"),
                    "due_on": task.get("due_on"),
                    "completed": task.get("completed"),
                    "assignee": task.get("assignee", {}).get("name") if task.get("assignee") else None,
                    "url": task.get("permalink_url"),
                })
            
            return ToolResult(
                success=True,
                data={
                    "tasks": formatted_tasks,
                    "count": len(formatted_tasks),
                },
            )
            
        except Exception as e:
            logger.error(f"Asana list tasks error: {e}")
            return ToolResult(success=False, error=str(e))
    
    async def _update_task(
        self,
        task_id: str,
        name: str | None = None,
        notes: str | None = None,
        due_on: str | None = None,
        completed: bool | None = None,
        **kwargs,
    ) -> ToolResult:
        """
        Update a task's properties.
        
        Returns:
            ToolResult with updated task data
        """
        # Handle LLM hallucination of 'due_date' parameter
        if due_on is None and "due_date" in kwargs:
            due_on = kwargs["due_date"]
        if not self._access_token and not self._refresh_token:
            return ToolResult(
                success=False,
                error="Asana not configured. Set ASANA_ACCESS_TOKEN or ASANA_REFRESH_TOKEN in .env",
            )
        
        # Try to refresh if we don't have an access token
        if not self._access_token:
            if not await self._refresh_access_token():
                return ToolResult(
                    success=False,
                    error="Failed to refresh Asana access token",
                )
        
        try:
            update_data: dict[str, Any] = {}
            
            if name is not None:
                update_data["name"] = name
            if notes is not None:
                update_data["notes"] = notes
            if due_on is not None:
                update_data["due_on"] = due_on
            if completed is not None:
                update_data["completed"] = completed
            
            if not update_data:
                return ToolResult(
                    success=False,
                    error="No updates provided",
                )
            
            response = await self._call_api("PUT", f"/tasks/{task_id}", data=update_data)
            task = response.get("data", {})
            
            logger.info(f"Updated Asana task: {task_id}")
            
            return ToolResult(
                success=True,
                data={
                    "gid": task.get("gid"),
                    "name": task.get("name"),
                    "url": task.get("permalink_url"),
                    "message": f"Updated task: {task.get('name')}",
                },
            )
            
        except Exception as e:
            logger.error(f"Asana update task error: {e}")
            return ToolResult(success=False, error=str(e))
    
    async def _complete_task(self, task_id: str) -> ToolResult:
        """
        Mark a task as completed.
        
        Returns:
            ToolResult confirming completion
        """
        return await self._update_task(task_id, completed=True)
    
    async def _list_projects(self, workspace: str | None = None) -> ToolResult:
        """
        List all projects in a workspace.
        
        Returns:
            ToolResult with list of projects
        """
        if not self._access_token and not self._refresh_token:
            return ToolResult(
                success=False,
                error="Asana not configured. Set ASANA_ACCESS_TOKEN or ASANA_REFRESH_TOKEN in .env",
            )
        
        # Try to refresh if we don't have an access token
        if not self._access_token:
            if not await self._refresh_access_token():
                return ToolResult(
                    success=False,
                    error="Failed to refresh Asana access token",
                )
        
        try:
            workspace = workspace or self._default_workspace
            
            if not workspace:
                return ToolResult(
                    success=False,
                    error="No workspace specified and ASANA_DEFAULT_WORKSPACE not set",
                )
            
            response = await self._call_api(
                "GET",
                "/projects",
                params={
                    "workspace": workspace,
                    "opt_fields": "name,color,permalink_url",
                },
            )
            
            projects = response.get("data", [])
            
            formatted_projects = []
            for proj in projects:
                formatted_projects.append({
                    "gid": proj.get("gid"),
                    "name": proj.get("name"),
                    "url": proj.get("permalink_url"),
                })
            
            return ToolResult(
                success=True,
                data={
                    "projects": formatted_projects,
                    "count": len(formatted_projects),
                },
            )
            
        except Exception as e:
            logger.error(f"Asana list projects error: {e}")
            return ToolResult(success=False, error=str(e))
    
    async def _get_project(self, project_id: str) -> ToolResult:
        """
        Get details of a specific project.
        
        Returns:
            ToolResult with project details
        """
        if not self._access_token and not self._refresh_token:
            return ToolResult(
                success=False,
                error="Asana not configured. Set ASANA_ACCESS_TOKEN or ASANA_REFRESH_TOKEN in .env",
            )
        
        # Try to refresh if we don't have an access token
        if not self._access_token:
            if not await self._refresh_access_token():
                return ToolResult(
                    success=False,
                    error="Failed to refresh Asana access token",
                )
        
        try:
            response = await self._call_api(
                "GET",
                f"/projects/{project_id}",
                params={
                    "opt_fields": "name,notes,color,owner.name,created_at,permalink_url",
                },
            )
            
            project = response.get("data", {})
            
            return ToolResult(
                success=True,
                data={
                    "gid": project.get("gid"),
                    "name": project.get("name"),
                    "notes": project.get("notes"),
                    "owner": project.get("owner", {}).get("name") if project.get("owner") else None,
                    "created_at": project.get("created_at"),
                    "url": project.get("permalink_url"),
                },
            )
            
        except Exception as e:
            logger.error(f"Asana get project error: {e}")
            return ToolResult(success=False, error=str(e))
    
    async def _add_comment(self, task_id: str, text: str) -> ToolResult:
        """
        Add a comment to a task.
        
        Returns:
            ToolResult confirming the comment was added
        """
        if not self._access_token and not self._refresh_token:
            return ToolResult(
                success=False,
                error="Asana not configured. Set ASANA_ACCESS_TOKEN or ASANA_REFRESH_TOKEN in .env",
            )
        
        # Try to refresh if we don't have an access token
        if not self._access_token:
            if not await self._refresh_access_token():
                return ToolResult(
                    success=False,
                    error="Failed to refresh Asana access token",
                )
        
        try:
            response = await self._call_api(
                "POST",
                f"/tasks/{task_id}/stories",
                data={"text": text},
            )
            
            story = response.get("data", {})
            
            logger.info(f"Added comment to task {task_id}")
            
            return ToolResult(
                success=True,
                data={
                    "gid": story.get("gid"),
                    "task_id": task_id,
                    "message": "Comment added successfully",
                },
            )
            
        except Exception as e:
            logger.error(f"Asana add comment error: {e}")
            return ToolResult(success=False, error=str(e))
    
    # =========================================================================
    # Helper Methods
    # =========================================================================
    
    async def _resolve_project(self, project: str) -> str | None:
        """
        Resolve a project name or GID to a GID.
        
        If the input looks like a GID (numeric), return it directly.
        Otherwise, search for a project by name.
        """
        # If it looks like a GID, return directly
        if project.isdigit():
            return project
        
        try:
            # Search for project by name
            response = await self._call_api(
                "GET",
                "/projects",
                params={
                    "workspace": self._default_workspace,
                    "opt_fields": "name",
                },
            )
            
            projects = response.get("data", [])
            
            # Case-insensitive match
            project_lower = project.lower()
            for proj in projects:
                if proj.get("name", "").lower() == project_lower:
                    return proj.get("gid")
            
            # Partial match
            for proj in projects:
                if project_lower in proj.get("name", "").lower():
                    return proj.get("gid")
            
            logger.warning(f"Project not found: {project}")
            return None
            
        except Exception as e:
            logger.error(f"Error resolving project: {e}")
            return None
    
    async def _resolve_user(self, user: str, project_gid: str | None = None) -> str | None:
        """
        Resolve a user name or email to a user GID.
        
        Search order:
        1. If 'me', return 'me'
        2. If looks like email or GID, return directly
        3. Search workspace users by name
        4. Search project members if project_gid provided
        5. Search all projects for members matching name
        """
        # If it's "me", return directly
        if user.lower() == "me":
            return "me"
        
        # If it looks like an email, return directly
        if "@" in user:
            return user
        
        # If it looks like a GID (numeric), return directly
        if user.isdigit():
            return user
        
        # Normalize: lowercase, remove spaces, underscores, and common separators
        user_lower = user.lower().replace(" ", "").replace("_", "").replace("-", "").replace(".", "")
        
        try:
            # 1. Search for users in workspace
            response = await self._call_api(
                "GET",
                f"/workspaces/{self._default_workspace}/users",
                params={
                    "opt_fields": "name,email",
                },
            )
            
            users = response.get("data", [])
            
            # Case-insensitive name match
            for u in users:
                name = u.get("name", "").lower()
                email = u.get("email", "").lower()
                
                # Normalize name for comparison
                name_normalized = name.replace(" ", "").replace("_", "").replace("-", "").replace(".", "")
                
                # Match by name (exact, contains, or normalized)
                if name and (name == user_lower or user_lower in name or name in user_lower or name_normalized == user_lower or user_lower in name_normalized or name_normalized in user_lower):
                    logger.info(f"Resolved user '{user}' to GID {u.get('gid')} (workspace user)")
                    return u.get("gid")
                
                # Match by email prefix (for users without display names)
                if email:
                    email_prefix = email.split("@")[0].replace(".", "").lower()
                    if user_lower in email_prefix or email_prefix in user_lower:
                        logger.info(f"Resolved user '{user}' to GID {u.get('gid')} via email {email}")
                        return u.get("gid")
            
            # 2. Search project members if we have a default project
            project_members = await self._get_project_members()
            for member in project_members:
                name = member.get("name", "").lower()
                email = member.get("email", "").lower()
                
                # Normalize name for comparison
                name_normalized = name.replace(" ", "").replace("_", "").replace("-", "").replace(".", "")
                
                # Match by name (exact, contains, or normalized)
                if name and (name == user_lower or user_lower in name or name in user_lower or name_normalized == user_lower or user_lower in name_normalized or name_normalized in user_lower):
                    logger.info(f"Resolved user '{user}' to GID {member.get('gid')} (project member)")
                    return member.get("gid")
                
                # Match by email prefix
                if email:
                    email_prefix = email.split("@")[0].replace(".", "").lower()
                    if user_lower in email_prefix or email_prefix in user_lower:
                        logger.info(f"Resolved user '{user}' to GID {member.get('gid')} via email {email}")
                        return member.get("gid")
            
            logger.warning(f"User not found: {user}")
            return None
            
        except Exception as e:
            logger.error(f"Error resolving user: {e}")
            return None
    
    async def _get_project_members(self) -> list[dict]:
        """Get all members from the default project."""
        try:
            # First, get projects in the workspace
            response = await self._call_api(
                "GET",
                "/projects",
                params={
                    "workspace": self._default_workspace,
                    "opt_fields": "name,members.name,members.email",
                    "limit": 10,
                },
            )
            
            all_members = []
            seen_gids = set()
            
            for project in response.get("data", []):
                members = project.get("members", [])
                for member in members:
                    gid = member.get("gid")
                    if gid and gid not in seen_gids:
                        seen_gids.add(gid)
                        all_members.append(member)
            
            logger.info(f"Found {len(all_members)} unique members across projects")
            return all_members
            
        except Exception as e:
            logger.error(f"Error getting project members: {e}")
            return []
    async def _search_tasks(
        self,
        query: str,
        project_id: str = "",
    ) -> ToolResult:
        """
        Search for tasks matching a query string.
        
        Uses Asana's typeahead API for fast, fuzzy matching.
        Use this BEFORE creating a new task to avoid duplicates.
        
        Args:
            query: Search query
            project_id: Optional project GID to scope search
            
        Returns:
            ToolResult with matching tasks
        """
        if not self._access_token and not self._refresh_token:
            return ToolResult(
                success=False,
                error="Asana not configured. Set ASANA_ACCESS_TOKEN or ASANA_REFRESH_TOKEN in .env",
            )
        
        if not self._access_token:
            if not await self._refresh_access_token():
                return ToolResult(
                    success=False,
                    error="Failed to refresh Asana access token",
                )
        
        try:
            workspace = self._default_workspace
            if not workspace:
                return ToolResult(
                    success=False,
                    error="No workspace configured. Set ASANA_DEFAULT_WORKSPACE in .env",
                )
            
            params = {
                "query": query,
                "type": "task",
                "opt_fields": "name,completed,due_on,assignee.name,permalink_url",
            }
            
            if project_id:
                # Use project-scoped task search
                response = await self._call_api(
                    "GET",
                    f"/workspaces/{workspace}/typeahead",
                    params=params,
                )
            else:
                response = await self._call_api(
                    "GET",
                    f"/workspaces/{workspace}/typeahead",
                    params=params,
                )
            
            tasks = response.get("data", [])
            
            results = []
            for task in tasks[:10]:
                results.append({
                    "gid": task.get("gid"),
                    "name": task.get("name"),
                    "completed": task.get("completed"),
                    "due_on": task.get("due_on"),
                    "assignee": (
                        task.get("assignee", {}).get("name")
                        if task.get("assignee")
                        else None
                    ),
                    "url": task.get("permalink_url"),
                })
            
            return ToolResult(
                success=True,
                data={"tasks": results, "count": len(results)},
            )
            
        except Exception as e:
            logger.error(f"Asana search tasks error: {e}")
            return ToolResult(success=False, error=str(e))
    
    async def _get_project_status(
        self,
        project_id: str,
    ) -> ToolResult:
        """
        Get project health snapshot — total tasks, completed, overdue.
        
        Fetches all tasks in a project and calculates completion metrics.
        Useful for "how is Project X doing?" queries.
        
        Args:
            project_id: Project GID
            
        Returns:
            ToolResult with project health metrics
        """
        if not self._access_token and not self._refresh_token:
            return ToolResult(
                success=False,
                error="Asana not configured. Set ASANA_ACCESS_TOKEN or ASANA_REFRESH_TOKEN in .env",
            )
        
        if not self._access_token:
            if not await self._refresh_access_token():
                return ToolResult(
                    success=False,
                    error="Failed to refresh Asana access token",
                )
        
        try:
            response = await self._call_api(
                "GET",
                f"/projects/{project_id}/tasks",
                params={"opt_fields": "completed,due_on,name"},
            )
            
            tasks = response.get("data", [])
            
            from datetime import date
            today = date.today().isoformat()
            
            total = len(tasks)
            completed = sum(1 for t in tasks if t.get("completed"))
            overdue = sum(
                1 for t in tasks
                if not t.get("completed")
                and t.get("due_on")
                and t["due_on"] < today
            )
            
            return ToolResult(
                success=True,
                data={
                    "project_id": project_id,
                    "total_tasks": total,
                    "completed": completed,
                    "in_progress": total - completed,
                    "overdue": overdue,
                    "completion_rate": (
                        f"{(completed / total * 100):.0f}%"
                        if total
                        else "N/A"
                    ),
                },
            )
            
        except Exception as e:
            logger.error(f"Asana project status error: {e}")
            return ToolResult(success=False, error=str(e))    
