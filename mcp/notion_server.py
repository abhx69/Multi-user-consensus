"""
Notion MCP Server.

Provides tools for Notion operations:
- Create pages
- Search pages
- Update page content
- Manage databases
"""

import logging
from typing import Any

from gaprio.config import settings
from gaprio.mcp.base_server import BaseMCPServer, MCPTool, ToolResult

logger = logging.getLogger(__name__)


class NotionMCPServer(BaseMCPServer):
    """
    MCP server for Notion operations.
    
    Provides tools for the agent to interact with Notion:
    - Creating summary pages from Slack discussions
    - Searching existing pages
    - Updating content
    
    Usage:
        server = NotionMCPServer()
        result = await server.execute(
            "notion_create_page",
            title="Meeting Summary",
            content="..."
        )
    """
    
    def __init__(self):
        """Initialize the Notion MCP server."""
        self._client = None
        super().__init__("notion")
    
    def _get_client(self):
        """Lazy initialization of Notion client."""
        if self._client is None:
            from notion_client import Client
            
            token = settings.notion_token
            if not token:
                raise RuntimeError("Notion token not configured")
            
            self._client = Client(auth=token)
        
        return self._client
    
    def _register_tools(self) -> None:
        """Register all Notion tools."""
        
        # Create page
        self.add_tool(MCPTool(
            name="notion_create_page",
            description=(
                "Create a new page in Notion. "
                "Use for summarizing Slack discussions, creating meeting notes, "
                "or creating tasks with due dates and assignees."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Page title",
                    },
                    "content": {
                        "type": "string",
                        "description": "Page content in markdown format",
                    },
                    "parent_id": {
                        "type": "string",
                        "description": "Parent page or database ID (optional, uses default)",
                    },
                    "icon": {
                        "type": "string",
                        "description": "Emoji icon for the page (optional)",
                    },
                    "due_date": {
                        "type": "string",
                        "description": "Due date in YYYY-MM-DD format (optional)",
                    },
                    "assignee": {
                        "type": "string",
                        "description": "Person to assign the task to (optional)",
                    },
                },
                "required": ["title", "content"],
            },
            handler=self._create_page,
        ))
        
        # Search pages
        self.add_tool(MCPTool(
            name="notion_search",
            description="Search for pages in Notion.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum results",
                        "default": 10,
                    },
                },
                "required": ["query"],
            },
            handler=self._search_pages,
        ))
        
        # Get page content
        self.add_tool(MCPTool(
            name="notion_get_page",
            description="Get the content of a Notion page.",
            parameters={
                "type": "object",
                "properties": {
                    "page_id": {
                        "type": "string",
                        "description": "Notion page ID",
                    },
                },
                "required": ["page_id"],
            },
            handler=self._get_page,
        ))
        
        # Append to page
        self.add_tool(MCPTool(
            name="notion_append_to_page",
            description="Append content to an existing Notion page.",
            parameters={
                "type": "object",
                "properties": {
                    "page_id": {
                        "type": "string",
                        "description": "Notion page ID",
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to append (markdown)",
                    },
                },
                "required": ["page_id", "content"],
            },
            handler=self._append_to_page,
        ))
        
        # List users
        self.add_tool(MCPTool(
            name="notion_list_users",
            description="List all users in the Notion workspace. Use this to find user IDs for assigning tasks.",
            parameters={
                "type": "object",
                "properties": {},
                "required": [],
            },
            handler=self._list_users,
        ))
        
        # List tasks from database
        self.add_tool(MCPTool(
            name="notion_list_tasks",
            description="List all tasks from the Notion database with their assignees and due dates. Use this to see all assigned tasks.",
            parameters={
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of tasks to return (default: 20)",
                        "default": 20,
                    },
                },
                "required": [],
            },
            handler=self._list_tasks,
        ))
    
    def _markdown_to_blocks(self, markdown: str) -> list[dict]:
        """
        Convert markdown to Notion blocks.
        
        Supports basic markdown:
        - Headers (#, ##, ###)
        - Bullet lists (-)
        - Numbered lists (1.)
        - Paragraphs
        - Bold (**text**)
        - Code blocks (```)
        """
        blocks = []
        lines = markdown.split("\n")
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Skip empty lines
            if not line:
                i += 1
                continue
            
            # Headers
            if line.startswith("### "):
                blocks.append({
                    "type": "heading_3",
                    "heading_3": {
                        "rich_text": [{"type": "text", "text": {"content": line[4:]}}]
                    }
                })
            elif line.startswith("## "):
                blocks.append({
                    "type": "heading_2",
                    "heading_2": {
                        "rich_text": [{"type": "text", "text": {"content": line[3:]}}]
                    }
                })
            elif line.startswith("# "):
                blocks.append({
                    "type": "heading_1",
                    "heading_1": {
                        "rich_text": [{"type": "text", "text": {"content": line[2:]}}]
                    }
                })
            
            # Bullet list
            elif line.startswith("- ") or line.startswith("* "):
                blocks.append({
                    "type": "bulleted_list_item",
                    "bulleted_list_item": {
                        "rich_text": [{"type": "text", "text": {"content": line[2:]}}]
                    }
                })
            
            # Numbered list
            elif line[0].isdigit() and ". " in line[:4]:
                content = line.split(". ", 1)[1] if ". " in line else line
                blocks.append({
                    "type": "numbered_list_item",
                    "numbered_list_item": {
                        "rich_text": [{"type": "text", "text": {"content": content}}]
                    }
                })
            
            # Code block
            elif line.startswith("```"):
                language = line[3:].strip() or "plain text"
                code_lines = []
                i += 1
                while i < len(lines) and not lines[i].strip().startswith("```"):
                    code_lines.append(lines[i])
                    i += 1
                
                blocks.append({
                    "type": "code",
                    "code": {
                        "rich_text": [{"type": "text", "text": {"content": "\n".join(code_lines)}}],
                        "language": language.lower(),
                    }
                })
            
            # Quote
            elif line.startswith("> "):
                blocks.append({
                    "type": "quote",
                    "quote": {
                        "rich_text": [{"type": "text", "text": {"content": line[2:]}}]
                    }
                })
            
            # Divider
            elif line in ["---", "***", "___"]:
                blocks.append({"type": "divider", "divider": {}})
            
            # Regular paragraph
            else:
                blocks.append({
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": [{"type": "text", "text": {"content": line}}]
                    }
                })
            
            i += 1
        
        return blocks
    
    async def _create_page(
        self,
        title: str,
        content: str,
        parent_id: str | None = None,
        icon: str | None = None,
        due_date: str | None = None,
        assignee: str | None = None,
    ) -> ToolResult:
        """
        Create a Notion page with smart property handling.
        
        Automatically detects database property names and maps:
        - Title: looks for 'title', 'Task', 'Name' properties
        - Date: looks for 'Date', 'Due Date', 'Deadline' properties
        - Assignee: looks for 'Assign', 'Assignee', 'Owner' properties
        
        Args:
            title: Page title
            content: Markdown content
            parent_id: Parent page/database ID
            icon: Optional emoji icon
            due_date: Optional due date in YYYY-MM-DD format
            assignee: Optional person to assign to
            
        Returns:
            ToolResult with page URL
        """
        try:
            client = self._get_client()
            
            # Determine parent database
            parent_id = parent_id or settings.notion_default_database_id
            
            if not parent_id:
                return ToolResult(
                    success=False,
                    error="No parent ID specified and NOTION_DEFAULT_DATABASE_ID not set",
                )
            
            # Convert content to blocks
            blocks = self._markdown_to_blocks(content)
            
            # Build page properties
            # Note: We use common property names directly since schema detection may fail
            # due to Notion API permissions. The property names must match your database.
            properties: dict[str, Any] = {}
            
            # Set the title property - try common names
            # Most databases use 'Task', 'Name', or 'Title' as the title property
            title_prop = "Task"  # Default for task databases (matches your database)
            properties[title_prop] = {
                "title": [{"type": "text", "text": {"content": title}}]
            }
            
            # Add due date if provided - try the exact property name from your database
            if due_date:
                # Try to set the "Date" property directly (matches your database)
                properties["Date"] = {"date": {"start": due_date}}
                logger.info(f"Set 'Date' property to {due_date}")
            
            # Add assignee if provided
            if assignee:
                # First, resolve the user name to a Notion user ID
                user_id = await self._resolve_user(assignee)
                
                if user_id:
                    # Set the "Assign" property directly (matches your database)
                    properties["Assign"] = {"people": [{"id": user_id}]}
                    logger.info(f"Set 'Assign' property to user {user_id} ({assignee})")
                else:
                    # Couldn't resolve user - add as callout instead
                    blocks.insert(0, {
                        "type": "callout",
                        "callout": {
                            "rich_text": [{"type": "text", "text": {"content": f"Assigned to: {assignee}"}}],
                            "icon": {"type": "emoji", "emoji": "👤"},
                        }
                    })
                    logger.warning(f"Could not resolve user '{assignee}', added callout instead")
            
            page_data: dict[str, Any] = {
                "parent": {"database_id": parent_id},
                "properties": properties,
                "children": blocks,
            }
            
            if icon:
                page_data["icon"] = {"type": "emoji", "emoji": icon}
            
            # Create the page
            response = client.pages.create(**page_data)
            
            page_url = response.get("url", "")
            page_id = response.get("id", "")
            
            logger.info(f"Created Notion page: {title}")
            
            extra_info = []
            if assignee:
                extra_info.append(f"assigned to {assignee}")
            if due_date:
                extra_info.append(f"due {due_date}")
            
            message = f"Created page: {title}"
            if extra_info:
                message += f" ({', '.join(extra_info)})"
            
            return ToolResult(
                success=True,
                data={
                    "id": page_id,
                    "url": page_url,
                    "title": title,
                    "assignee": assignee,
                    "due_date": due_date,
                    "message": message,
                },
            )
            
        except Exception as e:
            logger.error(f"Error creating page: {e}")
            return ToolResult(success=False, error=str(e))
    
    async def _get_database_schema(self, database_id: str) -> dict[str, Any]:
        """
        Query database schema to get property names and types.
        
        Returns a dict with:
        - properties: dict of property_name -> property_type
        - title_property: name of the title property
        """
        try:
            client = self._get_client()
            db = client.databases.retrieve(database_id)
            
            schema = {
                "properties": {},
                "title_property": "title",
            }
            
            for prop_name, prop_data in db.get("properties", {}).items():
                prop_type = prop_data.get("type", "")
                schema["properties"][prop_name] = prop_type
                
                # Track the title property
                if prop_type == "title":
                    schema["title_property"] = prop_name
            
            logger.info(f"Database schema: {schema['properties']}")
            return schema
            
        except Exception as e:
            logger.error(f"Error getting database schema: {e}")
            return {"properties": {}, "title_property": "title"}
    
    def _find_property_name(
        self,
        schema: dict[str, Any],
        candidates: list[str],
        expected_type: str,
    ) -> str | None:
        """
        Find a matching property name from the schema.
        
        Args:
            schema: Database schema from _get_database_schema
            candidates: List of property names to look for (in priority order)
            expected_type: Expected property type (e.g., 'date', 'people')
            
        Returns:
            The matching property name, or None if not found
        """
        props = schema.get("properties", {})
        
        # First, try exact matches (case-insensitive)
        for candidate in candidates:
            for prop_name, prop_type in props.items():
                if prop_name.lower() == candidate.lower() and prop_type == expected_type:
                    return prop_name
        
        # Then try partial matches
        for candidate in candidates:
            for prop_name, prop_type in props.items():
                if candidate.lower() in prop_name.lower() and prop_type == expected_type:
                    return prop_name
        
        # Finally, return any property of the expected type
        for prop_name, prop_type in props.items():
            if prop_type == expected_type:
                return prop_name
        
        return None
    
    async def _search_pages(
        self,
        query: str,
        limit: int = 10,
    ) -> ToolResult:
        """Search for Notion pages."""
        try:
            client = self._get_client()
            
            response = client.search(
                query=query,
                page_size=limit,
            )
            
            pages = []
            for result in response.get("results", []):
                if result.get("object") == "page":
                    # Get title from properties
                    title = "Untitled"
                    if "properties" in result:
                        title_prop = result["properties"].get("title", {})
                        if "title" in title_prop and title_prop["title"]:
                            title = title_prop["title"][0].get("plain_text", "Untitled")
                    
                    pages.append({
                        "id": result.get("id"),
                        "title": title,
                        "url": result.get("url"),
                        "created_time": result.get("created_time"),
                    })
            
            return ToolResult(
                success=True,
                data={
                    "pages": pages,
                    "count": len(pages),
                    "query": query,
                },
            )
            
        except Exception as e:
            logger.error(f"Error searching pages: {e}")
            return ToolResult(success=False, error=str(e))
    
    async def _get_page(self, page_id: str | None = None, **kwargs) -> ToolResult:
        """Get page content."""
        if not page_id:
            return ToolResult(
                success=False,
                error="page_id is required. Use notion_search first to find the page ID.",
            )
        try:
            client = self._get_client()
            
            # Get page metadata
            page = client.pages.retrieve(page_id)
            
            # Get page content (blocks)
            blocks_response = client.blocks.children.list(page_id)
            
            # Extract text content from blocks
            content_lines = []
            for block in blocks_response.get("results", []):
                block_type = block.get("type")
                block_data = block.get(block_type, {})
                
                if "rich_text" in block_data:
                    text = "".join(
                        t.get("plain_text", "")
                        for t in block_data["rich_text"]
                    )
                    content_lines.append(text)
            
            # Get title
            title = "Untitled"
            if "properties" in page:
                title_prop = page["properties"].get("title", {})
                if "title" in title_prop and title_prop["title"]:
                    title = title_prop["title"][0].get("plain_text", "Untitled")
            
            return ToolResult(
                success=True,
                data={
                    "id": page_id,
                    "title": title,
                    "url": page.get("url"),
                    "content": "\n".join(content_lines),
                },
            )
            
        except Exception as e:
            logger.error(f"Error getting page: {e}")
            return ToolResult(success=False, error=str(e))
    
    async def _append_to_page(
        self,
        page_id: str,
        content: str,
    ) -> ToolResult:
        """Append content to a page."""
        try:
            client = self._get_client()
            
            # Convert content to blocks
            blocks = self._markdown_to_blocks(content)
            
            # Append blocks
            client.blocks.children.append(
                block_id=page_id,
                children=blocks,
            )
            
            logger.info(f"Appended content to page {page_id}")
            
            return ToolResult(
                success=True,
                data={
                    "page_id": page_id,
                    "blocks_added": len(blocks),
                    "message": "Content appended successfully",
                },
            )
            
        except Exception as e:
            logger.error(f"Error appending to page: {e}")
            return ToolResult(success=False, error=str(e))
    
    async def _list_users(self) -> ToolResult:
        """List all users in the Notion workspace."""
        try:
            client = self._get_client()
            
            # Get all users
            response = client.users.list()
            
            users = []
            for user in response.get("results", []):
                if user.get("type") == "person":
                    users.append({
                        "id": user.get("id"),
                        "name": user.get("name"),
                        "email": user.get("person", {}).get("email"),
                        "avatar_url": user.get("avatar_url"),
                    })
            
            # Log for easy reference
            logger.info(f"Found {len(users)} Notion users")
            for u in users:
                logger.info(f"  - {u['name']}: {u['id']}")
            
            return ToolResult(
                success=True,
                data={
                    "users": users,
                    "count": len(users),
                    "message": f"Found {len(users)} users in workspace",
                },
            )
            
        except Exception as e:
            logger.error(f"Error listing users: {e}")
            return ToolResult(success=False, error=str(e))
    
    async def _resolve_user(self, name: str) -> str | None:
        """
        Resolve a user name to a Notion user ID.
        
        Search order:
        1. Custom user map from settings
        2. Workspace members (via users.list)
        3. Database guests (by scanning existing pages with Assign property)
        
        Returns the user ID if found, None otherwise.
        """
        if not name:
            return None
        
        name_lower = name.lower().strip()
        
        # 1. Check custom user map from settings first
        user_map = getattr(settings, 'notion_user_map', {})
        if user_map and name_lower in user_map:
            logger.info(f"Resolved '{name}' from user map")
            return user_map[name_lower]
        
        try:
            client = self._get_client()
            
            # 2. Check workspace members
            response = client.users.list()
            
            for user in response.get("results", []):
                if user.get("type") == "person":
                    user_name = user.get("name", "").lower()
                    # Exact or partial match
                    if name_lower == user_name or name_lower in user_name or user_name in name_lower:
                        logger.info(f"Resolved Notion user '{name}' to {user.get('id')} (workspace member)")
                        return user.get("id")
            
            # 3. Check database guests by searching existing pages
            db_id = settings.notion_default_database_id
            if db_id:
                guest_id = await self._find_guest_in_database(name, db_id)
                if guest_id:
                    return guest_id
            
            logger.warning(f"Could not resolve Notion user: {name}")
            return None
            
        except Exception as e:
            logger.error(f"Error resolving user: {e}")
            return None
    
    async def _find_guest_in_database(self, name: str, database_id: str) -> str | None:
        """
        Search for a guest user in existing database pages.
        
        Guests who have been assigned to pages in the database will appear
        in the 'Assign' property. This method scans those pages to find
        matching user IDs.
        
        Uses REST API directly since notion-client doesn't support databases.query.
        """
        import requests
        
        name_lower = name.lower().strip()
        
        try:
            # Use REST API directly
            headers = {
                'Authorization': f'Bearer {settings.notion_token}',
                'Notion-Version': '2022-06-28',
                'Content-Type': 'application/json'
            }
            
            response = requests.post(
                f'https://api.notion.com/v1/databases/{database_id}/query',
                headers=headers,
                json={}
            )
            
            if response.status_code != 200:
                logger.error(f"Database query failed: {response.status_code}")
                return None
            
            data = response.json()
            
            # Scan pages for users in the Assign property
            seen_users = {}  # name -> id
            for page in data.get("results", []):
                props = page.get("properties", {})
                
                # Check "Assign" property
                assign_prop = props.get("Assign", {})
                if assign_prop.get("type") == "people":
                    for person in assign_prop.get("people", []):
                        person_name = person.get("name", "")
                        person_id = person.get("id", "")
                        if person_name and person_id:
                            seen_users[person_name.lower()] = person_id
            
            logger.info(f"Found {len(seen_users)} users in database Assign property: {list(seen_users.keys())}")
            
            # Try to match the name
            for user_name, user_id in seen_users.items():
                if name_lower == user_name or name_lower in user_name or user_name in name_lower:
                    logger.info(f"Resolved '{name}' to {user_id} (found in database pages)")
                    return user_id
            
            logger.info(f"User '{name}' not found in database pages")
            return None
            
        except Exception as e:
            logger.error(f"Error finding guest in database: {e}")
            return None
    
    async def _list_tasks(self, limit: int = 20, **kwargs) -> ToolResult:
        """
        List tasks from the Notion database with assignees and due dates.
        
        Uses REST API to query the database directly.
        """
        import requests
        
        database_id = settings.notion_default_database_id
        if not database_id:
            return ToolResult(
                success=False,
                error="No default database configured. Set NOTION_DEFAULT_DATABASE_ID in .env",
            )
        
        try:
            headers = {
                "Authorization": f"Bearer {settings.notion_token}",
                "Notion-Version": "2022-06-28",
                "Content-Type": "application/json",
            }
            
            # Query the database
            url = f"https://api.notion.com/v1/databases/{database_id}/query"
            response = requests.post(
                url,
                headers=headers,
                json={"page_size": limit},
            )
            
            if response.status_code != 200:
                return ToolResult(
                    success=False,
                    error=f"Failed to query database: {response.status_code}",
                )
            
            data = response.json()
            tasks = []
            
            for page in data.get("results", []):
                props = page.get("properties", {})
                
                # Get task name/title
                task_name = "Untitled"
                for prop_name, prop_data in props.items():
                    if prop_data.get("type") == "title":
                        title_list = prop_data.get("title", [])
                        if title_list:
                            task_name = title_list[0].get("plain_text", "Untitled")
                        break
                
                # Get assignee
                assignee = None
                for prop_name in ["Assign", "Assignee", "Owner", "Person"]:
                    if prop_name in props:
                        people = props[prop_name].get("people", [])
                        if people:
                            assignee = people[0].get("name", "Unknown")
                        break
                
                # Get due date
                due_date = None
                for prop_name in ["Date", "Due Date", "Deadline", "Due"]:
                    if prop_name in props:
                        date_data = props[prop_name].get("date")
                        if date_data:
                            due_date = date_data.get("start")
                        break
                
                tasks.append({
                    "name": task_name,
                    "assignee": assignee,
                    "due_date": due_date,
                    "url": page.get("url"),
                })
            
            logger.info(f"Found {len(tasks)} tasks in Notion database")
            
            return ToolResult(
                success=True,
                data={
                    "tasks": tasks,
                    "count": len(tasks),
                    "message": f"Found {len(tasks)} tasks in Notion",
                },
            )
            
        except Exception as e:
            logger.error(f"Error listing tasks: {e}")
            return ToolResult(success=False, error=str(e))
