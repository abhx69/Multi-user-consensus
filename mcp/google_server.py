"""
Google Workspace MCP Server implementation.

Provides tools for interacting with Google Workspace:
- Gmail: Read, send, search emails
- Calendar: List, create, get events
- Drive: List, upload, download files
- Docs: List, read, create documents
- Sheets: Read, write, create spreadsheets
- Contacts: List, search contacts

Uses Personal OAuth 2.0 authentication with refresh tokens.
"""

import asyncio
import base64
import email
import logging
from email.message import EmailMessage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any

from gaprio.config import settings
from gaprio.mcp.base_server import BaseMCPServer, MCPTool, ToolResult

logger = logging.getLogger(__name__)


class GoogleMCPServer(BaseMCPServer):
    """
    MCP Server for Google Workspace tools.
    
    Provides Gmail, Calendar, and Drive capabilities.
    Uses Google's REST APIs with OAuth access token.
    """
    
    def __init__(self):
        """Initialize the Google MCP server."""
        self._client_id = settings.google_client_id
        self._client_secret = settings.google_client_secret
        self._refresh_token = settings.google_refresh_token
        self._access_token: str | None = None
        
        super().__init__("google")
    
    def _is_configured(self) -> bool:
        """Check if Google is configured (via .env refresh token OR client_id for DB tokens)."""
        return bool(self._refresh_token or self._client_id)
    
    async def _get_access_token(self) -> str | None:
        """
        Get a valid access token, refreshing if necessary.
        
        Token resolution order:
        1. DB tokens (when user_id is in context, from Express backend's MySQL)
        2. .env tokens (single-user fallback)
        
        Returns:
            Access token string, or None if not available
        """
        # --- Try DB tokens first (multi-user, via Express backend) ---
        from gaprio.db_tokens import get_current_user_id, get_connection_tokens, update_connection_tokens
        user_id = get_current_user_id()
        
        if user_id:
            tokens = await get_connection_tokens(user_id, "google")
            if tokens and tokens.get("refresh_token"):
                # Use DB-sourced refresh token
                client_id = self._client_id
                client_secret = self._client_secret
                refresh_token = tokens["refresh_token"]
                
                import httpx
                try:
                    async with httpx.AsyncClient() as client:
                        response = await client.post(
                            "https://oauth2.googleapis.com/token",
                            data={
                                "client_id": client_id,
                                "client_secret": client_secret,
                                "refresh_token": refresh_token,
                                "grant_type": "refresh_token",
                            },
                        )
                        
                        if response.status_code == 200:
                            data = response.json()
                            self._access_token = data["access_token"]
                            # Write refreshed token back to DB
                            from datetime import datetime, timedelta
                            expires_at = datetime.now() + timedelta(seconds=data.get("expires_in", 3600))
                            await update_connection_tokens(
                                user_id, "google",
                                access_token=self._access_token,
                                refresh_token=data.get("refresh_token", refresh_token),
                                expires_at=expires_at,
                            )
                            logger.debug("Google access token refreshed (from DB tokens)")
                            return self._access_token
                        else:
                            logger.warning(f"Google token refresh failed (DB): {response.text}")
                except Exception as e:
                    logger.warning(f"Google token refresh error (DB): {e}")
        
        # --- Fallback: .env tokens (single-user mode) ---
        if not self._refresh_token or not self._client_id or not self._client_secret:
            logger.warning("Google OAuth not configured")
            return None
        
        # Always refresh to get a valid token
        import httpx
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://oauth2.googleapis.com/token",
                    data={
                        "client_id": self._client_id,
                        "client_secret": self._client_secret,
                        "refresh_token": self._refresh_token,
                        "grant_type": "refresh_token",
                    },
                )
                
                if response.status_code == 200:
                    data = response.json()
                    self._access_token = data["access_token"]
                    logger.debug("Google access token refreshed")
                    return self._access_token
                else:
                    logger.error(f"Google token refresh failed: {response.text}")
                    return None
                    
        except Exception as e:
            logger.error(f"Google token refresh error: {e}")
            return None
    
    def _get_headers(self) -> dict[str, str]:
        """Get headers with authorization token."""
        return {
            "Authorization": f"Bearer {self._access_token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
    
    async def _call_api(
        self,
        method: str,
        url: str,
        data: dict | None = None,
        params: dict | None = None,
    ) -> dict[str, Any]:
        """
        Make an authenticated API call to Google.
        
        Args:
            method: HTTP method
            url: Full API URL
            data: Request body
            params: Query parameters
            
        Returns:
            API response as dict
        """
        import httpx
        
        # Ensure we have a valid token
        if not self._access_token:
            await self._get_access_token()
        
        if not self._access_token:
            raise RuntimeError("Failed to get Google access token")
        
        async with httpx.AsyncClient() as client:
            response = await client.request(
                method,
                url,
                headers=self._get_headers(),
                json=data,
                params=params,
            )
            
            # Handle token expiry
            if response.status_code == 401:
                logger.info("Google token expired, refreshing")
                await self._get_access_token()
                response = await client.request(
                    method,
                    url,
                    headers=self._get_headers(),
                    json=data,
                    params=params,
                )
            
            response.raise_for_status()
            
            if response.content:
                return response.json()
            return {}
    
    def _register_tools(self) -> None:
        """Register all Google Workspace tools."""
        
        # =====================================================================
        # Gmail Tools
        # =====================================================================
        
        self.add_tool(MCPTool(
            name="google_list_emails",
            description="List emails from Gmail. Returns subject, sender, date, and snippet.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Gmail search query (e.g., 'is:unread', 'from:boss@company.com')",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of emails to return (default: 10)",
                    },
                },
            },
            handler=self._list_emails,
        ))
        
        self.add_tool(MCPTool(
            name="google_read_email",
            description="Read the full content of a specific email",
            parameters={
                "type": "object",
                "properties": {
                    "email_id": {
                        "type": "string",
                        "description": "Email message ID",
                    },
                },
                "required": ["email_id"],
            },
            handler=self._read_email,
        ))
        
        self.add_tool(MCPTool(
            name="google_send_email",
            description="Send an email via Gmail",
            parameters={
                "type": "object",
                "properties": {
                    "to": {
                        "type": "string",
                        "description": "Recipient email address",
                    },
                    "subject": {
                        "type": "string",
                        "description": "Email subject",
                    },
                    "body": {
                        "type": "string",
                        "description": "Email body (plain text)",
                    },
                    "cc": {
                        "type": "string",
                        "description": "CC recipients (comma-separated)",
                    },
                },
                "required": ["to", "subject", "body"],
            },
            handler=self._send_email,
        ))
        
        self.add_tool(MCPTool(
            name="google_create_draft",
            description="Create a draft email in Gmail (for review before sending)",
            parameters={
                "type": "object",
                "properties": {
                    "to": {
                        "type": "string",
                        "description": "Recipient email address",
                    },
                    "subject": {
                        "type": "string",
                        "description": "Email subject",
                    },
                    "body": {
                        "type": "string",
                        "description": "Email body (plain text)",
                    },
                    "cc": {
                        "type": "string",
                        "description": "CC recipients (comma-separated)",
                    },
                },
                "required": ["to", "subject", "body"],
            },
            handler=self._create_draft,
        ))
        
        self.add_tool(MCPTool(
            name="google_send_draft",
            description="Send an existing draft by ID",
            parameters={
                "type": "object",
                "properties": {
                    "draft_id": {
                        "type": "string",
                        "description": "ID of the draft to send",
                    },
                },
                "required": ["draft_id"],
            },
            handler=self._send_draft,
        ))
        
        self.add_tool(MCPTool(
            name="google_update_draft",
            description="Update an existing draft (e.g. change subject, body, recipient)",
            parameters={
                "type": "object",
                "properties": {
                    "draft_id": {
                        "type": "string",
                        "description": "ID of the draft to update",
                    },
                    "to": {
                        "type": "string",
                        "description": "New recipient (optional)",
                    },
                    "subject": {
                        "type": "string",
                        "description": "New subject (optional)",
                    },
                    "body": {
                        "type": "string",
                        "description": "New body content (optional)",
                    },
                },
                "required": ["draft_id"],
            },
            handler=self._update_draft,
        ))
        
        # =====================================================================
        # Calendar Tools
        # =====================================================================
        
        self.add_tool(MCPTool(
            name="google_list_events",
            description="List upcoming calendar events",
            parameters={
                "type": "object",
                "properties": {
                    "time_min": {
                        "type": "string",
                        "description": "Start time in ISO format (default: now)",
                    },
                    "time_max": {
                        "type": "string",
                        "description": "End time in ISO format",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum events to return (default: 10)",
                    },
                },
            },
            handler=self._list_events,
        ))
        
        self.add_tool(MCPTool(
            name="google_create_event",
            description="Create a new calendar event",
            parameters={
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "Event title",
                    },
                    "start": {
                        "type": "string",
                        "description": "Start time in ISO format (e.g., 2025-02-10T14:00:00)",
                    },
                    "end": {
                        "type": "string",
                        "description": "End time in ISO format",
                    },
                    "description": {
                        "type": "string",
                        "description": "Event description",
                    },
                    "attendees": {
                        "type": "string",
                        "description": "Attendee emails (comma-separated)",
                    },
                    "location": {
                        "type": "string",
                        "description": "Event location",
                    },
                },
                "required": ["summary", "start", "end"],
            },
            handler=self._create_event,
        ))
        
        self.add_tool(MCPTool(
            name="google_get_event",
            description="Get details of a specific calendar event",
            parameters={
                "type": "object",
                "properties": {
                    "event_id": {
                        "type": "string",
                        "description": "Event ID",
                    },
                },
                "required": ["event_id"],
            },
            handler=self._get_event,
        ))
        
        # =====================================================================
        # Drive Tools
        # =====================================================================
        
        self.add_tool(MCPTool(
            name="google_list_files",
            description="List files from Google Drive",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Drive search query (e.g., \"name contains 'report'\")",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum files to return (default: 10)",
                    },
                },
            },
            handler=self._list_files,
        ))
        
        self.add_tool(MCPTool(
            name="google_upload_file",
            description="Upload a file to Google Drive",
            parameters={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "File name",
                    },
                    "content": {
                        "type": "string",
                        "description": "File content (text)",
                    },
                    "mime_type": {
                        "type": "string",
                        "description": "MIME type (default: text/plain)",
                    },
                    "folder_id": {
                        "type": "string",
                        "description": "Parent folder ID (optional)",
                    },
                },
                "required": ["name", "content"],
            },
            handler=self._upload_file,
        ))
        
        self.add_tool(MCPTool(
            name="google_get_file",
            description="Get metadata and download link for a file",
            parameters={
                "type": "object",
                "properties": {
                    "file_id": {
                        "type": "string",
                        "description": "File ID",
                    },
                },
                "required": ["file_id"],
            },
            handler=self._get_file,
        ))
        
        # =====================================================================
        # Google Docs Tools
        # =====================================================================
        
        self.add_tool(MCPTool(
            name="google_list_docs",
            description="List Google Docs documents from Drive",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query to filter documents by name",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum documents to return (default: 10)",
                    },
                },
            },
            handler=self._list_docs,
        ))
        
        self.add_tool(MCPTool(
            name="google_read_doc",
            description="Read the text content of a Google Docs document",
            parameters={
                "type": "object",
                "properties": {
                    "document_id": {
                        "type": "string",
                        "description": "Google Doc document ID",
                    },
                },
                "required": ["document_id"],
            },
            handler=self._read_doc,
        ))
        
        self.add_tool(MCPTool(
            name="google_create_doc",
            description="Create a new Google Docs document",
            parameters={
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Document title",
                    },
                    "body": {
                        "type": "string",
                        "description": "Initial text content for the document",
                    },
                },
                "required": ["title"],
            },
            handler=self._create_doc,
        ))
        
        # =====================================================================
        # Google Sheets Tools
        # =====================================================================
        
        self.add_tool(MCPTool(
            name="google_read_sheet",
            description="Read data from a Google Sheets spreadsheet",
            parameters={
                "type": "object",
                "properties": {
                    "spreadsheet_id": {
                        "type": "string",
                        "description": "Spreadsheet ID",
                    },
                    "range": {
                        "type": "string",
                        "description": "A1 notation range (e.g., 'Sheet1!A1:D10')",
                    },
                },
                "required": ["spreadsheet_id", "range"],
            },
            handler=self._read_sheet,
        ))
        
        self.add_tool(MCPTool(
            name="google_write_sheet",
            description="Write data to a Google Sheets spreadsheet",
            parameters={
                "type": "object",
                "properties": {
                    "spreadsheet_id": {
                        "type": "string",
                        "description": "Spreadsheet ID",
                    },
                    "range": {
                        "type": "string",
                        "description": "A1 notation range (e.g., 'Sheet1!A1')",
                    },
                    "values": {
                        "type": "string",
                        "description": "JSON-encoded 2D array of values, e.g. '[[\"Name\",\"Age\"],[\"Alice\",\"30\"]]'",
                    },
                },
                "required": ["spreadsheet_id", "range", "values"],
            },
            handler=self._write_sheet,
        ))
        
        self.add_tool(MCPTool(
            name="google_create_sheet",
            description="Create a new Google Sheets spreadsheet",
            parameters={
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Spreadsheet title",
                    },
                },
                "required": ["title"],
            },
            handler=self._create_sheet,
        ))
        
        # =====================================================================
        # Google Contacts Tools
        # =====================================================================
        
        self.add_tool(MCPTool(
            name="google_list_contacts",
            description="List contacts from Google Contacts",
            parameters={
                "type": "object",
                "properties": {
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum contacts to return (default: 10)",
                    },
                },
            },
            handler=self._list_contacts,
        ))
        
        self.add_tool(MCPTool(
            name="google_search_contacts",
            description="Search contacts by name or email",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (name or email)",
                    },
                },
                "required": ["query"],
            },
            handler=self._search_contacts,
        ))
        
        self.add_tool(MCPTool(
            name="google_search_emails",
            description="Search Gmail using a query string with optional date range filtering.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Gmail search query (e.g., 'from:sarah subject:bug')",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum emails to return (default: 10)",
                        "default": 10,
                    },
                },
                "required": ["query"],
            },
            handler=self._search_emails,
        ))
        
        self.add_tool(MCPTool(
            name="google_list_upcoming_deadlines",
            description="List calendar events in the next N days that contain deadline-related keywords like 'deadline', 'due', 'review', 'launch', 'release', 'submit', 'final'.",
            parameters={
                "type": "object",
                "properties": {
                    "days_ahead": {
                        "type": "integer",
                        "description": "How many days ahead to search (default: 7)",
                        "default": 7,
                    },
                },
            },
            handler=self._list_upcoming_deadlines,
        ))
    
    # =========================================================================
    # Gmail Handlers
    # =========================================================================
    
    async def _list_emails(
        self,
        query: str = "",
        max_results: int = 10,
    ) -> ToolResult:
        """List emails matching a query."""
        if not self._is_configured():
            return ToolResult(
                success=False,
                error="Google not configured. Set GOOGLE_REFRESH_TOKEN in .env",
            )
        
        try:
            # Search for messages
            params = {"maxResults": max_results}
            if query:
                params["q"] = query
            
            response = await self._call_api(
                "GET",
                "https://gmail.googleapis.com/gmail/v1/users/me/messages",
                params=params,
            )
            
            messages = response.get("messages", [])
            
            # Fetch details for each message
            emails = []
            for msg in messages[:max_results]:
                details = await self._call_api(
                    "GET",
                    f"https://gmail.googleapis.com/gmail/v1/users/me/messages/{msg['id']}",
                    params={"format": "metadata", "metadataHeaders": ["Subject", "From", "Date"]},
                )
                
                headers = {h["name"]: h["value"] for h in details.get("payload", {}).get("headers", [])}
                
                emails.append({
                    "id": msg["id"],
                    "subject": headers.get("Subject", "(no subject)"),
                    "from": headers.get("From", ""),
                    "date": headers.get("Date", ""),
                    "snippet": details.get("snippet", ""),
                })
            
            return ToolResult(
                success=True,
                data={"emails": emails, "count": len(emails)},
            )
            
        except Exception as e:
            logger.error(f"Gmail list error: {e}")
            return ToolResult(success=False, error=str(e))
    
    async def _read_email(self, email_id: str) -> ToolResult:
        """Read the full content of an email."""
        if not self._is_configured():
            return ToolResult(
                success=False,
                error="Google not configured. Set GOOGLE_REFRESH_TOKEN in .env",
            )
        
        try:
            response = await self._call_api(
                "GET",
                f"https://gmail.googleapis.com/gmail/v1/users/me/messages/{email_id}",
                params={"format": "full"},
            )
            
            # Extract headers
            headers = {h["name"]: h["value"] for h in response.get("payload", {}).get("headers", [])}
            
            # Extract body
            body = ""
            payload = response.get("payload", {})
            
            if "body" in payload and payload["body"].get("data"):
                body = base64.urlsafe_b64decode(payload["body"]["data"]).decode("utf-8", errors="ignore")
            elif "parts" in payload:
                for part in payload["parts"]:
                    if part.get("mimeType") == "text/plain" and part.get("body", {}).get("data"):
                        body = base64.urlsafe_b64decode(part["body"]["data"]).decode("utf-8", errors="ignore")
                        break
            
            return ToolResult(
                success=True,
                data={
                    "id": email_id,
                    "subject": headers.get("Subject", ""),
                    "from": headers.get("From", ""),
                    "to": headers.get("To", ""),
                    "date": headers.get("Date", ""),
                    "body": body[:2000] if body else "(no content)",  # Truncate long bodies
                },
            )
            
        except Exception as e:
            logger.error(f"Gmail read error: {e}")
            return ToolResult(success=False, error=str(e))
    
    async def _send_email(
        self,
        to: str,
        subject: str,
        body: str,
        cc: str = "",
    ) -> ToolResult:
        """Send an email via Gmail."""
        if not self._is_configured():
            return ToolResult(
                success=False,
                error="Google not configured. Set GOOGLE_REFRESH_TOKEN in .env",
            )
        
        try:
            # Create message
            message = MIMEMultipart()
            message["to"] = to
            message["subject"] = subject
            if cc:
                message["cc"] = cc
            
            message.attach(MIMEText(body, "plain"))
            
            # Encode
            raw = base64.urlsafe_b64encode(message.as_bytes()).decode("utf-8")
            
            # Send
            response = await self._call_api(
                "POST",
                "https://gmail.googleapis.com/gmail/v1/users/me/messages/send",
                data={"raw": raw},
            )
            
            msg_id = response.get("id", "")
            
            logger.info(f"Sent email to {to}: {subject}")
            
            return ToolResult(
                success=True,
                data={
                    "id": msg_id,
                    "to": to,
                    "subject": subject,
                    "message": f"Email sent to {to}",
                },
            )
            
        except Exception as e:
            logger.error(f"Gmail send error: {e}")
            return ToolResult(success=False, error=str(e))

    async def _create_draft(
        self,
        to: str,
        subject: str,
        body: str,
        cc: str = "",
    ) -> ToolResult:
        """Create a draft email in Gmail (for review before sending)."""
        if not self._is_configured():
            return ToolResult(success=False, error="Google not configured")
        
        try:
            message = MIMEMultipart()
            message["to"] = to
            message["subject"] = subject
            if cc:
                message["cc"] = cc
            message.attach(MIMEText(body, "plain"))
            raw = base64.urlsafe_b64encode(message.as_bytes()).decode("utf-8")
            
            response = await self._call_api(
                "POST",
                "https://gmail.googleapis.com/gmail/v1/users/me/drafts",
                data={"message": {"raw": raw}},
            )
            
            draft_id = response.get("id", "")
            msg_data = response.get("message", {})
            thread_id = msg_data.get("threadId", "")
            
            return ToolResult(
                success=True,
                data={
                    "id": draft_id,
                    "thread_id": thread_id,
                    "to": to,
                    "subject": subject,
                    "body": body,  # Include body for UI edit
                    "message": "Draft created successfully",
                    "url": f"https://mail.google.com/mail/u/0/#drafts/{msg_data.get('id')}",
                },
            )
        except Exception as e:
            logger.error(f"Gmail create draft error: {e}")
            return ToolResult(success=False, error=str(e))

    async def _send_draft(self, draft_id: str) -> ToolResult:
        """Send an existing draft by ID."""
        if not self._is_configured():
            return ToolResult(success=False, error="Google not configured")
        
        try:
            response = await self._call_api(
                "POST",
                "https://gmail.googleapis.com/gmail/v1/users/me/drafts/send",
                data={"id": draft_id},
            )
            
            msg_id = response.get("id", "")
            return ToolResult(
                success=True,
                data={
                    "id": msg_id,
                    "message": f"Draft {draft_id} sent successfully",
                },
            )
        except Exception as e:
            logger.error(f"Gmail send draft error: {e}")
            return ToolResult(success=False, error=str(e))
            
    async def _update_draft(
        self,
        draft_id: str,
        to: str | None = None,
        subject: str | None = None,
        body: str | None = None,
    ) -> ToolResult:
        """Update an existing draft."""
        if not self._is_configured():
            return ToolResult(success=False, error="Google not configured")
            
        try:
            # 1. Fetch current draft to get missing fields
            current_resp = await self._call_api(
                "GET",
                f"https://gmail.googleapis.com/gmail/v1/users/me/drafts/{draft_id}?format=raw"
            )
            
            raw_data = current_resp.get("message", {}).get("raw", "")
            if not raw_data:
                return ToolResult(success=False, error="Could not fetch existing draft content")
                
            msg_bytes = base64.urlsafe_b64decode(raw_data)
            current_msg = email.message_from_bytes(msg_bytes)
            
            # 2. Determine new values (use provided or fallback to existing)
            new_to = to if to is not None else current_msg.get("to", "")
            new_subject = subject if subject is not None else current_msg.get("subject", "")
            
            # Extract existing body if needed
            current_body = ""
            if body is None:
                if current_msg.is_multipart():
                    for part in current_msg.walk():
                        if part.get_content_type() == "text/plain":
                            current_body = part.get_payload(decode=True).decode("utf-8")
                            break
                else:
                    current_body = current_msg.get_payload(decode=True).decode("utf-8")
            
            new_body = body if body is not None else current_body
            
            # 3. Create new message
            message = MIMEMultipart()
            message["to"] = new_to
            message["subject"] = new_subject
            # message["from"] is set by Gmail API automatically
            
            message.attach(MIMEText(new_body, "plain"))
            
            raw = base64.urlsafe_b64encode(message.as_bytes()).decode("utf-8")
            
            # 4. Update via API
            response = await self._call_api(
                "PUT",
                f"https://gmail.googleapis.com/gmail/v1/users/me/drafts/{draft_id}",
                data={"message": {"raw": raw}},
            )
            
            msg_data = response.get("message", {})
            
            return ToolResult(
                success=True,
                data={
                    "id": response.get("id"),
                    "to": new_to,
                    "subject": new_subject,
                    "body": new_body, # Return body for UI update
                    "url": f"https://mail.google.com/mail/u/0/#drafts/{msg_data.get('id')}",
                    "message": "Draft updated successfully",
                },
            )
            
        except httpx.HTTPStatusError as e:
            logger.error(f"Gmail update draft API error: {e.response.text}")
            return ToolResult(success=False, error=f"Google API Error: {e.response.text}")
        except Exception as e:
            logger.error(f"Gmail update draft error: {e}")
            return ToolResult(success=False, error=str(e))
    
    # =========================================================================
    # Calendar Handlers
    # =========================================================================
    
    async def _list_events(
        self,
        time_min: str | None = None,
        time_max: str | None = None,
        max_results: int = 10,
    ) -> ToolResult:
        """List upcoming calendar events."""
        if not self._is_configured():
            return ToolResult(
                success=False,
                error="Google not configured. Set GOOGLE_REFRESH_TOKEN in .env",
            )
        
        try:
            from datetime import datetime, timezone
            
            def _ensure_tz(ts: str) -> str:
                """Ensure a timestamp has timezone info (Google requires RFC 3339)."""
                if "+" not in ts and "Z" not in ts and ts[-1] != "Z":
                    return ts + "Z"
                return ts
            
            params: dict[str, Any] = {
                "maxResults": max_results,
                "singleEvents": True,
                "orderBy": "startTime",
            }
            
            if time_min:
                params["timeMin"] = _ensure_tz(time_min)
            else:
                # Default to now
                params["timeMin"] = datetime.now(timezone.utc).isoformat()
            
            if time_max:
                params["timeMax"] = _ensure_tz(time_max)

            
            response = await self._call_api(
                "GET",
                "https://www.googleapis.com/calendar/v3/calendars/primary/events",
                params=params,
            )
            
            events = []
            for event in response.get("items", []):
                start = event.get("start", {})
                end = event.get("end", {})
                
                events.append({
                    "id": event.get("id"),
                    "summary": event.get("summary", "(no title)"),
                    "start": start.get("dateTime") or start.get("date"),
                    "end": end.get("dateTime") or end.get("date"),
                    "location": event.get("location"),
                    "url": event.get("htmlLink"),
                })
            
            return ToolResult(
                success=True,
                data={"events": events, "count": len(events)},
            )
            
        except Exception as e:
            logger.error(f"Calendar list error: {e}")
            return ToolResult(success=False, error=str(e))
    
    async def _create_event(
        self,
        summary: str,
        start: str,
        end: str,
        description: str = "",
        attendees: str = "",
        location: str = "",
    ) -> ToolResult:
        """Create a new calendar event."""
        if not self._is_configured():
            return ToolResult(
                success=False,
                error="Google not configured. Set GOOGLE_REFRESH_TOKEN in .env",
            )
        
        try:
            event_data: dict[str, Any] = {
                "summary": summary,
                "start": {"dateTime": start, "timeZone": "UTC"},
                "end": {"dateTime": end, "timeZone": "UTC"},
            }
            
            if description:
                event_data["description"] = description
            if location:
                event_data["location"] = location
            if attendees:
                event_data["attendees"] = [{"email": a.strip()} for a in attendees.split(",")]
            
            response = await self._call_api(
                "POST",
                "https://www.googleapis.com/calendar/v3/calendars/primary/events",
                data=event_data,
            )
            
            logger.info(f"Created calendar event: {summary}")
            
            return ToolResult(
                success=True,
                data={
                    "id": response.get("id"),
                    "summary": summary,
                    "url": response.get("htmlLink"),
                    "message": f"Created event: {summary}",
                },
            )
            
        except Exception as e:
            logger.error(f"Calendar create error: {e}")
            return ToolResult(success=False, error=str(e))
    
    async def _get_event(self, event_id: str) -> ToolResult:
        """Get details of a specific calendar event."""
        if not self._is_configured():
            return ToolResult(
                success=False,
                error="Google not configured. Set GOOGLE_REFRESH_TOKEN in .env",
            )
        
        try:
            response = await self._call_api(
                "GET",
                f"https://www.googleapis.com/calendar/v3/calendars/primary/events/{event_id}",
            )
            
            start = response.get("start", {})
            end = response.get("end", {})
            
            return ToolResult(
                success=True,
                data={
                    "id": response.get("id"),
                    "summary": response.get("summary"),
                    "description": response.get("description"),
                    "start": start.get("dateTime") or start.get("date"),
                    "end": end.get("dateTime") or end.get("date"),
                    "location": response.get("location"),
                    "attendees": [a.get("email") for a in response.get("attendees", [])],
                    "url": response.get("htmlLink"),
                },
            )
            
        except Exception as e:
            logger.error(f"Calendar get error: {e}")
            return ToolResult(success=False, error=str(e))
    
    # =========================================================================
    # Drive Handlers
    # =========================================================================
    
    async def _list_files(
        self,
        query: str = "",
        max_results: int = 10,
    ) -> ToolResult:
        """List files from Google Drive."""
        if not self._is_configured():
            return ToolResult(
                success=False,
                error="Google not configured. Set GOOGLE_REFRESH_TOKEN in .env",
            )
        
        try:
            params: dict[str, Any] = {
                "pageSize": max_results,
                "fields": "files(id,name,mimeType,createdTime,modifiedTime,webViewLink)",
            }
            
            if query:
                params["q"] = query
            
            response = await self._call_api(
                "GET",
                "https://www.googleapis.com/drive/v3/files",
                params=params,
            )
            
            files = []
            for f in response.get("files", []):
                files.append({
                    "id": f.get("id"),
                    "name": f.get("name"),
                    "mimeType": f.get("mimeType"),
                    "createdTime": f.get("createdTime"),
                    "modifiedTime": f.get("modifiedTime"),
                    "url": f.get("webViewLink"),
                })
            
            return ToolResult(
                success=True,
                data={"files": files, "count": len(files)},
            )
            
        except Exception as e:
            logger.error(f"Drive list error: {e}")
            return ToolResult(success=False, error=str(e))
    
    async def _upload_file(
        self,
        name: str,
        content: str,
        mime_type: str = "text/plain",
        folder_id: str | None = None,
    ) -> ToolResult:
        """Upload a file to Google Drive."""
        if not self._is_configured():
            return ToolResult(
                success=False,
                error="Google not configured. Set GOOGLE_REFRESH_TOKEN in .env",
            )
        
        try:
            import httpx
            
            # Ensure we have a token
            await self._get_access_token()
            
            # Multipart upload
            metadata: dict[str, Any] = {"name": name, "mimeType": mime_type}
            if folder_id:
                metadata["parents"] = [folder_id]
            
            # Use simple upload for small files
            boundary = "gaprio_upload_boundary"
            
            body = (
                f"--{boundary}\r\n"
                f'Content-Type: application/json; charset=UTF-8\r\n\r\n'
                f'{{"name": "{name}", "mimeType": "{mime_type}"}}\r\n'
                f"--{boundary}\r\n"
                f"Content-Type: {mime_type}\r\n\r\n"
                f"{content}\r\n"
                f"--{boundary}--"
            )
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://www.googleapis.com/upload/drive/v3/files?uploadType=multipart",
                    headers={
                        "Authorization": f"Bearer {self._access_token}",
                        "Content-Type": f"multipart/related; boundary={boundary}",
                    },
                    content=body.encode("utf-8"),
                )
                
                response.raise_for_status()
                data = response.json()
            
            logger.info(f"Uploaded file to Drive: {name}")
            
            return ToolResult(
                success=True,
                data={
                    "id": data.get("id"),
                    "name": name,
                    "url": f"https://drive.google.com/file/d/{data.get('id')}/view",
                    "message": f"Uploaded file: {name}",
                },
            )
            
        except Exception as e:
            logger.error(f"Drive upload error: {e}")
            return ToolResult(success=False, error=str(e))
    
    async def _get_file(self, file_id: str) -> ToolResult:
        """Get metadata and download link for a file."""
        if not self._is_configured():
            return ToolResult(
                success=False,
                error="Google not configured. Set GOOGLE_REFRESH_TOKEN in .env",
            )
        
        try:
            response = await self._call_api(
                "GET",
                f"https://www.googleapis.com/drive/v3/files/{file_id}",
                params={"fields": "id,name,mimeType,size,createdTime,modifiedTime,webViewLink,webContentLink"},
            )
            
            return ToolResult(
                success=True,
                data={
                    "id": response.get("id"),
                    "name": response.get("name"),
                    "mimeType": response.get("mimeType"),
                    "size": response.get("size"),
                    "createdTime": response.get("createdTime"),
                    "modifiedTime": response.get("modifiedTime"),
                    "viewUrl": response.get("webViewLink"),
                    "downloadUrl": response.get("webContentLink"),
                },
            )
            
        except Exception as e:
            logger.error(f"Drive get error: {e}")
            return ToolResult(success=False, error=str(e))
    
    # =========================================================================
    # Google Docs Handlers
    # =========================================================================
    
    async def _list_docs(
        self,
        query: str = "",
        max_results: int = 10,
    ) -> ToolResult:
        """List Google Docs documents."""
        if not self._is_configured():
            return ToolResult(success=False, error="Google not configured")
        
        try:
            params: dict[str, Any] = {
                "pageSize": max_results,
                "q": "mimeType='application/vnd.google-apps.document'",
                "fields": "files(id,name,createdTime,modifiedTime,webViewLink)",
            }
            if query:
                params["q"] += f" and name contains '{query}'"
            
            response = await self._call_api(
                "GET",
                "https://www.googleapis.com/drive/v3/files",
                params=params,
            )
            
            docs = [
                {
                    "id": f.get("id"),
                    "name": f.get("name"),
                    "createdTime": f.get("createdTime"),
                    "modifiedTime": f.get("modifiedTime"),
                    "url": f.get("webViewLink"),
                }
                for f in response.get("files", [])
            ]
            
            return ToolResult(success=True, data={"documents": docs, "count": len(docs)})
            
        except Exception as e:
            logger.error(f"Docs list error: {e}")
            return ToolResult(success=False, error=str(e))
    
    async def _read_doc(self, document_id: str) -> ToolResult:
        """Read text content of a Google Doc."""
        if not self._is_configured():
            return ToolResult(success=False, error="Google not configured")
        
        try:
            response = await self._call_api(
                "GET",
                f"https://docs.googleapis.com/v1/documents/{document_id}",
            )
            
            # Extract plain text from the document body
            text_parts: list[str] = []
            body = response.get("body", {})
            for element in body.get("content", []):
                paragraph = element.get("paragraph", {})
                for el in paragraph.get("elements", []):
                    text_run = el.get("textRun", {})
                    if text_run.get("content"):
                        text_parts.append(text_run["content"])
            
            content = "".join(text_parts)
            
            return ToolResult(
                success=True,
                data={
                    "id": document_id,
                    "title": response.get("title", ""),
                    "content": content[:5000],  # Truncate very long docs
                },
            )
            
        except Exception as e:
            logger.error(f"Docs read error: {e}")
            return ToolResult(success=False, error=str(e))
    
    async def _create_doc(
        self,
        title: str,
        body: str = "",
    ) -> ToolResult:
        """Create a new Google Doc."""
        if not self._is_configured():
            return ToolResult(success=False, error="Google not configured")
        
        try:
            # Create the document
            response = await self._call_api(
                "POST",
                "https://docs.googleapis.com/v1/documents",
                data={"title": title},
            )
            
            doc_id = response.get("documentId", "")
            
            # Insert body text if provided
            if body and doc_id:
                await self._call_api(
                    "POST",
                    f"https://docs.googleapis.com/v1/documents/{doc_id}:batchUpdate",
                    data={
                        "requests": [
                            {
                                "insertText": {
                                    "location": {"index": 1},
                                    "text": body,
                                }
                            }
                        ]
                    },
                )
            
            logger.info(f"Created Google Doc: {title}")
            
            return ToolResult(
                success=True,
                data={
                    "id": doc_id,
                    "title": title,
                    "url": f"https://docs.google.com/document/d/{doc_id}/edit",
                    "message": f"Created document: {title}",
                },
            )
            
        except Exception as e:
            logger.error(f"Docs create error: {e}")
            return ToolResult(success=False, error=str(e))
    
    # =========================================================================
    # Google Sheets Handlers
    # =========================================================================
    
    async def _read_sheet(
        self,
        spreadsheet_id: str,
        range: str,
    ) -> ToolResult:
        """Read data from a Google Sheets spreadsheet."""
        if not self._is_configured():
            return ToolResult(success=False, error="Google not configured")
        
        try:
            import urllib.parse
            encoded_range = urllib.parse.quote(range)
            
            response = await self._call_api(
                "GET",
                f"https://sheets.googleapis.com/v4/spreadsheets/{spreadsheet_id}/values/{encoded_range}",
            )
            
            values = response.get("values", [])
            
            return ToolResult(
                success=True,
                data={
                    "range": response.get("range", range),
                    "values": values,
                    "rows": len(values),
                },
            )
            
        except Exception as e:
            logger.error(f"Sheets read error: {e}")
            return ToolResult(success=False, error=str(e))
    
    async def _write_sheet(
        self,
        spreadsheet_id: str,
        range: str,
        values: str,
    ) -> ToolResult:
        """Write data to a Google Sheets spreadsheet."""
        if not self._is_configured():
            return ToolResult(success=False, error="Google not configured")
        
        try:
            import json as _json
            import urllib.parse
            
            parsed_values = _json.loads(values)
            encoded_range = urllib.parse.quote(range)
            
            response = await self._call_api(
                "PUT",
                f"https://sheets.googleapis.com/v4/spreadsheets/{spreadsheet_id}/values/{encoded_range}",
                data={
                    "range": range,
                    "majorDimension": "ROWS",
                    "values": parsed_values,
                },
                params={"valueInputOption": "USER_ENTERED"},
            )
            
            logger.info(f"Wrote to sheet {spreadsheet_id} range {range}")
            
            return ToolResult(
                success=True,
                data={
                    "updatedRange": response.get("updatedRange", range),
                    "updatedRows": response.get("updatedRows", 0),
                    "updatedCells": response.get("updatedCells", 0),
                    "message": f"Updated {response.get('updatedCells', 0)} cells",
                },
            )
            
        except _json.JSONDecodeError:
            return ToolResult(success=False, error="Invalid JSON in 'values' parameter")
        except Exception as e:
            logger.error(f"Sheets write error: {e}")
            return ToolResult(success=False, error=str(e))
    
    async def _create_sheet(self, title: str) -> ToolResult:
        """Create a new Google Sheets spreadsheet."""
        if not self._is_configured():
            return ToolResult(success=False, error="Google not configured")
        
        try:
            response = await self._call_api(
                "POST",
                "https://sheets.googleapis.com/v4/spreadsheets",
                data={"properties": {"title": title}},
            )
            
            sheet_id = response.get("spreadsheetId", "")
            
            logger.info(f"Created spreadsheet: {title}")
            
            return ToolResult(
                success=True,
                data={
                    "id": sheet_id,
                    "title": title,
                    "url": response.get("spreadsheetUrl", f"https://docs.google.com/spreadsheets/d/{sheet_id}/edit"),
                    "message": f"Created spreadsheet: {title}",
                },
            )
            
        except Exception as e:
            logger.error(f"Sheets create error: {e}")
            return ToolResult(success=False, error=str(e))
    
    # =========================================================================
    # Google Contacts Handlers
    # =========================================================================
    
    async def _list_contacts(
        self,
        max_results: int = 10,
    ) -> ToolResult:
        """List contacts from Google Contacts."""
        if not self._is_configured():
            return ToolResult(success=False, error="Google not configured")
        
        try:
            response = await self._call_api(
                "GET",
                "https://people.googleapis.com/v1/people/me/connections",
                params={
                    "pageSize": max_results,
                    "personFields": "names,emailAddresses,phoneNumbers,organizations",
                },
            )
            
            contacts = []
            for person in response.get("connections", []):
                names = person.get("names", [{}])
                emails = person.get("emailAddresses", [])
                phones = person.get("phoneNumbers", [])
                orgs = person.get("organizations", [])
                
                contacts.append({
                    "name": names[0].get("displayName", "") if names else "",
                    "emails": [e.get("value", "") for e in emails],
                    "phones": [p.get("value", "") for p in phones],
                    "organization": orgs[0].get("name", "") if orgs else "",
                })
            
            return ToolResult(
                success=True,
                data={"contacts": contacts, "count": len(contacts)},
            )
            
        except Exception as e:
            logger.error(f"Contacts list error: {e}")
            return ToolResult(success=False, error=str(e))
    
    async def _search_contacts(self, query: str) -> ToolResult:
        """Search contacts by name or email."""
        if not self._is_configured():
            return ToolResult(success=False, error="Google not configured")
        
        try:
            response = await self._call_api(
                "GET",
                "https://people.googleapis.com/v1/people:searchContacts",
                params={
                    "query": query,
                    "readMask": "names,emailAddresses,phoneNumbers,organizations",
                },
            )
            
            contacts = []
            for result in response.get("results", []):
                person = result.get("person", {})
                names = person.get("names", [{}])
                emails = person.get("emailAddresses", [])
                phones = person.get("phoneNumbers", [])
                orgs = person.get("organizations", [])
                
                contacts.append({
                    "name": names[0].get("displayName", "") if names else "",
                    "emails": [e.get("value", "") for e in emails],
                    "phones": [p.get("value", "") for p in phones],
                    "organization": orgs[0].get("name", "") if orgs else "",
                })
            
            return ToolResult(
                success=True,
                data={"contacts": contacts, "count": len(contacts)},
            )
            
        except Exception as e:
            logger.error(f"Contacts search error: {e}")
            return ToolResult(success=False, error=str(e))
        
    async def _search_emails(
        self,
        query: str,
        max_results: int = 10,
    ) -> ToolResult:
        """Search Gmail with a query string (e.g., 'from:sarah subject:bug')."""
        if not self._is_configured():
            return ToolResult(
                success=False,
                error="Google not configured. Set GOOGLE_REFRESH_TOKEN in .env",
            )
        
        try:
            response = await self._call_api(
                "GET",
                "https://gmail.googleapis.com/gmail/v1/users/me/messages",
                params={"q": query, "maxResults": max_results},
            )
            
            messages = response.get("messages", [])
            
            results = []
            for msg_ref in messages[:max_results]:
                msg_resp = await self._call_api(
                    "GET",
                    f"https://gmail.googleapis.com/gmail/v1/users/me/messages/{msg_ref['id']}",
                    params={
                        "format": "metadata",
                        "metadataHeaders": ["From", "Subject", "Date"],
                    },
                )
                
                hdrs = {
                    h["name"]: h["value"]
                    for h in msg_resp.get("payload", {}).get("headers", [])
                }
                results.append({
                    "id": msg_ref["id"],
                    "from": hdrs.get("From", ""),
                    "subject": hdrs.get("Subject", ""),
                    "date": hdrs.get("Date", ""),
                    "snippet": msg_resp.get("snippet", ""),
                })
            
            return ToolResult(
                success=True,
                data={"emails": results, "count": len(results)},
            )
            
        except Exception as e:
            logger.error(f"Gmail search error: {e}")
            return ToolResult(success=False, error=str(e))
    
    async def _list_upcoming_deadlines(
        self,
        days_ahead: int = 7,
    ) -> ToolResult:
        """Find calendar events with deadline keywords in the next N days."""
        if not self._is_configured():
            return ToolResult(
                success=False,
                error="Google not configured. Set GOOGLE_REFRESH_TOKEN in .env",
            )
        
        try:
            from datetime import datetime, timedelta, timezone
            
            now = datetime.now(timezone.utc).isoformat()
            future = (datetime.now(timezone.utc) + timedelta(days=days_ahead)).isoformat()
            
            response = await self._call_api(
                "GET",
                "https://www.googleapis.com/calendar/v3/calendars/primary/events",
                params={
                    "timeMin": now,
                    "timeMax": future,
                    "singleEvents": True,
                    "orderBy": "startTime",
                    "maxResults": 50,
                },
            )
            
            events = response.get("items", [])
            
            deadline_keywords = [
                "deadline", "due", "review", "launch",
                "release", "submit", "final",
            ]
            
            deadlines = []
            for event in events:
                title = event.get("summary", "").lower()
                desc = event.get("description", "").lower()
                if any(kw in title or kw in desc for kw in deadline_keywords):
                    deadlines.append({
                        "title": event.get("summary"),
                        "start": (
                            event.get("start", {}).get("dateTime")
                            or event.get("start", {}).get("date")
                        ),
                        "description": event.get("description", "")[:200],
                    })
            
            return ToolResult(
                success=True,
                data={"deadlines": deadlines, "count": len(deadlines)},
            )
            
        except Exception as e:
            logger.error(f"Calendar deadlines error: {e}")
            return ToolResult(success=False, error=str(e))    

