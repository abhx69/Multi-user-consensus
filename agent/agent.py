"""
Core Agent implementation for Gaprio.

The Agent is the central orchestrator that:
1. Receives user messages from Slack
2. Builds context from memory and RAG
3. Determines which tools to use (if any)
4. Executes tool calls via MCP servers
5. Generates and returns responses

Architecture:
    User Message -> Agent -> [Memory + RAG + Tools] -> Response
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from gaprio.agent.llm_provider import LLMProvider, get_llm_provider
from gaprio.agent.prompts import (
    SYSTEM_PROMPT,
    TOOL_SELECTION_PROMPT,
    RAG_CONTEXT_PROMPT,
)
from gaprio.memory.memory_manager import MemoryManager
from gaprio.rag.retriever import Retriever

logger = logging.getLogger(__name__)


@dataclass
class ToolDefinition:
    """
    Definition of a tool that the agent can use.
    
    Attributes:
        name: Unique identifier for the tool
        description: Human-readable description of what the tool does
        parameters: JSON Schema describing the tool's parameters
        handler: Async function that executes the tool
    """
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., Any]
    
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


@dataclass
class AgentContext:
    """
    Context passed to the agent for each request.
    
    This contains all the information the agent needs to process a request,
    including memory, RAG context, and conversation history.
    """
    user_id: str
    channel_id: str
    message: str
    conversation_history: list[dict[str, str]] = field(default_factory=list)
    memory_context: str = ""
    rag_context: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResponse:
    """
    Response from the agent.
    
    Contains the generated response text and optional metadata about
    tool calls, memory updates, and other actions taken.
    """
    text: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    memory_updates: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    data: dict[str, Any] | None = field(default=None)


class Agent:
    """
    The core Gaprio Agent.
    
    Orchestrates all components to process user requests:
    - LLM for intelligence
    - Memory for context and persistence
    - RAG for knowledge retrieval
    - Tools for actions
    
    Example:
        agent = Agent()
        agent.register_tool(slack_tool)
        
        context = AgentContext(
            user_id="U123",
            channel_id="C456",
            message="Summarize #general for today"
        )
        response = await agent.process(context)
    """
    
    def __init__(
        self,
        llm_provider: LLMProvider | None = None,
        memory_manager: MemoryManager | None = None,
        retriever: Retriever | None = None,
    ):
        """
        Initialize the Agent.
        
        Args:
            llm_provider: LLM backend (defaults to configured provider)
            memory_manager: Memory system (created if not provided)
            retriever: RAG retriever (created if not provided)
        """
        self.llm = llm_provider or get_llm_provider()
        self.memory = memory_manager or MemoryManager()
        self.retriever = retriever or Retriever()
        self.tools: dict[str, ToolDefinition] = {}
        
        logger.info(f"Agent initialized with {self.llm.__class__.__name__}")
    
    def register_tool(self, tool: ToolDefinition) -> None:
        """
        Register a tool that the agent can use.
        
        Args:
            tool: Tool definition to register
        """
        self.tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")
    
    def unregister_tool(self, name: str) -> None:
        """
        Unregister a tool by name.
        
        Args:
            name: Name of the tool to remove
        """
        if name in self.tools:
            del self.tools[name]
            logger.info(f"Unregistered tool: {name}")
    
    async def process(self, context: AgentContext) -> AgentResponse:
        """
        Process a user request and generate a response.
        
        This is the main entry point for the agent. It:
        1. Enriches context with memory and RAG
        2. Determines if tools are needed
        3. Executes tool calls (with intelligent result chaining)
        4. Generates the final response
        
        Args:
            context: The request context with user message and metadata
            
        Returns:
            AgentResponse with the generated text and metadata
        """
        logger.info(f"Processing request from {context.user_id}: {context.message[:50]}...")
        
        # Step 1: Enrich context with memory
        context.memory_context = await self._recall_memory(context)
        
        # Step 2: Check if RAG context is needed
        if await self._should_use_rag(context):
            context.rag_context = await self._retrieve_rag_context(context)
        
        # Step 3: Determine if tools are needed and which ones
        tool_plan = await self._plan_tools(context)
        
        # Step 4: Execute tool calls with result chaining
        tool_results = []
        if tool_plan.get("tools_needed", False):
            tool_calls = tool_plan.get("tool_calls", [])
            tool_results = await self._execute_tools_with_chaining(context, tool_calls)
        
        # Step 5: Generate final response
        response = await self._generate_response(context, tool_results)
        
        # Step 6: Update memory with any new information
        memory_updates = await self._update_memory(context, response)
        response.memory_updates = memory_updates
        
        return response
    
    async def plan_only(self, context: AgentContext) -> AgentResponse:
        """
        Plan tool calls WITHOUT executing them.
        
        Returns an AgentResponse with:
        - tool_calls: list of planned tools (with name + params) but NOT executed
        - text: conversational response if no tools needed
        
        The frontend will show these as pending action cards for user approval.
        """
        logger.info(f"Planning request from {context.user_id}: {context.message[:50]}...")
        
        # Step 1: Enrich context with memory
        context.memory_context = await self._recall_memory(context)
        
        # Step 2: Check if RAG context is needed
        if await self._should_use_rag(context):
            context.rag_context = await self._retrieve_rag_context(context)
        
        # Step 3: Determine if tools are needed and which ones
        tool_plan = await self._plan_tools(context)
        
        # Step 4: If tools are needed, return the plan without executing
        if tool_plan.get("tools_needed", False):
            tool_calls = tool_plan.get("tool_calls", [])
            
            # Build plan items with tool name and params
            planned = []
            for tc in tool_calls:
                tool_name = tc.get("tool", "unknown")
                params = tc.get("params", {})
                planned.append({
                    "name": tool_name,
                    "success": None,  # Not executed yet
                    "data": None,
                    "parameters": params,
                    "status": "pending",
                })
            
            # Generate a brief description of what we're about to do
            tool_names = [tc.get("tool", "") for tc in tool_calls]
            friendly_names = [self._friendly_tool_name(n) for n in tool_names]
            description = f"I'll {', '.join(friendly_names).lower()} for you. Please review and approve the actions below."
            
            return AgentResponse(
                text=description,
                tool_calls=planned,
            )
        
        # Step 5: No tools needed — generate conversational response
        response = await self._generate_response(context, [])
        
        # Step 6: Update memory
        memory_updates = await self._update_memory(context, response)
        response.memory_updates = memory_updates
        
        return response
    
    @staticmethod
    def _friendly_tool_name(tool_name: str) -> str:
        """Convert tool_name like 'asana_create_task' into 'create task in Asana'."""
        service_map = {
            "google": "Gmail/Google", "slack": "Slack", "asana": "Asana",
        }
        parts = tool_name.split("_", 1)
        service = service_map.get(parts[0], parts[0].title())
        action = parts[1].replace("_", " ") if len(parts) > 1 else tool_name
        return f"{action} via {service}"
    
    async def _recall_memory(self, context: AgentContext) -> str:
        """
        Recall relevant information from memory.
        
        Searches MEMORY.md, daily logs, and profile files for context
        relevant to the user's request.
        """
        try:
            memory_context = await self.memory.recall(
                query=context.message,
                user_id=context.user_id,
            )
            return memory_context
        except Exception as e:
            logger.warning(f"Memory recall failed: {e}")
            return ""
    
    async def _should_use_rag(self, context: AgentContext) -> bool:
        """
        Determine if RAG context should be retrieved.
        
        RAG is useful when:
        - User asks about channel content
        - User references past discussions
        - User needs information from indexed messages
        """
        # Keywords that suggest RAG would be helpful
        rag_triggers = [
            "summarize", "summary", "messages", "channel", "discussed",
            "talked about", "what happened", "conversation", "recent",
            "past", "earlier", "yesterday", "today", "this week",
        ]
        
        message_lower = context.message.lower()
        return any(trigger in message_lower for trigger in rag_triggers)
    
    async def _retrieve_rag_context(self, context: AgentContext) -> str:
        """
        Retrieve relevant context from the RAG system.
        
        Searches the vector store for messages similar to the user's query.
        """
        try:
            results = await self.retriever.retrieve(
                query=context.message,
                channel_id=context.channel_id,
                limit=5,
            )
            return self._format_rag_results(results)
        except Exception as e:
            logger.warning(f"RAG retrieval failed: {e}")
            return ""
    
    def _format_rag_results(self, results: list[dict]) -> str:
        """Format RAG results into a context string."""
        if not results:
            return ""
        
        formatted = []
        for r in results:
            formatted.append(f"[{r.get('timestamp', 'unknown')}] {r.get('user', 'user')}: {r.get('text', '')}")
        
        return "\n".join(formatted)
    
    async def _plan_tools(self, context: AgentContext) -> dict[str, Any]:
        """
        Determine which tools (if any) are needed for the request.
        
        Uses the LLM to analyze the request and decide on tool usage.
        """
        if not self.tools:
            return {"tools_needed": False}
        
        import re
        msg_lower = context.message.lower().strip()
        
        # =====================================================================
        # STEP 1: FAST-TRACK — If message clearly wants a tool action, skip
        #         all conversational checks to avoid false negatives.
        # =====================================================================
        action_verbs = [
            "create", "make", "add", "post", "send", "list", "show", "get",
            "read", "write", "upload", "search", "find", "fetch", "check",
            "summarize", "schedule", "remind", "assign", "update", "delete",
        ]
        service_keywords = [
            # Google
            "email", "gmail", "calendar", "event", "drive", "doc", "sheet",
            "spreadsheet", "contact",
            # Slack
            "slack", "channel", "#",
            # Asana
            "asana", "task", "project",
        ]
        
        has_action = any(v in msg_lower for v in action_verbs)
        has_service = any(s in msg_lower for s in service_keywords)
        
        if has_action and has_service:
            logger.info(f"Fast-tracking to tool selection (action + service detected)")
            # Fall through to LLM-based tool selection below
        else:
            # =================================================================
            # STEP 2: CONVERSATIONAL CHECK — Only runs when no clear action
            # =================================================================
            
            # Greetings: only match at the START of the message
            greeting_patterns = [
                r"^hello\b", r"^hi\b", r"^hey\b", r"^good morning\b",
                r"^good evening\b", r"^thanks\b", r"^thank you\b",
            ]
            for pattern in greeting_patterns:
                if re.search(pattern, msg_lower):
                    logger.info(f"Skipping tools for greeting: {pattern}")
                    return {"tools_needed": False, "reason": "Greeting"}
            
            # Exact conversational patterns (full-phrase matching)
            no_tool_patterns = [
                # Questions about user/agent
                "what do you know about me", "what you know about me", "who am i",
                "what's my name", "tell me about myself", "remember me", "do you know me",
                "know about me", "about me?",
                # Agent capability questions
                "what can you do", "what are your abilities", "how can you help",
                # PREFERENCE STATEMENTS (update memory, NOT trigger tools)
                "my name is", "call me",
                "my role is", "my job is",
                "my timezone is",
                "remember that", "remember this", "note that", "keep in mind",
                "my team is", "my project is", "my company is",
            ]
            
            for pattern in no_tool_patterns:
                if pattern in msg_lower:
                    logger.info(f"Skipping tools for conversational message: {pattern}")
                    return {"tools_needed": False, "reason": "Conversational question"}
            
            # REGEX-BASED: Detect role/team member statements
            role_patterns = [
                r"^\w+\s+is\s+(?:\w+\s+)*(?:head|lead|manager|director|ceo|cto|cfo|coo|vp|founder|co-founder)\b",
                r"^\w+\s+is\s+(?:the\s+)?(?:\w+\s+)*(?:developer|designer|engineer|analyst|architect|specialist|consultant|coordinator)\b",
                r"^\w+\s+is\s+(?:in charge of|responsible for|part of)\s+",
                r"^\w+\s+(?:joined|joins)\s+(?:as|the)\s+",
            ]
            
            for pattern in role_patterns:
                if re.search(pattern, msg_lower):
                    logger.info(f"Skipping tools for role/team statement")
                    return {"tools_needed": False, "reason": "Team/role information statement"}
            
            # Skip if message is short and doesn't contain action keywords
            if len(msg_lower.split()) <= 5 and not has_action:
                logger.info(f"Skipping tools for short non-action message")
                return {"tools_needed": False, "reason": "Short conversational message"}
        
        # =====================================================================
        # STEP 3: MULTI-SERVICE DETECTION — If message mentions 2+ distinct
        #         services, skip single-tool routing and let LLM plan.
        # =====================================================================
        service_groups = {
            "asana": ["asana"],
            "slack": ["slack", "#"],
            "google": ["email", "gmail", "calendar", "event", "drive", "doc", "sheet", "spreadsheet", "contact"],
        }
        
        detected_services = set()
        for service, keywords in service_groups.items():
            if any(kw in msg_lower for kw in keywords):
                detected_services.add(service)
        
        is_multi_service = len(detected_services) >= 2
        
        if is_multi_service:
            logger.info(f"Multi-service request detected ({', '.join(detected_services)}) — skipping single-tool routing, using LLM planner")
            # Fall through directly to LLM-based tool selection (Step 5)
        else:
            # =================================================================
            # STEP 3b: Explicit routing for Slack channel posts (single-service)
            # =================================================================
            slack_post_patterns = [
                r"update\s+(?:on\s+)?#(\w+)",
                r"post\s+(?:to\s+)?#(\w+)",
                r"tell\s+#(\w+)\s+(?:that|about)",
                r"notify\s+#(\w+)",
                r"inform\s+#(\w+)",
                r"send\s+(?:to\s+)?#(\w+)",
                r"message\s+#(\w+)",
            ]
            
            for pattern in slack_post_patterns:
                match = re.search(pattern, msg_lower)
                if match:
                    channel = match.group(1)
                    text_content = re.sub(pattern, "", msg_lower).strip()
                    text_content = re.sub(r"^(?:that|about|regarding|:)\s*", "", text_content).strip()
                    
                    if not text_content or len(text_content) < 5:
                        text_content = context.message
                    
                    logger.info(f"Routing to slack_post_message: channel={channel}")
                    return {
                        "tools_needed": True,
                        "tool_calls": [{
                            "tool": "slack_post_message",
                            "params": {
                                "channel": channel,
                                "text": text_content,
                            }
                        }]
                    }
        
        # =====================================================================
        # STEP 4: Deterministic routing for Google Workspace tools
        #         (Only for single-service requests)
        # =====================================================================
        if not is_multi_service:
            
            # --- Gmail: Create Draft (Proactive Verification with Smart Extraction) ---
            email_match = re.search(
                r"send\s+(?:an?\s+)?(?:g?e?mail)\s+to\s+(\S+)(?:\s+(?:that|saying|with body|about|content)\s+(.*))?",
                msg_lower,
                re.IGNORECASE
            )
            if email_match:
                to_addr = email_match.group(1)
                raw_body = email_match.group(2)
                
                subject = "Message from Gaprio"
                body = raw_body.strip() if raw_body else "Draft created from command."
                
                # Try to extract explicit subject if present
                subj_match = re.search(r"(?:subject|sub)\s+[\"']?(.+?)[\"']?(?:\s+|$)", msg_lower)
                if subj_match:
                    subject = subj_match.group(1).strip()
                
                logger.info(f"Routing to google_create_draft (Regex): to={to_addr}")
                return {
                    "tools_needed": True,
                    "tool_calls": [{"tool": "google_create_draft", "params": {"to": to_addr, "subject": subject, "body": body}}]
                }
                
            # --- Gmail: Send Draft ---
            draft_send_match = re.search(r"(?:send|confirm)\s+(?:the\s+)?draft\s+([a-zA-Z0-9_-]+)", msg_lower)
            if draft_send_match:
                draft_id = draft_send_match.group(1)
                logger.info(f"Routing to google_send_draft: id={draft_id}")
                return {
                    "tools_needed": True,
                    "tool_calls": [{"tool": "google_send_draft", "params": {"draft_id": draft_id}}]
                }
            
            # --- Gmail: Update Draft (Inline Editing) ---
            if msg_lower.startswith("update draft"):
                # Use original message for case-sensitive body content
                original_msg = context.message
                update_match = re.search(
                    r"update\s+draft\s+([^\s]+)\s+to\s+\"([\s\S]*?)\"\s+subject\s+\"([\s\S]*?)\"\s+body\s+\"([\s\S]*)\"",
                    original_msg,
                    re.IGNORECASE
                )
                if update_match:
                    draft_id, to, subject, body = update_match.groups()
                    
                    logger.info(f"Routing to google_update_draft: id={draft_id}")
                    return {
                        "tools_needed": True,
                        "tool_calls": [{"tool": "google_update_draft", "params": {
                            "draft_id": draft_id,
                            "to": to,
                            "subject": subject,
                            "body": body
                        }}]
                    }
            
            # --- Gmail: List emails ---
            if re.search(r"(?:list|show|get|check|read)\s+(?:my\s+)?(?:recent\s+)?(?:\d+\s+)?(?:emails?|inbox|mail)", msg_lower):
                max_r = 10
                num_match = re.search(r"(\d+)\s+(?:most\s+)?(?:recent\s+)?(?:emails?|mail)", msg_lower)
                if num_match:
                    max_r = int(num_match.group(1))
                logger.info(f"Routing to google_list_emails: max={max_r}")
                return {
                    "tools_needed": True,
                    "tool_calls": [{"tool": "google_list_emails", "params": {"max_results": max_r}}]
                }
            
            # --- Calendar: List events ---
            if (re.search(r"(?:what|list|show|get|check|any)\s+.*(?:event|calendar|schedule|meeting)", msg_lower) or \
               re.search(r"(?:event|calendar|schedule|meeting).*(?:today|tomorrow|this week|next week)", msg_lower)) and \
               not re.match(r"^(?:create|add|schedule|make|new)\b", msg_lower):
                logger.info(f"Routing to google_list_events")
                return {
                    "tools_needed": True,
                    "tool_calls": [{"tool": "google_list_events", "params": {"max_results": 10}}]
                }

            
            # --- Drive: List files ---
            if re.search(r"(?:list|show|get)\s+(?:my\s+)?(?:recent\s+)?(?:google\s+)?(?:drive\s+)?files", msg_lower) or \
               re.search(r"(?:list|show|get)\s+(?:my\s+)?(?:google\s+)?drive", msg_lower):
                logger.info(f"Routing to google_list_files")
                return {
                    "tools_needed": True,
                    "tool_calls": [{"tool": "google_list_files", "params": {"max_results": 10}}]
                }
            
            # --- Docs: Create doc ---
            if re.search(r"(?:create|make)\s+(?:a\s+)?(?:new\s+)?(?:google\s+)?doc", msg_lower):
                title = "Untitled Document"
                content = ""
                title_match = re.search(r"(?:called|named|titled)\s+[\"']?(.+?)[\"']?\s+(?:with|containing|content)", msg_lower)
                if not title_match:
                    title_match = re.search(r"(?:called|named|titled)\s+[\"']?(.+?)$", msg_lower)
                if title_match:
                    title = title_match.group(1).strip().strip("'\"")
                content_match = re.search(r"(?:with\s+(?:the\s+)?content|containing|body)\s+[\"']?(.+?)$", msg_lower)
                if content_match:
                    content = content_match.group(1).strip().strip("'\"")
                logger.info(f"Routing to google_create_doc: title={title}")
                return {
                    "tools_needed": True,
                    "tool_calls": [{"tool": "google_create_doc", "params": {"title": title, "body": content or "Created by Gaprio Agent"}}]
                }
            
            # --- Docs: List docs ---
            if re.search(r"(?:list|show|get)\s+(?:my\s+)?(?:google\s+)?docs", msg_lower):
                logger.info(f"Routing to google_list_docs")
                return {
                    "tools_needed": True,
                    "tool_calls": [{"tool": "google_list_docs", "params": {"max_results": 10}}]
                }
            
            # --- Sheets: Create sheet ---
            if re.search(r"(?:create|make)\s+(?:a\s+)?(?:new\s+)?(?:google\s+)?(?:spreadsheet|sheet)", msg_lower):
                title = "Untitled Spreadsheet"
                title_match = re.search(r"(?:called|named|titled)\s+[\"']?(.+?)$", msg_lower)
                if title_match:
                    title = title_match.group(1).strip().strip("'\"")
                logger.info(f"Routing to google_create_sheet: title={title}")
                return {
                    "tools_needed": True,
                    "tool_calls": [{"tool": "google_create_sheet", "params": {"title": title}}]
                }
            
            # --- Contacts: List contacts ---
            if re.search(r"(?:list|show|get)\s+(?:my\s+)?(?:google\s+)?contacts", msg_lower):
                logger.info(f"Routing to google_list_contacts")
                return {
                    "tools_needed": True,
                    "tool_calls": [{"tool": "google_list_contacts", "params": {"max_results": 20}}]
                }
            
            # --- Contacts: Search contacts ---
            contact_search = re.search(r"(?:search|find)\s+(?:my\s+)?contacts?\s+(?:for\s+)?[\"']?(.+?)$", msg_lower)
            if contact_search:
                query = contact_search.group(1).strip().strip("'\"")
                logger.info(f"Routing to google_search_contacts: query={query}")
                return {
                    "tools_needed": True,
                    "tool_calls": [{"tool": "google_search_contacts", "params": {"query": query}}]
                }

        
        tools_desc = self._build_tools_description()
        
        prompt = TOOL_SELECTION_PROMPT.format(
            tools_description=tools_desc,
            user_request=context.message,
            memory_context=context.memory_context or "No relevant memory found.",
        )
        
        try:
            response = await self.llm.generate(
                prompt=prompt,
                system_prompt="You are a tool selection assistant. Respond only with valid JSON.",
                temperature=0.2,  # Lower temperature for more deterministic selection
            )
            
            # Parse the JSON response
            return self._parse_tool_plan(response)
            
        except Exception as e:
            logger.error(f"Tool planning failed: {e}")
            return {"tools_needed": False}
    
    def _build_tools_description(self) -> str:
        """Build a detailed description of available tools for the LLM."""
        descriptions = []
        for name, tool in self.tools.items():
            tool_desc = f"### {name}\n{tool.description}\n"
            
            # Add parameter details
            params = tool.parameters.get("properties", {})
            required = tool.parameters.get("required", [])
            
            if params:
                tool_desc += "Parameters:\n"
                for param_name, param_info in params.items():
                    param_type = param_info.get("type", "any")
                    param_desc = param_info.get("description", "")
                    is_required = param_name in required
                    req_str = " (required)" if is_required else " (optional)"
                    default = param_info.get("default")
                    default_str = f", default: {default}" if default is not None else ""
                    tool_desc += f"  - {param_name} ({param_type}{req_str}): {param_desc}{default_str}\n"
            
            descriptions.append(tool_desc)
        
        return "\n".join(descriptions)
    
    def _parse_tool_plan(self, response: str) -> dict[str, Any]:
        """
        Parse the LLM's tool planning response.
        
        Attempts to recover from incomplete JSON by:
        1. Normal JSON parsing
        2. Trying to complete truncated JSON 
        3. Extracting tool name and params from partial JSON
        """
        try:
            # Try to extract JSON from the response
            start = response.find("{")
            end = response.rfind("}") + 1
            if start != -1 and end > start:
                json_str = response[start:end]
                data = json.loads(json_str)
                
                # --- SAFETY FILTER: Email vs Slack ---
                # If creating a draft, DO NOT post to Slack (common hallucination)
                if data.get("tools_needed") and data.get("tool_calls"):
                    calls = data["tool_calls"]
                    has_draft = any(c.get("tool") == "google_create_draft" for c in calls)
                    if has_draft:
                        # Filter out slack_post_message
                        original_len = len(calls)
                        data["tool_calls"] = [
                            c for c in calls 
                            if c.get("tool") != "slack_post_message"
                        ]
                        if len(data["tool_calls"]) < original_len:
                            logger.info("Fixed double-routing: Removed slack_post_message because google_create_draft is present.")
                            
                return data
        except json.JSONDecodeError:
            pass
        
        # Try to recover from truncated JSON
        try:
            # Look for patterns that indicate a tool call
            if '"tool":' in response or "'tool':" in response:
                # Try to extract tool name
                import re
                tool_match = re.search(r'"tool"\s*:\s*"([^"]+)"', response)
                if tool_match:
                    tool_name = tool_match.group(1)
                    
                    # Try to extract params - might be incomplete
                    params = {}
                    
                    # Extract common params for all tool types
                    common_params = [
                        # Asana task params
                        "name", "title", "project", "assignee", "due_on", "due_date", "content", "notes",
                        # Slack params
                        "channel", "text", "thread_ts", "hours",
                        # Google params
                        "to", "subject", "body", "summary", "start", "end",
                    ]
                    for param in common_params:
                        param_match = re.search(rf'"{param}"\s*:\s*"([^"]*)"', response)
                        if param_match:
                            params[param] = param_match.group(1)
                    
                    if tool_name:
                        logger.info(f"Recovered partial tool call: {tool_name} with params {params}")
                        
                        # Apply filter here too
                        if tool_name == "google_create_draft":
                            # Implicit filter: we are only returning one tool call here anyway
                            pass
                            
                        return {
                            "tools_needed": True,
                            "tool_calls": [{"tool": tool_name, "params": params}]
                        }
        except Exception as e:
            logger.warning(f"JSON recovery failed: {e}")
        
        logger.warning(f"Failed to parse tool plan: {response[:200]}")
        return {"tools_needed": False}
    
    async def _execute_tools(self, tool_calls: list[dict]) -> list[dict[str, Any]]:
        """
        Execute the planned tool calls.
        
        Each tool call is executed via its registered handler.
        Results are collected and returned for response generation.
        """
        results = []
        
        for call in tool_calls:
            tool_name = call.get("tool")
            params = call.get("params", {})
            
            if tool_name not in self.tools:
                logger.warning(f"Unknown tool: {tool_name}")
                results.append({
                    "tool": tool_name,
                    "success": False,
                    "error": f"Unknown tool: {tool_name}",
                })
                continue
            
            try:
                tool = self.tools[tool_name]
                result = await tool.handler(**params)
                results.append({
                    "tool": tool_name,
                    "success": True,
                    "result": result,
                })
                logger.info(f"Tool {tool_name} executed successfully")
                
            except Exception as e:
                logger.error(f"Tool {tool_name} failed: {e}")
                results.append({
                    "tool": tool_name,
                    "success": False,
                    "error": str(e),
                })
        
        return results
    
    async def _execute_tools_with_chaining(
        self,
        context: AgentContext,
        tool_calls: list[dict],
    ) -> list[dict[str, Any]]:
        """
        Execute tools with intelligent result chaining.
        
        For workflows like "read messages and post summary", this method:
        1. Executes read tools first
        2. Generates summaries from read results
        3. Uses actual content (not placeholders) for post operations
        """
        results = []
        collected_content = ""  # Content from read operations
        
        for call in tool_calls:
            tool_name = call.get("tool")
            params = call.get("params", {})
            
            if tool_name not in self.tools:
                logger.warning(f"Unknown tool: {tool_name}")
                results.append({
                    "tool": tool_name,
                    "success": False,
                    "error": f"Unknown tool: {tool_name}",
                })
                continue
            
            try:
                # Check if this is a post operation with placeholder text
                if tool_name == "slack_post_message":
                    text = params.get("text", "")
                    
                    # If text looks like a placeholder, generate real content
                    looks_like_placeholder = not text or "[" in text or "<" in text or text.lower().strip() == "summary"

                    if collected_content and (looks_like_placeholder or len(text) < 20):
                        # Generate summary from collected content
                        summary = await self._generate_summary(
                            context, collected_content
                        )
                        params["text"] = summary
                    elif looks_like_placeholder:
                        params["text"] = "I couldn't retrieve the messages to summarize. Please try again."
                
                tool = self.tools[tool_name]
                result = await tool.handler(**params)
                
                # Collect content from read operations for later use
                if tool_name == "slack_read_messages":
                    if result and hasattr(result, 'data') and result.data:
                        messages = result.data.get("messages", [])
                        if messages:
                            collected_content += "\n".join([
                                f"- {m.get('text', '')}" for m in messages[:20]
                            ]) + "\n"
                
                results.append({
                    "tool": tool_name,
                    "success": result.success if hasattr(result, 'success') else True,
                    "result": result,
                })
                logger.info(f"Tool {tool_name} executed successfully")
                
            except Exception as e:
                logger.error(f"Tool {tool_name} failed: {e}")
                results.append({
                    "tool": tool_name,
                    "success": False,
                    "error": str(e),
                })
        
        return results
    
    async def _generate_summary(self, context: AgentContext, content: str) -> str:
        """Generate a concise summary from content."""
        try:
            prompt = f"""Summarize the following Slack messages in a clean, friendly format.
Keep it concise (3-5 bullet points max). Use clear language.

Messages:
{content[:2000]}

Write a brief, friendly summary:"""
            
            summary = await self.llm.generate(
                prompt=prompt,
                system_prompt="You are a helpful assistant. Write clean, concise summaries.",
                temperature=0.5,
            )
            return summary.strip()
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            return "Summary of recent messages (see original channel for details)"
    
    async def _generate_response(
        self,
        context: AgentContext,
        tool_results: list[dict[str, Any]],
    ) -> AgentResponse:
        """
        Generate the final response to the user.
        
        If tools were executed, format results directly (bypass LLM).
        If no tools, use LLM for conversational response.
        """
        
        # =====================================================================
        # CASE 1: Tool results exist — format directly, DO NOT use LLM
        # =====================================================================
        if tool_results:
            response_text, ui_data = self._format_tool_results(context, tool_results)
            return AgentResponse(
                text=response_text,
                data=ui_data,
                tool_calls=[{
                    "name": r["tool"], 
                    "success": r["success"],
                    "data": r.get("result", {}).data if hasattr(r.get("result", {}), 'data') else None,
                } for r in tool_results],
            )
        
        # =====================================================================
        # CASE 2: No tools — use LLM for conversational response
        # =====================================================================
        prompt_parts = [f"User request: {context.message}"]
        
        if context.memory_context:
            prompt_parts.append(f"\nRelevant memory:\n{context.memory_context}")
        
        if context.rag_context:
            prompt_parts.append(f"\nRelevant channel messages:\n{context.rag_context}")
        
        # SPECIAL HANDLING: "About me" questions
        msg_lower = context.message.lower().strip()
        about_me_patterns = [
            "what do you know about me", "what you know about me", "who am i",
            "what's my name", "tell me about myself", "remember me", "do you know me",
            "know about me", "about me?", "know me",
        ]
        is_about_me = any(p in msg_lower for p in about_me_patterns)
        
        if is_about_me:
            prompt_parts.append("\n=== INSTRUCTION ===")
            prompt_parts.append("The user is asking what you know about them.")
            prompt_parts.append("Use the 'Relevant memory' section above to answer detailedly.")
            prompt_parts.append("If memory is empty, honestly say you don't know much yet.")
        
        full_prompt = "\n".join(prompt_parts)
        
        try:
            response_text = await self.llm.generate(
                prompt=full_prompt,
                system_prompt=SYSTEM_PROMPT,
                temperature=0.7,
            )
            
            # POST-PROCESS: If LLM echoes system prompt, replace with simple reply
            generic_phrases = [
                "what would you like", "how can i help", "let's get started",
                "your abilities", "i'm gaprio", "i am gaprio", "user request:",
                "=== tool results", "=== response rules", "core behavior rules",
                "tool action truthfulness", "context persistence",
            ]
            is_generic = any(phrase in response_text.lower() for phrase in generic_phrases)
            
            if is_generic:
                response_text = "Hey! How can I help you today? I can manage your emails, calendar, drive, docs, sheets, contacts, Slack, and Asana."
            
            return AgentResponse(text=response_text)
            
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return AgentResponse(
                text=f"I apologize, but I encountered an error processing your request: {e}",
                metadata={"error": str(e)},
            )
    
    def _format_tool_results(
        self,
        context: AgentContext,
        tool_results: list[dict[str, Any]],
    ) -> tuple[str, dict[str, Any] | None]:
        """
        Format tool results into a clean, structured response.
        
        Returns:
            Tuple of (formatted_text, ui_data)
        """
        lines = []
        ui_data = None
        
        for result in tool_results:
            tool_name = result.get("tool", "unknown")
            is_success = result.get("success", False)
            tool_result = result.get("result", {})
            
            if not is_success:
                error = result.get("error", "Unknown error")
                if hasattr(tool_result, "error") and tool_result.error:
                    error = tool_result.error
                lines.append(f"❌ **{self._friendly_tool_name(tool_name)}** failed: {error}")
                continue
            
            if not hasattr(tool_result, "data") or not tool_result.data:
                lines.append(f"✅ **{self._friendly_tool_name(tool_name)}** completed.")
                continue
            
            data = tool_result.data
            
            # =================================================================
            # Gmail
            # =================================================================
            if tool_name == "google_create_draft":
                draft_id = data.get("id", "unknown")
                lines.append(f"📝 **Draft Created for Review**")
                lines.append(f"   To: {data.get('to', 'recipient')}")
                if data.get("subject"):
                    lines.append(f"   Subject: {data['subject']}")
                if data.get("url"):
                    lines.append(f"   🔗 **[Open in Gmail to Edit/Send]({data['url']})**")
                lines.append(f"\n👉 **Reply 'send draft {draft_id}' to confirm sending.**")
                
                # Prepare UI data for frontend card
                ui_data = {
                    "type": "draft_review",
                    "draft_id": draft_id,
                    "to": data.get("to", ""),
                    "subject": data.get("subject", ""),
                    "body": data.get("body", ""),  # Include body for editing
                    "url": data.get("url", ""),
                }
            
            elif tool_name == "google_update_draft":
                draft_id = data.get("id", "unknown")
                lines.append(f"📝 **Draft Updated**")
                
                # Return updated data for UI refresh
                ui_data = {
                    "type": "draft_review",
                    "draft_id": draft_id,
                    "to": data.get("to", ""),
                    "subject": data.get("subject", ""),
                    "body": data.get("body", ""),
                    "url": data.get("url", ""),
                }
            
            elif tool_name == "google_send_draft":
                lines.append(f"✅ **Email Sent Successfully** (Draft {data.get('id')} sent)")
            
            elif tool_name == "google_send_email":
                lines.append(f"✅ **Email sent** to {data.get('to', 'recipient')}")
                if data.get("subject"):
                    lines.append(f"   Subject: {data['subject']}")
                if data.get("id"):
                    lines.append(f"   Message ID: {data['id']}")
            
            elif tool_name == "google_list_emails":
                emails = data.get("emails", [])
                count = data.get("count", len(emails))
                lines.append(f"📧 **{count} emails found:**\n")
                for i, email in enumerate(emails[:15], 1):
                    subj = email.get("subject", "(no subject)")
                    sender = email.get("from", "Unknown")
                    date = email.get("date", "")
                    snippet = email.get("snippet", "")[:80]
                    lines.append(f"{i}. **{subj}**")
                    lines.append(f"   From: {sender}")
                    lines.append(f"   Date: {date}")
                    if snippet:
                        lines.append(f"   Preview: {snippet}...")
                    lines.append("")
            
            elif tool_name == "google_read_email":
                lines.append(f"📧 **Email Details:**")
                lines.append(f"   Subject: {data.get('subject', '(none)')}")
                lines.append(f"   From: {data.get('from', 'Unknown')}")
                lines.append(f"   Date: {data.get('date', '')}")
                lines.append(f"\n{data.get('body', '(empty)')}")
            
            # =================================================================
            # Calendar
            # =================================================================
            elif tool_name == "google_list_events":
                events = data.get("events", [])
                count = data.get("count", len(events))
                if count == 0:
                    lines.append("📅 **No upcoming events found.**")
                else:
                    lines.append(f"📅 **{count} events:**\n")
                    for i, evt in enumerate(events, 1):
                        lines.append(f"{i}. **{evt.get('summary', '(no title)')}**")
                        lines.append(f"   Start: {evt.get('start', 'N/A')}")
                        lines.append(f"   End: {evt.get('end', 'N/A')}")
                        if evt.get("location"):
                            lines.append(f"   Location: {evt['location']}")
                        if evt.get("url"):
                            lines.append(f"   Link: {evt['url']}")
                        lines.append("")
            
            elif tool_name == "google_create_event":
                lines.append(f"✅ **Event created:** {data.get('summary', 'New Event')}")
                if data.get("url"):
                    lines.append(f"   Link: {data['url']}")
            
            # =================================================================
            # Drive
            # =================================================================
            elif tool_name == "google_list_files":
                files = data.get("files", [])
                count = data.get("count", len(files))
                if count == 0:
                    lines.append("📁 **No files found in Google Drive.**")
                else:
                    lines.append(f"📁 **{count} files in Google Drive:**\n")
                    for i, f in enumerate(files, 1):
                        lines.append(f"{i}. **{f.get('name', 'Untitled')}**")
                        lines.append(f"   Type: {f.get('mimeType', 'Unknown')}")
                        if f.get("webViewLink"):
                            lines.append(f"   Link: {f['webViewLink']}")
                        lines.append("")
            
            elif tool_name == "google_upload_file":
                lines.append(f"✅ **File uploaded:** {data.get('name', 'file')}")
                if data.get("webViewLink"):
                    lines.append(f"   Link: {data['webViewLink']}")
            
            # =================================================================
            # Docs
            # =================================================================
            elif tool_name == "google_create_doc":
                lines.append(f"✅ **Google Doc created:** {data.get('title', 'Untitled')}")
                if data.get("url"):
                    lines.append(f"   Link: {data['url']}")
                if data.get("document_id"):
                    lines.append(f"   ID: {data['document_id']}")
            
            elif tool_name == "google_list_docs":
                docs = data.get("documents", data.get("files", []))
                count = data.get("count", len(docs))
                if count == 0:
                    lines.append("📄 **No Google Docs found.**")
                else:
                    lines.append(f"📄 **{count} Google Docs:**\n")
                    for i, doc in enumerate(docs, 1):
                        lines.append(f"{i}. **{doc.get('name', 'Untitled')}**")
                        if doc.get("webViewLink"):
                            lines.append(f"   Link: {doc['webViewLink']}")
                        lines.append("")
            
            elif tool_name == "google_read_doc":
                lines.append(f"📄 **Document:** {data.get('title', 'Untitled')}")
                content = data.get("content", "(empty)")
                lines.append(f"\n{content[:2000]}")
            
            # =================================================================
            # Sheets
            # =================================================================
            elif tool_name == "google_create_sheet":
                lines.append(f"✅ **Spreadsheet created:** {data.get('title', 'Untitled')}")
                if data.get("url"):
                    lines.append(f"   Link: {data['url']}")
            
            elif tool_name in ("google_read_sheet", "google_write_sheet"):
                lines.append(f"✅ **Sheet operation completed.**")
                if data.get("values"):
                    lines.append(f"   Rows: {len(data['values'])}")
            
            # =================================================================
            # Contacts
            # =================================================================
            elif tool_name in ("google_list_contacts", "google_search_contacts"):
                contacts = data.get("contacts", [])
                count = data.get("count", len(contacts))
                if count == 0:
                    lines.append("👤 **No contacts found.**")
                else:
                    lines.append(f"👤 **{count} contacts:**\n")
                    for i, c in enumerate(contacts, 1):
                        name = c.get("name", "Unknown")
                        email = c.get("email", "")
                        phone = c.get("phone", "")
                        lines.append(f"{i}. **{name}**")
                        if email:
                            lines.append(f"   Email: {email}")
                        if phone:
                            lines.append(f"   Phone: {phone}")
                        lines.append("")
            
            # =================================================================
            # Slack
            # =================================================================
            elif tool_name == "slack_post_message":
                channel = data.get("channel") or data.get("channel_name", "channel")
                lines.append(f"✅ **Message posted** to #{channel}")
                if data.get("ts"):
                    lines.append(f"   Timestamp: {data['ts']}")
            
            elif tool_name == "slack_read_messages":
                msgs = data.get("messages", [])
                count = data.get("count", len(msgs))
                lines.append(f"📨 **{count} messages retrieved.**")
                
                # Add the structured text summary if available
                if data.get("structured_text"):
                    lines.append("\n" + data["structured_text"])
            
            # =================================================================
            # Asana
            # =================================================================
            elif "asana" in tool_name:
                tasks = data.get("tasks", [])
                if tasks:
                    source = "Asana"
                    lines.append(f"**{source} ({len(tasks)} items):**\n")
                    for i, task in enumerate(tasks, 1):
                        lines.append(f"{i}. **{task.get('name', 'Untitled')}**")
                        lines.append(f"   Assigned: {task.get('assignee', 'Unassigned')}")
                        lines.append(f"   Due: {task.get('due_date') or task.get('due_on', 'No deadline')}")
                        if task.get("url"):
                            lines.append(f"   {task['url']}")
                        lines.append("")
                else:
                    name = data.get("name") or data.get("title", "")
                    url = data.get("url") or data.get("permalink_url", "")
                    message = data.get("message", "")
                    if name and url:
                        lines.append(f"✅ **{name}**: {url}")
                    elif name:
                        lines.append(f"✅ Created: **{name}**")
                    elif message:
                        lines.append(f"✅ {message}")
                    else:
                        lines.append(f"✅ **{self._friendly_tool_name(tool_name)}** completed.")
            
            # =================================================================
            # Generic fallback
            # =================================================================
            else:
                name = data.get("name") or data.get("title") or data.get("summary", "")
                url = data.get("url") or data.get("page_url") or data.get("html_url", "")
                message = data.get("message", "")
                count = data.get("count")
                
                if name and url:
                    lines.append(f"✅ **{name}**: {url}")
                elif name:
                    lines.append(f"✅ {self._friendly_tool_name(tool_name)}: **{name}**")
                elif message:
                    lines.append(f"✅ {message}")
                elif count is not None:
                    lines.append(f"✅ {self._friendly_tool_name(tool_name)}: {count} items")
                else:
                    lines.append(f"✅ **{self._friendly_tool_name(tool_name)}** completed.")
                    lines.append(f"   Data: {str(data)[:200]}")
        
        return "\n".join(lines) if lines else "Done!", ui_data
    
    @staticmethod
    def _friendly_tool_name(tool_name: str) -> str:
        """Convert tool_name like 'google_send_email' to 'Send Email'."""
        name = tool_name.replace("google_", "").replace("slack_", "").replace("asana_", "")
        return name.replace("_", " ").title()
    
    async def _update_memory(
        self,
        context: AgentContext,
        response: AgentResponse,
    ) -> list[dict[str, Any]]:
        """
        Update memory with new information from this interaction.
        
        Decides what (if anything) should be saved to memory:
        - Important decisions or commitments -> MEMORY.md
        - Event details -> Daily log
        - User preferences -> Profile
        """
        updates = []
        
        # Extract and save any preferences from the user's message
        try:
            preferences = self.memory.extract_and_save_preferences(
                message=context.message,
                user_id=context.user_id,
            )
            if preferences:
                updates.append({
                    "type": "preferences",
                    "learned": preferences,
                })
                logger.info(f"Learned {len(preferences)} preference(s) from user")
        except Exception as e:
            logger.warning(f"Failed to extract preferences: {e}")
        
        # Log the interaction to the daily log
        try:
            self.memory.log_interaction(
                user_id=context.user_id,
                channel_id=context.channel_id,
                message=context.message,
                response=response.text,
            )
            updates.append({"type": "daily_log", "status": "logged"})
        except Exception as e:
            logger.warning(f"Failed to log interaction: {e}")
        
        return updates
    
    async def close(self) -> None:
        """Clean up resources."""
        await self.llm.close()
        await self.retriever.close()
