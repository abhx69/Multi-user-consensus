"""
Knowledge Graph MCP Server.
Lets the LLM query the knowledge graph to understand relationships
between people, tasks, documents, conversations, and emails across
all connected platforms.
Tools:
    knowledge_graph_search   — fulltext search across graph nodes
    get_person_context       — all tasks/docs/conversations for a person
    get_topic_context        — all related content for a topic (2-hop search)
Requires:
    - MySQL tables: knowledge_graph_nodes, knowledge_graph_edges
    - db/knowledge_graph_queries.py (query functions)
"""
import logging
from gaprio.mcp.base_server import BaseMCPServer, MCPTool, ToolResult
logger = logging.getLogger(__name__)
class KnowledgeGraphMCPServer(BaseMCPServer):
    """
    MCP Server for Knowledge Graph queries.
    
    Provides tools for the agent to search and traverse the
    knowledge graph, finding relationships between entities
    across platforms (Slack, Gmail, Asana, Google Docs).
    """
    
    def __init__(self):
        """Initialize the Knowledge Graph MCP server."""
        super().__init__(name="knowledge_graph")
    
    def _register_tools(self) -> None:
        """Register all Knowledge Graph tools."""
        
        self.add_tool(MCPTool(
            name="knowledge_graph_search",
            description=(
                "Search the knowledge graph for nodes matching a query. "
                "Returns people, tasks, documents, conversations, and emails "
                "related to the search term, along with their connections."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search term",
                    },
                    "workspace_id": {
                        "type": "string",
                        "description": "Workspace ID",
                    },
                    "node_type": {
                        "type": "string",
                        "description": (
                            "Filter by type: person, task, document, "
                            "conversation, email, meeting"
                        ),
                        "default": "",
                    },
                },
                "required": ["query", "workspace_id"],
            },
            handler=self._handle_search,
        ))
        
        self.add_tool(MCPTool(
            name="get_person_context",
            description=(
                "Get all tasks, documents, conversations, and emails "
                "connected to a specific person in the knowledge graph."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "name_or_email": {
                        "type": "string",
                        "description": "Person's name or email",
                    },
                    "workspace_id": {
                        "type": "string",
                        "description": "Workspace ID",
                    },
                },
                "required": ["name_or_email", "workspace_id"],
            },
            handler=self._handle_person_context,
        ))
        
        self.add_tool(MCPTool(
            name="get_topic_context",
            description=(
                "Get all related content across platforms for a specific topic. "
                "Searches the knowledge graph and returns connected nodes "
                "up to 2 hops away."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "Topic to search for",
                    },
                    "workspace_id": {
                        "type": "string",
                        "description": "Workspace ID",
                    },
                },
                "required": ["topic", "workspace_id"],
            },
            handler=self._handle_topic_context,
        ))
    
    # =========================================================================
    # Handlers
    # =========================================================================
    
    async def _handle_search(
        self,
        query: str,
        workspace_id: str,
        node_type: str = "",
    ) -> ToolResult:
        """
        Search the knowledge graph for matching nodes.
        
        For each found node, also fetches 1-hop related nodes
        to provide immediate context.
        """
        try:
            from gaprio.db.knowledge_graph_queries import (
                search_nodes,
                get_related_nodes,
            )
            
            nodes = await search_nodes(
                workspace_id,
                query,
                node_type=node_type or None,
                limit=10,
            )
            
            # For each found node, get 1-hop related nodes
            enriched = []
            for node in nodes:
                related = await get_related_nodes(node["id"], max_hops=1)
                enriched.append({**node, "related": related[:5]})
            
            return ToolResult(
                success=True,
                data={"nodes": enriched, "count": len(enriched)},
            )
            
        except Exception as e:
            logger.error(f"Knowledge graph search error: {e}")
            return ToolResult(success=False, error=str(e))
    
    async def _handle_person_context(
        self,
        name_or_email: str,
        workspace_id: str,
    ) -> ToolResult:
        """
        Get complete context for a person — all connected tasks,
        documents, conversations, emails, and meetings.
        """
        try:
            from gaprio.db.knowledge_graph_queries import (
                search_nodes,
                get_person_context,
            )
            
            # Search for the person by name
            people = await search_nodes(
                workspace_id,
                name_or_email,
                node_type="person",
                limit=3,
            )
            
            if not people:
                return ToolResult(
                    success=True,
                    data={
                        "person": None,
                        "message": f"No person found matching '{name_or_email}'",
                    },
                )
            
            person = people[0]
            context = await get_person_context(
                workspace_id,
                person["external_id"],
            )
            
            return ToolResult(success=True, data=context)
            
        except Exception as e:
            logger.error(f"Person context error: {e}")
            return ToolResult(success=False, error=str(e))
    
    async def _handle_topic_context(
        self,
        topic: str,
        workspace_id: str,
    ) -> ToolResult:
        """
        Get all related content for a topic, traversing the graph
        up to 2 hops from matching nodes.
        
        This connects the dots across platforms: a Slack discussion
        about "login bug" → related Asana task → related Google Doc.
        """
        try:
            from gaprio.db.knowledge_graph_queries import (
                search_nodes,
                get_related_nodes,
            )
            
            nodes = await search_nodes(workspace_id, topic, limit=5)
            
            all_related = []
            seen_ids = set()
            
            for node in nodes:
                related = await get_related_nodes(node["id"], max_hops=2)
                for r in related:
                    if r["id"] not in seen_ids:
                        seen_ids.add(r["id"])
                        all_related.append(r)
            
            return ToolResult(
                success=True,
                data={
                    "topic": topic,
                    "direct_matches": nodes,
                    "related": all_related[:20],
                },
            )
            
        except Exception as e:
            logger.error(f"Topic context error: {e}")
            return ToolResult(success=False, error=str(e))