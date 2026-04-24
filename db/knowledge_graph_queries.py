# db/knowledge_graph_queries.py — stub signatures
async def search_nodes(workspace_id: str, query: str, node_type: str = None, limit: int = 10) -> list[dict]:
    """Fulltext search across knowledge_graph_nodes table."""
    ...
async def get_related_nodes(node_id: int, max_hops: int = 1) -> list[dict]:
    """Get nodes connected to a given node up to N hops away via knowledge_graph_edges."""
    ...
async def get_person_context(workspace_id: str, external_id: str) -> dict:
    """Get all tasks, docs, conversations, emails connected to a person."""
    ...