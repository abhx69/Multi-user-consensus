"""
RAG Retriever for context retrieval.

Provides the interface between the agent and the vector store.
Handles query processing, retrieval, and context formatting.

The retriever is the final step in RAG:
1. Query from user/agent
2. Search vector store
3. Format results for LLM context
"""

import logging
from typing import Any

from gaprio.rag.vector_store import VectorStore

logger = logging.getLogger(__name__)


class Retriever:
    """
    Retrieves relevant context from the vector store.
    
    Provides semantic search over indexed Slack messages,
    formatting results for use in agent prompts.
    
    Usage:
        retriever = Retriever()
        
        # Get context for a query
        results = await retriever.retrieve(
            query="deployment issues",
            channel_id="C12345",
            limit=5
        )
        
        # Format for prompt
        context = retriever.format_context(results)
    """
    
    def __init__(self, vector_store: VectorStore | None = None):
        """
        Initialize the retriever.
        
        Args:
            vector_store: Vector store instance (created if not provided)
        """
        self.vector_store = vector_store or VectorStore()
        
        logger.info("Retriever initialized")
    
    async def retrieve(
        self,
        query: str,
        channel_id: str | None = None,
        channel_ids: list[str] | None = None,
        limit: int = 5,
        min_score: float | None = None,
    ) -> list[dict[str, Any]]:
        """
        Retrieve relevant messages for a query.
        
        Args:
            query: The search query
            channel_id: Single channel to search
            channel_ids: Multiple channels to search
            limit: Maximum results to return
            min_score: Optional minimum similarity score
            
        Returns:
            List of relevant message dicts
        """
        if channel_id:
            results = self.vector_store.search(channel_id, query, limit=limit)
        elif channel_ids:
            results = self.vector_store.search_all_channels(query, channel_ids, limit=limit)
        else:
            logger.warning("No channel specified for retrieval")
            return []
        
        # Filter by similarity score if specified
        if min_score is not None:
            # Note: ChromaDB uses distance (lower = more similar)
            # Convert to similarity if needed
            results = [r for r in results if r.get("distance", 1.0) <= (1 - min_score)]
        
        logger.debug(f"Retrieved {len(results)} results for: {query[:50]}...")
        return results
    
    async def retrieve_for_summarization(
        self,
        channel_id: str,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """
        Retrieve messages for channel summarization.
        
        Gets a broader set of recent messages rather than
        focused semantic search.
        
        Args:
            channel_id: Channel to summarize
            limit: Number of messages to retrieve
            
        Returns:
            Recent messages from the channel
        """
        # For summarization, we want recent messages
        # This is more of a direct fetch than semantic search
        # The indexer should have recent messages
        
        # Use a generic query to get diverse results
        results = self.vector_store.search(
            channel_id,
            "recent discussion conversation",  # Broad query
            limit=limit,
        )
        
        # Sort by timestamp if available
        results.sort(
            key=lambda x: x.get("timestamp", "0"),
            reverse=True,
        )
        
        return results
    
    def format_context(
        self,
        results: list[dict[str, Any]],
        include_metadata: bool = True,
    ) -> str:
        """
        Format retrieval results for LLM context.
        
        Args:
            results: Retrieved messages
            include_metadata: Include user/timestamp info
            
        Returns:
            Formatted context string
        """
        if not results:
            return "No relevant context found."
        
        lines = []
        for r in results:
            if include_metadata:
                user = r.get("user", "unknown")
                ts = r.get("timestamp", "")
                # Format timestamp if it's a Slack timestamp
                if "." in ts:
                    ts = ts.split(".")[0]  # Remove decimal part
                
                line = f"[{ts}] {user}: {r.get('text', '')}"
            else:
                line = r.get("text", "")
            
            lines.append(line)
        
        return "\n".join(lines)
    
    def format_context_for_summary(
        self,
        results: list[dict[str, Any]],
        channel_name: str = "channel",
    ) -> str:
        """
        Format context specifically for summarization.
        
        Provides more structure for the LLM to generate summaries.
        
        Args:
            results: Retrieved messages
            channel_name: Name of the channel
            
        Returns:
            Formatted context for summarization
        """
        if not results:
            return f"No messages found in #{channel_name}."
        
        header = f"Messages from #{channel_name} ({len(results)} messages):\n"
        header += "=" * 40 + "\n"
        
        messages = []
        for r in results:
            user = r.get("user", "unknown")
            text = r.get("text", "")
            messages.append(f"• [{user}]: {text}")
        
        return header + "\n".join(messages)
    
    async def get_similar_discussions(
        self,
        topic: str,
        channel_ids: list[str],
        limit: int = 10,
    ) -> dict[str, list[dict]]:
        """
        Find similar discussions across channels.
        
        Useful for finding related conversations when
        researching a topic.
        
        Args:
            topic: Topic to search for
            channel_ids: Channels to search
            limit: Max results per channel
            
        Returns:
            Dict mapping channel_id to relevant messages
        """
        results = {}
        
        for channel_id in channel_ids:
            channel_results = self.vector_store.search(
                channel_id,
                topic,
                limit=limit,
            )
            if channel_results:
                results[channel_id] = channel_results
        
        return results
    
    async def close(self) -> None:
        """Clean up resources."""
        await self.vector_store.close()
