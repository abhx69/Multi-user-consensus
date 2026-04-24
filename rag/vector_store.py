"""
Vector store for RAG using ChromaDB.

Provides embedding storage and similarity search for Slack messages.
Uses sentence-transformers for local embedding generation (no API calls).

The vector store is the foundation of the RAG system:
1. Messages are embedded and stored with metadata
2. Queries are embedded and matched to similar messages
3. Retrieved context is passed to the agent
"""

import logging
from pathlib import Path
from typing import Any

from gaprio.config import settings

logger = logging.getLogger(__name__)

# Lazy imports - only load heavy dependencies when needed
_chroma_client = None
_embedding_model = None


def _get_chroma_client():
    """Lazy initialization of ChromaDB client."""
    global _chroma_client
    if _chroma_client is None:
        import chromadb
        
        persist_dir = str(settings.chroma_persist_dir)
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        
        # Use new PersistentClient API (ChromaDB >= 0.4.0)
        _chroma_client = chromadb.PersistentClient(
            path=persist_dir,
            settings=chromadb.config.Settings(
                anonymized_telemetry=False,
            )
        )
        logger.info(f"ChromaDB initialized at {persist_dir}")
    
    return _chroma_client


def _get_embedding_model():
    """Lazy initialization of embedding model."""
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        
        model_name = settings.embedding_model
        _embedding_model = SentenceTransformer(model_name)
        logger.info(f"Loaded embedding model: {model_name}")
    
    return _embedding_model


class VectorStore:
    """
    Vector store for message embeddings.
    
    Uses ChromaDB for storage and sentence-transformers for embeddings.
    Each Slack channel gets its own collection for isolation.
    
    Usage:
        store = VectorStore()
        
        # Add messages
        store.add_messages("channel_123", messages)
        
        # Search
        results = store.search("channel_123", "deployment issues", limit=5)
    """
    
    def __init__(self, collection_prefix: str = "slack"):
        """
        Initialize the vector store.
        
        Args:
            collection_prefix: Prefix for collection names
        """
        self.collection_prefix = collection_prefix
        self._collections: dict[str, Any] = {}
        
        logger.info("VectorStore initialized")
    
    def _get_collection(self, channel_id: str):
        """Get or create a collection for a channel."""
        collection_name = f"{self.collection_prefix}_{channel_id}"
        
        if collection_name not in self._collections:
            client = _get_chroma_client()
            self._collections[collection_name] = client.get_or_create_collection(
                name=collection_name,
                metadata={"channel_id": channel_id},
            )
        
        return self._collections[collection_name]
    
    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for texts."""
        model = _get_embedding_model()
        embeddings = model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()
    
    def add_messages(
        self,
        channel_id: str,
        messages: list[dict[str, Any]],
    ) -> int:
        """
        Add messages to the vector store.
        
        Args:
            channel_id: Slack channel ID
            messages: List of message dicts with text, user, timestamp, etc.
            
        Returns:
            Number of messages added
        """
        if not messages:
            return 0
        
        collection = self._get_collection(channel_id)
        
        # Prepare data for ChromaDB
        ids = []
        documents = []
        metadatas = []
        
        for msg in messages:
            msg_id = f"{channel_id}_{msg.get('ts', '')}"
            text = msg.get("text", "")
            
            if not text.strip():
                continue
            
            ids.append(msg_id)
            documents.append(text)
            metadatas.append({
                "user": msg.get("user", "unknown"),
                "timestamp": msg.get("ts", ""),
                "channel_id": channel_id,
                "thread_ts": msg.get("thread_ts", ""),
            })
        
        if not documents:
            return 0
        
        # Generate embeddings
        embeddings = self._embed_texts(documents)
        
        # Upsert to handle duplicates
        collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        
        logger.info(f"Added {len(documents)} messages to {channel_id}")
        return len(documents)
    
    def search(
        self,
        channel_id: str,
        query: str,
        limit: int = 5,
        where: dict | None = None,
    ) -> list[dict[str, Any]]:
        """
        Search for similar messages.
        
        Args:
            channel_id: Channel to search in
            query: Search query
            limit: Maximum results to return
            where: Optional filter conditions
            
        Returns:
            List of matching messages with metadata
        """
        collection = self._get_collection(channel_id)
        
        # Generate query embedding
        query_embedding = self._embed_texts([query])[0]
        
        # Search
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=limit,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        
        # Format results
        formatted = []
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                formatted.append({
                    "text": doc,
                    "user": results["metadatas"][0][i].get("user", "unknown"),
                    "timestamp": results["metadatas"][0][i].get("timestamp", ""),
                    "distance": results["distances"][0][i] if results["distances"] else None,
                })
        
        return formatted
    
    def search_all_channels(
        self,
        query: str,
        channel_ids: list[str],
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Search across multiple channels.
        
        Args:
            query: Search query
            channel_ids: Channels to search
            limit: Maximum results per channel
            
        Returns:
            Combined results from all channels
        """
        all_results = []
        
        for channel_id in channel_ids:
            results = self.search(channel_id, query, limit=limit)
            for r in results:
                r["channel_id"] = channel_id
            all_results.extend(results)
        
        # Sort by distance (similarity)
        all_results.sort(key=lambda x: x.get("distance", float("inf")))
        
        return all_results[:limit * 2]  # Return more for cross-channel
    
    def delete_channel(self, channel_id: str) -> bool:
        """
        Delete all data for a channel.
        
        Args:
            channel_id: Channel to delete
            
        Returns:
            True if successful
        """
        try:
            collection_name = f"{self.collection_prefix}_{channel_id}"
            client = _get_chroma_client()
            client.delete_collection(collection_name)
            
            if collection_name in self._collections:
                del self._collections[collection_name]
            
            logger.info(f"Deleted collection for {channel_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete channel {channel_id}: {e}")
            return False
    
    def get_stats(self, channel_id: str) -> dict[str, Any]:
        """
        Get statistics for a channel's collection.
        
        Args:
            channel_id: Channel to check
            
        Returns:
            Dict with count and other stats
        """
        try:
            collection = self._get_collection(channel_id)
            return {
                "channel_id": channel_id,
                "count": collection.count(),
            }
        except Exception:
            return {"channel_id": channel_id, "count": 0}
    
    def persist(self) -> None:
        """Persist the database to disk.
        
        Note: With PersistentClient, data is auto-persisted.
        This method is kept for API compatibility.
        """
        # PersistentClient auto-persists, no manual call needed
        logger.debug("ChromaDB using PersistentClient - auto-persisting enabled")
    
    async def close(self) -> None:
        """Clean up resources."""
        self.persist()
