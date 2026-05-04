"""
Slack message indexer for RAG.

Responsible for:
1. Fetching messages from Slack channels
2. Indexing them in the vector store
3. Managing indexing frequency and history

The indexer runs periodically (configurable) to keep the
knowledge base up to date with recent conversations.
"""

import logging
from datetime import datetime, timedelta
from typing import Any

from gaprio.config import settings
from gaprio.memory.heartbeat import HeartbeatManager
from gaprio.rag.vector_store import VectorStore

logger = logging.getLogger(__name__)


class SlackIndexer:
    """
    Indexes Slack channel messages for RAG retrieval.
    
    Fetches messages from Slack and adds them to the vector store.
    Tracks indexing history to avoid re-indexing unchanged data.
    
    Usage:
        indexer = SlackIndexer(slack_client)
        
        # Index a specific channel
        await indexer.index_channel("C12345")
        
        # Index all channels the bot is in
        await indexer.index_all_channels()
    """
    
    def __init__(
        self,
        slack_client: Any = None,
        vector_store: VectorStore | None = None,
        heartbeat: HeartbeatManager | None = None,
    ):
        """
        Initialize the indexer.
        
        Args:
            slack_client: Slack WebClient instance
            vector_store: Vector store for embeddings
            heartbeat: Heartbeat manager for tracking
        """
        self.slack_client = slack_client
        self.vector_store = vector_store or VectorStore()
        self.heartbeat = heartbeat or HeartbeatManager()
        self.message_limit = settings.rag_index_message_count
        
        logger.info("SlackIndexer initialized")
    
    def set_slack_client(self, client: Any) -> None:
        """
        Set the Slack client.
        
        This allows late binding if the client isn't available at init time.
        
        Args:
            client: Slack WebClient instance
        """
        self.slack_client = client
    
    async def should_index(self, channel_id: str) -> bool:
        """
        Check if a channel should be indexed.
        
        Based on the indexing frequency setting and last index time.
        
        Args:
            channel_id: Channel to check
            
        Returns:
            True if channel should be indexed
        """
        operation = f"index_channel_{channel_id}"
        interval = timedelta(hours=settings.rag_index_frequency_hours)
        return self.heartbeat.should_run(operation, interval)
    
    async def index_channel(
        self,
        channel_id: str,
        force: bool = False,
    ) -> dict[str, Any]:
        """
        Index messages from a Slack channel.
        
        Fetches recent messages and adds them to the vector store.
        
        Args:
            channel_id: Slack channel ID
            force: Index even if not due
            
        Returns:
            Dict with indexing stats
        """
        if not self.slack_client:
            logger.error("No Slack client configured")
            return {"success": False, "error": "No Slack client"}
        
        # Check if indexing is needed
        if not force and not await self.should_index(channel_id):
            logger.debug(f"Skipping {channel_id} - not due for indexing")
            return {"success": True, "skipped": True, "reason": "Not due"}
        
        try:
            # Fetch messages from Slack
            messages = await self._fetch_messages(channel_id)
            
            if not messages:
                logger.info(f"No messages to index in {channel_id}")
                return {"success": True, "count": 0}
            
            # Add to vector store
            count = self.vector_store.add_messages(channel_id, messages)
            
            # Record the indexing
            self.heartbeat.record_check(f"index_channel_{channel_id}")
            
            logger.info(f"Indexed {count} messages from {channel_id}")
            return {
                "success": True,
                "count": count,
                "channel_id": channel_id,
            }
            
        except Exception as e:
            logger.error(f"Failed to index {channel_id}: {e}")
            return {"success": False, "error": str(e)}
    
    async def _fetch_messages(
        self,
        channel_id: str,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Fetch and filter messages from a Slack channel.
        
        Uses MessageFilter to:
        - Remove noise (bots, system messages)
        - Preserve Gaprio decisions
        - De-duplicate messages
        
        Args:
            channel_id: Channel to fetch from
            limit: Max messages to fetch (default from settings)
            
        Returns:
            List of cleaned message dicts
        """
        from gaprio.rag.message_filter import filter_for_indexing
        
        limit = limit or self.message_limit
        
        try:
            # Use Slack's conversations.history API
            response = await self._call_slack_api(
                "conversations.history",
                channel=channel_id,
                limit=limit,
            )
            
            if not response.get("ok"):
                logger.error(f"Slack API error: {response.get('error')}")
                return []
            
            messages = response.get("messages", [])
            
            # Apply smart filtering for indexing
            result = filter_for_indexing(messages)
            
            # Format for vector store
            cleaned = []
            for msg in result.messages:
                text = msg.get("text", "").strip()
                if not text:
                    continue
                
                cleaned.append({
                    "text": text,
                    "user": msg.get("user", "unknown"),
                    "ts": msg.get("ts", ""),
                    "thread_ts": msg.get("thread_ts", ""),
                    "is_gaprio_decision": msg.get("_is_gaprio_decision", False),
                })
            
            logger.debug(
                f"Fetched {len(messages)} messages, "
                f"filtered to {len(cleaned)} for indexing"
            )
            
            return cleaned
            
        except Exception as e:
            logger.error(f"Error fetching messages: {e}")
            return []
    
    async def _call_slack_api(self, method: str, **kwargs) -> dict:
        """
        Call a Slack API method.
        
        Handles both sync and async clients.
        """
        client_method = getattr(self.slack_client, method.replace(".", "_"), None)
        
        if client_method is None:
            # Try using api_call
            if hasattr(self.slack_client, "api_call"):
                return self.slack_client.api_call(method, **kwargs)
            raise ValueError(f"Cannot call Slack API method: {method}")
        
        # Check if async
        import asyncio
        if asyncio.iscoroutinefunction(client_method):
            return await client_method(**kwargs)
        else:
            return client_method(**kwargs)
    
    async def index_all_channels(
        self,
        channel_ids: list[str] | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        """
        Index multiple channels.
        
        If no channel IDs provided, attempts to auto-discover
        channels the bot is a member of.
        
        Args:
            channel_ids: List of channels to index (or auto-discover)
            force: Force indexing even if not due
            
        Returns:
            Dict with overall stats and per-channel results
        """
        if not self.slack_client:
            return {"success": False, "error": "No Slack client"}
        
        # Auto-discover channels if not provided
        if not channel_ids:
            channel_ids = await self._get_bot_channels()
        
        results = {
            "success": True,
            "total_indexed": 0,
            "channels": {},
        }
        
        for channel_id in channel_ids:
            result = await self.index_channel(channel_id, force=force)
            results["channels"][channel_id] = result
            
            if result.get("success") and not result.get("skipped"):
                results["total_indexed"] += result.get("count", 0)
        
        logger.info(f"Indexed {results['total_indexed']} messages across {len(channel_ids)} channels")
        return results
    
    async def _get_bot_channels(self) -> list[str]:
        """
        Get channels the bot is a member of.
        
        Returns:
            List of channel IDs
        """
        try:
            response = await self._call_slack_api(
                "conversations.list",
                types="public_channel,private_channel",
                exclude_archived=True,
            )
            
            if not response.get("ok"):
                return []
            
            # Filter to channels where bot is a member
            channels = [
                c["id"] for c in response.get("channels", [])
                if c.get("is_member", False)
            ]
            
            return channels
            
        except Exception as e:
            logger.error(f"Error getting channels: {e}")
            return []
    
    async def index_recent(
        self,
        channel_id: str,
        hours: int = 24,
    ) -> dict[str, Any]:
        """
        Index only recent messages from a channel.
        
        Useful for quick updates without full re-indexing.
        
        Args:
            channel_id: Channel to index
            hours: How many hours back to look
            
        Returns:
            Indexing stats
        """
        if not self.slack_client:
            return {"success": False, "error": "No Slack client"}
        
        try:
            # Calculate oldest timestamp
            oldest = datetime.now() - timedelta(hours=hours)
            oldest_ts = str(oldest.timestamp())
            
            response = await self._call_slack_api(
                "conversations.history",
                channel=channel_id,
                oldest=oldest_ts,
                limit=200,
            )
            
            if not response.get("ok"):
                return {"success": False, "error": response.get("error")}
            
            messages = response.get("messages", [])
            
            # Clean and add to vector store
            cleaned = []
            for msg in messages:
                if msg.get("subtype"):
                    continue
                text = msg.get("text", "").strip()
                if text:
                    cleaned.append({
                        "text": text,
                        "user": msg.get("user", "unknown"),
                        "ts": msg.get("ts", ""),
                    })
            
            count = self.vector_store.add_messages(channel_id, cleaned)
            
            return {
                "success": True,
                "count": count,
                "hours": hours,
            }
            
        except Exception as e:
            logger.error(f"Error indexing recent: {e}")
            return {"success": False, "error": str(e)}
    
    async def index_emails(self, emails: list[dict], user_id: str) -> int:
        """Index Gmail emails into ChromaDB for semantic search."""
        collection_name = f"emails_{user_id}"
        collection = self.vector_store.get_or_create_collection(collection_name)

        documents = []
        metadatas = []
        ids = []

        for email in emails:
            doc_text = (
                f"From: {email.get('from', '')}\n"
                f"Subject: {email.get('subject', '')}\n"
                f"Body: {email.get('snippet', email.get('body', ''))}"
            )
            documents.append(doc_text)
            metadatas.append({
                "from": email.get("from", ""),
                "subject": email.get("subject", ""),
                "date": email.get("date", ""),
                "email_id": email.get("id", ""),
            })
            ids.append(f"email_{email.get('id', '')}")

        if documents:
            collection.add(documents=documents, metadatas=metadatas, ids=ids)

        return len(documents)

    async def index_docs(self, docs: list[dict], user_id: str) -> int:
        """Index Google Docs into ChromaDB for semantic search."""
        collection_name = f"docs_{user_id}"
        collection = self.vector_store.get_or_create_collection(collection_name)

        documents = []
        metadatas = []
        ids = []

        for doc in docs:
            doc_text = f"Title: {doc.get('title', '')}\nContent: {doc.get('content', '')[:2000]}"
            documents.append(doc_text)
            metadatas.append({
                "title": doc.get("title", ""),
                "doc_id": doc.get("id", ""),
                "modified": doc.get("modifiedTime", ""),
            })
            ids.append(f"doc_{doc.get('id', '')}")

        if documents:
            collection.add(documents=documents, metadatas=metadatas, ids=ids)

        return len(documents)

    async def index_asana_tasks(self, tasks: list[dict], workspace_id: str) -> int:
        """Index Asana tasks into ChromaDB for semantic search."""
        collection_name = f"asana_{workspace_id}"
        collection = self.vector_store.get_or_create_collection(collection_name)

        documents = []
        metadatas = []
        ids = []

        for task in tasks:
            doc_text = f"Task: {task.get('name', '')}\nNotes: {task.get('notes', '')}"
            documents.append(doc_text)
            metadatas.append({
                "name": task.get("name", ""),
                "gid": task.get("gid", ""),
                "project": task.get("project_name", ""),
                "assignee": task.get("assignee", {}).get("name", ""),
            })
            ids.append(f"asana_{task.get('gid', '')}")

        if documents:
            collection.add(documents=documents, metadatas=metadatas, ids=ids)

        return len(documents)
