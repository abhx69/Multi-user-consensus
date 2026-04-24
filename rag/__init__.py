"""
RAG module - Retrieval Augmented Generation for Slack channel knowledge.

Components:
- vector_store.py: ChromaDB integration for embedding storage
- indexer.py: Slack channel message indexing (past 200 messages)
- retriever.py: Query-based context retrieval for agent
"""

from gaprio.rag.vector_store import VectorStore
from gaprio.rag.indexer import SlackIndexer
from gaprio.rag.retriever import Retriever

__all__ = ["VectorStore", "SlackIndexer", "Retriever"]
