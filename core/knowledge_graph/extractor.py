"""
Knowledge Graph Extractor — Builds graph from platform data.

Processes messages, emails, docs, and tasks to create nodes and
edges in the knowledge graph, enabling cross-platform context.
"""

import logging
import re
from typing import Any
from gaprio.db.knowledge_graph_queries import upsert_node, upsert_edge, search_nodes

logger = logging.getLogger(__name__)


class KnowledgeGraphExtractor:
    """Extracts entities and relationships from platform data."""

    async def extract_from_slack_message(
        self, msg: dict, channel_id: str, workspace_id: str
    ) -> list[int]:
        """Extract nodes/edges from a Slack message."""
        created_node_ids = []

        # Create conversation node for the channel
        conv_node_id = await upsert_node(
            workspace_id=workspace_id,
            node_type="conversation",
            external_id=f"slack:{channel_id}",
            title=f"Slack #{msg.get('channel_name', channel_id)}",
            summary=msg.get("text", "")[:200],
            metadata={"platform": "slack", "channel_id": channel_id},
        )
        created_node_ids.append(conv_node_id)

        # Create person node for the sender
        user_id = msg.get("user", "unknown")
        person_node_id = await upsert_node(
            workspace_id=workspace_id,
            node_type="person",
            external_id=f"slack_user:{user_id}",
            title=msg.get("user_name", user_id),
            summary="",
            metadata={"platform": "slack", "slack_id": user_id},
        )
        created_node_ids.append(person_node_id)

        # Edge: person → mentioned_in → conversation
        await upsert_edge(
            source_id=person_node_id,
            target_id=conv_node_id,
            relationship_type="mentioned_in",
            weight=1.0,
            metadata={"timestamp": msg.get("ts", "")},
        )

        # Detect task mentions in message text
        text = msg.get("text", "")
        task_patterns = [
            r"(?:create|add|make)\s+(?:a\s+)?task[:\s]+(.+?)(?:\.|$)",
            r"(?:TODO|todo|FIXME|fixme)[:\s]+(.+?)(?:\.|$)",
        ]
        for pattern in task_patterns:
            match = re.search(pattern, text)
            if match:
                task_title = match.group(1).strip()[:100]
                task_node_id = await upsert_node(
                    workspace_id=workspace_id,
                    node_type="task",
                    external_id=f"detected:{task_title[:50]}",
                    title=task_title,
                    summary=f"Detected from Slack message in #{channel_id}",
                    metadata={"source": "slack_detection", "channel_id": channel_id},
                )
                await upsert_edge(
                    source_id=task_node_id,
                    target_id=conv_node_id,
                    relationship_type="mentioned_in",
                    weight=0.8,
                    metadata={},
                )
                created_node_ids.append(task_node_id)

        return created_node_ids

    async def extract_from_email(
        self, email: dict, user_id: str, workspace_id: str
    ) -> list[int]:
        """Extract nodes/edges from an email."""
        created_node_ids = []

        # Create email node
        email_node_id = await upsert_node(
            workspace_id=workspace_id,
            node_type="email",
            external_id=f"gmail:{email.get('id', '')}",
            title=email.get("subject", "No subject"),
            summary=email.get("snippet", "")[:200],
            metadata={
                "from": email.get("from", ""),
                "to": email.get("to", []),
                "date": email.get("date", ""),
            },
        )
        created_node_ids.append(email_node_id)

        # Create person node for sender
        sender = email.get("from", "unknown")
        sender_node_id = await upsert_node(
            workspace_id=workspace_id,
            node_type="person",
            external_id=f"email:{sender}",
            title=sender.split("<")[0].strip() if "<" in sender else sender,
            summary="",
            metadata={"email": sender},
        )
        created_node_ids.append(sender_node_id)

        # Edge: person → sent_email → email
        await upsert_edge(
            source_id=sender_node_id,
            target_id=email_node_id,
            relationship_type="sent_email",
            weight=1.0,
            metadata={},
        )

        # Link to existing tasks if subject matches
        subject = email.get("subject", "")
        existing_tasks = await search_nodes(workspace_id, subject, node_type="task", limit=3)
        for task in existing_tasks:
            await upsert_edge(
                source_id=email_node_id,
                target_id=task["id"],
                relationship_type="related_to",
                weight=0.7,
                metadata={"match_type": "subject_similarity"},
            )

        return created_node_ids

    async def extract_from_doc_activity(
        self, doc: dict, editor: str, workspace_id: str
    ) -> list[int]:
        """Extract nodes/edges from Google Docs activity."""
        doc_node_id = await upsert_node(
            workspace_id=workspace_id,
            node_type="document",
            external_id=f"gdoc:{doc.get('id', '')}",
            title=doc.get("name", "Untitled Document"),
            summary=f"Last modified by {editor}",
            metadata={"doc_id": doc.get("id"), "modified_time": doc.get("modifiedTime")},
        )

        editor_node_id = await upsert_node(
            workspace_id=workspace_id,
            node_type="person",
            external_id=f"gdoc_user:{editor}",
            title=editor,
            summary="",
            metadata={"source": "google_docs"},
        )

        await upsert_edge(
            source_id=editor_node_id,
            target_id=doc_node_id,
            relationship_type="edited",
            weight=1.0,
            metadata={"modified_time": doc.get("modifiedTime", "")},
        )

        return [doc_node_id, editor_node_id]

    async def extract_from_asana_task(
        self, task: dict, workspace_id: str
    ) -> list[int]:
        """Extract nodes/edges from an Asana task."""
        task_node_id = await upsert_node(
            workspace_id=workspace_id,
            node_type="task",
            external_id=f"asana:{task.get('gid', '')}",
            title=task.get("name", "Untitled Task"),
            summary=task.get("notes", "")[:200],
            metadata={
                "project": task.get("project_name", ""),
                "due_on": task.get("due_on", ""),
                "completed": task.get("completed", False),
            },
        )

        created = [task_node_id]

        # Link assignee
        assignee = task.get("assignee", {})
        if assignee and assignee.get("name"):
            assignee_node_id = await upsert_node(
                workspace_id=workspace_id,
                node_type="person",
                external_id=f"asana_user:{assignee.get('gid', '')}",
                title=assignee["name"],
                summary="",
                metadata={"asana_gid": assignee.get("gid")},
            )
            await upsert_edge(
                source_id=assignee_node_id,
                target_id=task_node_id,
                relationship_type="assigned_to",
                weight=1.0,
                metadata={},
            )
            created.append(assignee_node_id)

        return created