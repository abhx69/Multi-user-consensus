"""
Knowledge Graph database queries.

CRUD operations for nodes and edges in the knowledge graph.
Uses aiomysql connection pool for async database access.
"""

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


async def upsert_node(
    workspace_id: str,
    node_type: str,
    external_id: str,
    title: str,
    summary: str,
    metadata: dict = None,
) -> int:
    """Create or update a knowledge graph node. Returns node ID."""
    from gaprio.db.connection import get_pool
    pool = await get_pool()

    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            # Check if node already exists
            await cur.execute(
                "SELECT id FROM knowledge_graph_nodes WHERE workspace_id = %s AND external_id = %s",
                (workspace_id, external_id),
            )
            existing = await cur.fetchone()

            if existing:
                await cur.execute(
                    """UPDATE knowledge_graph_nodes
                       SET title = %s, summary = %s, metadata = %s, updated_at = NOW()
                       WHERE id = %s""",
                    (title, summary, json.dumps(metadata or {}), existing[0]),
                )
                await conn.commit()
                return existing[0]
            else:
                await cur.execute(
                    """INSERT INTO knowledge_graph_nodes
                       (workspace_id, node_type, external_id, title, summary, metadata, created_at, updated_at)
                       VALUES (%s, %s, %s, %s, %s, %s, NOW(), NOW())""",
                    (workspace_id, node_type, external_id, title, summary, json.dumps(metadata or {})),
                )
                await conn.commit()
                return cur.lastrowid


async def upsert_edge(
    source_id: int,
    target_id: int,
    relationship_type: str,
    weight: float = 1.0,
    metadata: dict = None,
) -> int:
    """Create or update a knowledge graph edge. Returns edge ID."""
    from gaprio.db.connection import get_pool
    pool = await get_pool()

    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """SELECT id FROM knowledge_graph_edges
                   WHERE source_node_id = %s AND target_node_id = %s AND relationship_type = %s""",
                (source_id, target_id, relationship_type),
            )
            existing = await cur.fetchone()

            if existing:
                await cur.execute(
                    "UPDATE knowledge_graph_edges SET weight = %s, metadata = %s WHERE id = %s",
                    (weight, json.dumps(metadata or {}), existing[0]),
                )
                await conn.commit()
                return existing[0]
            else:
                await cur.execute(
                    """INSERT INTO knowledge_graph_edges
                       (source_node_id, target_node_id, relationship_type, weight, metadata, created_at)
                       VALUES (%s, %s, %s, %s, %s, NOW())""",
                    (source_id, target_id, relationship_type, weight, json.dumps(metadata or {})),
                )
                await conn.commit()
                return cur.lastrowid


async def get_related_nodes(node_id: int, max_hops: int = 2, node_type_filter: str = None) -> list[dict]:
    """Get nodes connected to a given node, up to max_hops away."""
    from gaprio.db.connection import get_pool
    pool = await get_pool()

    results = []
    visited = {node_id}
    current_ids = [node_id]

    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            for hop in range(max_hops):
                if not current_ids:
                    break

                placeholders = ",".join(["%s"] * len(current_ids))

                # Get neighbors (both directions)
                query = f"""
                    SELECT DISTINCT n.id, n.node_type, n.external_id, n.title, n.summary, n.metadata,
                           e.relationship_type, e.weight
                    FROM knowledge_graph_edges e
                    JOIN knowledge_graph_nodes n ON (
                        (e.target_node_id = n.id AND e.source_node_id IN ({placeholders}))
                        OR
                        (e.source_node_id = n.id AND e.target_node_id IN ({placeholders}))
                    )
                """
                params = current_ids + current_ids

                if node_type_filter:
                    query += " WHERE n.node_type = %s"
                    params.append(node_type_filter)

                await cur.execute(query, params)
                rows = await cur.fetchall()

                next_ids = []
                for row in rows:
                    nid = row[0]
                    if nid not in visited:
                        visited.add(nid)
                        next_ids.append(nid)
                        results.append({
                            "id": nid,
                            "node_type": row[1],
                            "external_id": row[2],
                            "title": row[3],
                            "summary": row[4],
                            "metadata": json.loads(row[5]) if row[5] else {},
                            "relationship": row[6],
                            "weight": float(row[7]),
                            "hop": hop + 1,
                        })

                current_ids = next_ids

    return results


async def search_nodes(
    workspace_id: str, query: str, node_type: str = None, limit: int = 10
) -> list[dict]:
    """Search knowledge graph nodes by title/summary text."""
    from gaprio.db.connection import get_pool
    pool = await get_pool()

    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            sql = """SELECT id, node_type, external_id, title, summary, metadata
                     FROM knowledge_graph_nodes
                     WHERE workspace_id = %s AND (title LIKE %s OR summary LIKE %s)"""
            params = [workspace_id, f"%{query}%", f"%{query}%"]

            if node_type:
                sql += " AND node_type = %s"
                params.append(node_type)

            sql += " LIMIT %s"
            params.append(limit)

            await cur.execute(sql, params)
            rows = await cur.fetchall()

    return [
        {
            "id": r[0], "node_type": r[1], "external_id": r[2],
            "title": r[3], "summary": r[4],
            "metadata": json.loads(r[5]) if r[5] else {},
        }
        for r in rows
    ]


async def get_person_context(workspace_id: str, person_external_id: str) -> dict:
    """Get all tasks, docs, conversations linked to a person."""
    from gaprio.db.connection import get_pool
    pool = await get_pool()

    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            # Find the person node
            await cur.execute(
                "SELECT id, title FROM knowledge_graph_nodes WHERE workspace_id = %s AND external_id = %s",
                (workspace_id, person_external_id),
            )
            person = await cur.fetchone()

            if not person:
                return {"person": None, "related": []}

    related = await get_related_nodes(person[0], max_hops=2)

    return {
        "person": {"id": person[0], "name": person[1]},
        "tasks": [r for r in related if r["node_type"] == "task"],
        "documents": [r for r in related if r["node_type"] == "document"],
        "conversations": [r for r in related if r["node_type"] == "conversation"],
        "emails": [r for r in related if r["node_type"] == "email"],
    }