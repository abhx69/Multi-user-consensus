"""
Team Memory — Workspace-level learning.

Stores and retrieves action patterns for a team. When the team
repeatedly takes certain actions in response to certain triggers,
these patterns are stored and suggested to improve future planning.

Example: "Every time this team detects a production bug, they always
create an Asana task AND email the client."
"""

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


class TeamMemory:
    """Stores and retrieves workspace-level action patterns."""

    async def store_pattern(
        self,
        workspace_id: str,
        trigger: str,
        actions: list[str],
        success_rate: float = 1.0,
    ) -> int:
        """Store a new action pattern for a workspace."""
        from gaprio.db.connection import get_pool
        pool = await get_pool()

        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """INSERT INTO team_patterns
                       (workspace_id, trigger_description, actions, success_rate, usage_count, created_at)
                       VALUES (%s, %s, %s, %s, 1, NOW())
                       ON DUPLICATE KEY UPDATE
                           usage_count = usage_count + 1,
                           success_rate = (success_rate * usage_count + %s) / (usage_count + 1)""",
                    (workspace_id, trigger, json.dumps(actions), success_rate, success_rate),
                )
                await conn.commit()
                return cur.lastrowid

    async def search_patterns(
        self, workspace_id: str, query: str, limit: int = 5
    ) -> list[dict]:
        """Search for relevant patterns by trigger description."""
        from gaprio.db.connection import get_pool
        pool = await get_pool()

        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """SELECT id, trigger_description, actions, success_rate, usage_count
                       FROM team_patterns
                       WHERE workspace_id = %s AND trigger_description LIKE %s
                       ORDER BY usage_count DESC, success_rate DESC
                       LIMIT %s""",
                    (workspace_id, f"%{query}%", limit),
                )
                rows = await cur.fetchall()

        return [
            {
                "id": r[0],
                "trigger": r[1],
                "actions": json.loads(r[2]) if r[2] else [],
                "success_rate": float(r[3]),
                "usage_count": r[4],
            }
            for r in rows
        ]