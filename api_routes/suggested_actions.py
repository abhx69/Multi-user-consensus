"""
Suggested Actions CRUD API.

Endpoints:
  GET  /suggested-actions/{user_id}         → list pending suggestions
  POST /suggested-actions/{id}/execute      → run a suggested action
  POST /suggested-actions/{id}/reject       → dismiss a suggestion
  PATCH /suggested-actions/{id}/edit        → update params
  GET  /suggested-actions/{user_id}/count   → badge count
"""

import json
import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/suggested-actions", tags=["Suggested Actions"])


class EditParams(BaseModel):
    new_params: dict[str, Any] = Field(..., description="Updated tool parameters")


@router.get("/{user_id}")
async def list_suggested_actions(user_id: str, limit: int = 20) -> dict[str, Any]:
    """Return all pending suggested actions for a user, newest first."""
    from gaprio.db.connection import get_pool
    pool = await get_pool()

    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """SELECT id, tool_name, tool_params, description,
                          confidence_score, urgency, source_platform,
                          source_reference, created_at
                   FROM suggested_actions
                   WHERE user_id = %s AND status = 'pending'
                   ORDER BY created_at DESC
                   LIMIT %s""",
                (user_id, limit)
            )
            rows = await cur.fetchall()

    actions = []
    for row in rows:
        actions.append({
            "id": row[0],
            "tool": row[1],
            "params": json.loads(row[2]) if row[2] else {},
            "description": row[3],
            "confidence": float(row[4]) if row[4] else 0.7,
            "urgency": row[5] or "medium",
            "source_platform": row[6],
            "source_reference": row[7] or "",
            "created_at": row[8].isoformat() if row[8] else None,
        })

    return {"actions": actions, "count": len(actions)}


@router.post("/{action_id}/execute")
async def execute_suggested_action(action_id: int) -> dict[str, Any]:
    """Execute a suggested action and update its status in DB."""
    from gaprio.db.connection import get_pool
    from gaprio.api import _agent
    from gaprio.db_tokens import set_current_user_id

    if not _agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    pool = await get_pool()

    # Fetch the action
    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT user_id, tool_name, tool_params FROM suggested_actions WHERE id = %s",
                (action_id,)
            )
            row = await cur.fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Action not found")

    user_id, tool_name, tool_params_json = row
    params = json.loads(tool_params_json) if tool_params_json else {}

    set_current_user_id(str(user_id))

    if tool_name not in _agent.tools:
        raise HTTPException(status_code=400, detail=f"Tool '{tool_name}' not available")

    # Execute the tool
    try:
        tool = _agent.tools[tool_name]
        result = await tool.handler(**params)
        result_data = result if isinstance(result, dict) else str(result)

        # Update DB
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """UPDATE suggested_actions
                       SET status = 'executed', executed_at = NOW(),
                           execution_result = %s
                       WHERE id = %s""",
                    (json.dumps(result_data) if isinstance(result_data, dict) else result_data, action_id)
                )
                await conn.commit()

        # Emit WebSocket event
        try:
            from gaprio.api.websocket import emit_to_user
            await emit_to_user(str(user_id), "action_executed", {
                "action_id": action_id, "result": result_data
            })
        except Exception:
            pass

        return {"success": True, "action_id": action_id, "result": result_data}

    except Exception as e:
        logger.error(f"Execute suggested action {action_id} failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{action_id}/reject")
async def reject_suggested_action(action_id: int) -> dict[str, Any]:
    """Dismiss a suggested action."""
    from gaprio.db.connection import get_pool
    pool = await get_pool()

    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "UPDATE suggested_actions SET status = 'rejected' WHERE id = %s",
                (action_id,)
            )
            await conn.commit()

            # Get user_id for WebSocket
            await cur.execute("SELECT user_id FROM suggested_actions WHERE id = %s", (action_id,))
            row = await cur.fetchone()

    if row:
        try:
            from gaprio.api.websocket import emit_to_user
            await emit_to_user(str(row[0]), "action_rejected", {"action_id": action_id})
        except Exception:
            pass

    return {"success": True, "action_id": action_id, "status": "rejected"}


@router.patch("/{action_id}/edit")
async def edit_suggested_action(action_id: int, body: EditParams) -> dict[str, Any]:
    """Update tool parameters for a suggested action (does NOT execute)."""
    from gaprio.db.connection import get_pool
    pool = await get_pool()

    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """UPDATE suggested_actions
                   SET tool_params = %s, status = 'pending'
                   WHERE id = %s""",
                (json.dumps(body.new_params), action_id)
            )
            await conn.commit()

            await cur.execute(
                "SELECT tool_name, tool_params, description FROM suggested_actions WHERE id = %s",
                (action_id,)
            )
            row = await cur.fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Action not found")

    return {
        "success": True,
        "action_id": action_id,
        "tool": row[0],
        "params": json.loads(row[1]) if row[1] else {},
        "description": row[2],
        "status": "pending",
    }


@router.get("/{user_id}/count")
async def count_pending_actions(user_id: str) -> dict[str, int]:
    """Return count of pending actions (for notification badge)."""
    from gaprio.db.connection import get_pool
    pool = await get_pool()

    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT COUNT(*) FROM suggested_actions WHERE user_id = %s AND status = 'pending'",
                (user_id,)
            )
            row = await cur.fetchone()

    return {"count": row[0] if row else 0}