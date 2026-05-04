"""
Improved analyze-context route with DB storage,
deduplication, confidence scoring, and WebSocket push.
"""

import hashlib
import json
import logging
import re
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter()


class AnalyzeContextRequest(BaseModel):
    user_id: str | int = Field(..., description="User identifier")
    platform: str = Field(..., description="Source: slack, gmail, google_docs")
    channel_id: str = Field(default="", description="Channel/context identifier")
    context: str = Field(..., description="The raw content to analyze")
    metadata: dict[str, Any] = Field(default_factory=dict)


def _compute_content_hash(platform: str, context: str) -> str:
    """Generate a SHA-256 hash for deduplication."""
    raw = f"{platform}:{context[:500]}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


@router.post("/analyze-context")
async def analyze_context_improved(request: AnalyzeContextRequest) -> dict[str, Any]:
    """
    Analyze platform context → suggest actions → store in DB → push via WebSocket.

    Improvements over original:
      1. Content-hash deduplication (skip if already analyzed)
      2. Suggestions stored in MySQL suggested_actions table
      3. Confidence score per suggestion
      4. Source reference per suggestion
      5. WebSocket push after storing
      6. Smart truncation (last 2000 + first 500 chars)
    """
    from gaprio.db_tokens import set_current_user_id
    from gaprio.agent.prompts import MONITORING_PROMPT

    # Access globals from main app
    from gaprio.api import _agent
    if not _agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    set_current_user_id(str(request.user_id))

    # --- Step 1: Deduplication check ---
    content_hash = _compute_content_hash(request.platform, request.context)

    from gaprio.db.connection import get_pool
    pool = await get_pool()
    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT id FROM suggested_actions WHERE content_hash = %s AND status = 'pending'",
                (content_hash,)
            )
            if await cur.fetchone():
                logger.info(f"Skipping duplicate context (hash={content_hash[:12]}...)")
                return {"suggestions": [], "skipped": True, "reason": "duplicate"}

    # --- Step 2: Smart truncation ---
    ctx = request.context
    if len(ctx) > 2500:
        ctx = ctx[:500] + "\n...[truncated]...\n" + ctx[-2000:]

    # --- Step 3: Build prompt and call LLM ---
    tools_desc = "\n".join([
        f"- **{name}**: {tool.description}\n  Parameters: {tool.parameters}"
        for name, tool in _agent.tools.items()
    ])

    prompt = MONITORING_PROMPT.format(
        tools_description=tools_desc,
        platform=request.platform,
        channel=request.channel_id or "unknown",
        context=ctx,
    )

    raw_response = await _agent.llm.generate(prompt)

    # --- Step 4: Parse response ---
    cleaned = raw_response.strip()
    cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r'```\s*$', '', cleaned, flags=re.MULTILINE)
    cleaned = cleaned.strip()

    suggestions = []
    json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group())
            suggestions = parsed.get("suggestions", [])
        except json.JSONDecodeError:
            logger.warning("Failed to parse suggestions JSON")

    # --- Step 5: Validate, store in DB, push via WebSocket ---
    valid_suggestions = []
    for s in suggestions:
        tool_name = s.get("tool", "")
        if tool_name not in _agent.tools:
            continue

        suggestion = {
            "tool": tool_name,
            "params": s.get("params", s.get("parameters", {})),
            "description": s.get("description", f"Execute {tool_name}"),
            "confidence": min(1.0, max(0.0, float(s.get("confidence", 0.7)))),
            "urgency": s.get("urgency", "medium"),
            "source_reference": s.get("source_reference", ""),
        }

        # Store in MySQL
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """INSERT INTO suggested_actions
                       (user_id, workspace_id, source_platform, source_content_hash,
                        tool_name, tool_params, description, confidence_score,
                        urgency, source_reference, status, content_hash, created_at)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'pending', %s, NOW())""",
                    (
                        str(request.user_id),
                        request.metadata.get("workspace_id", "default"),
                        request.platform,
                        content_hash,
                        suggestion["tool"],
                        json.dumps(suggestion["params"]),
                        suggestion["description"],
                        suggestion["confidence"],
                        suggestion["urgency"],
                        suggestion["source_reference"],
                        content_hash,
                    )
                )
                action_id = cur.lastrowid
                await conn.commit()

        suggestion["action_id"] = action_id
        valid_suggestions.append(suggestion)

        # Push via WebSocket
        try:
            from gaprio.api.websocket import emit_to_user
            await emit_to_user(str(request.user_id), "new_suggestion", suggestion)
        except Exception as ws_err:
            logger.warning(f"WebSocket push failed: {ws_err}")

    logger.info(f"Stored {len(valid_suggestions)} suggestions from {request.platform}")
    return {"suggestions": valid_suggestions}