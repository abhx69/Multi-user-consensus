"""
Celery background tasks for proactive monitoring.

These tasks poll Slack, Gmail, and Docs on a schedule,
feed content to /analyze-context, and store suggestions.
"""

import logging
import httpx
from celery import Celery

logger = logging.getLogger(__name__)

# Initialize Celery with Redis broker
celery_app = Celery(
    "gaprio_tasks",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/1",
)

# Configuration
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
)

AI_ENGINE_URL = "http://localhost:8000"


@celery_app.task(bind=True, max_retries=3)
def poll_slack_for_suggestions(self):
    """
    Poll Slack channels for new messages and analyze them.
    Runs every 2 minutes via Celery Beat.

    Flow:
      1. Get all active workspaces from DB
      2. For each workspace with Slack connected:
         fetch last 20 messages from monitored channels
      3. Format as context string
      4. POST to /analyze-context
      5. Suggestions auto-stored + pushed via WebSocket
    """
    import asyncio

    async def _poll():
        from gaprio.db.connection import get_pool
        pool = await get_pool()

        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                # Get workspaces with active Slack connections
                await cur.execute(
                    """SELECT DISTINCT u.id as user_id, c.access_token, c.workspace_id
                       FROM connections c
                       JOIN users u ON c.user_id = u.id
                       WHERE c.provider = 'slack' AND c.access_token IS NOT NULL"""
                )
                connections = await cur.fetchall()

        async with httpx.AsyncClient(timeout=30) as client:
            for user_id, token, workspace_id in connections:
                try:
                    # Use Slack API to fetch recent messages
                    slack_resp = await client.get(
                        "https://slack.com/api/conversations.list",
                        headers={"Authorization": f"Bearer {token}"},
                        params={"types": "public_channel", "limit": 5}
                    )
                    channels = slack_resp.json().get("channels", [])

                    for channel in channels[:3]:  # Top 3 active channels
                        ch_id = channel["id"]
                        ch_name = channel.get("name", ch_id)

                        history_resp = await client.get(
                            "https://slack.com/api/conversations.history",
                            headers={"Authorization": f"Bearer {token}"},
                            params={"channel": ch_id, "limit": 20}
                        )
                        messages = history_resp.json().get("messages", [])

                        if not messages:
                            continue

                        # Build context string
                        context_parts = [f"Channel: #{ch_name}\n"]
                        for msg in reversed(messages):
                            user = msg.get("user", "unknown")
                            text = msg.get("text", "")
                            context_parts.append(f"[{user}]: {text}")

                        context = "\n".join(context_parts)

                        # Analyze context
                        await client.post(
                            f"{AI_ENGINE_URL}/analyze-context",
                            json={
                                "user_id": user_id,
                                "platform": "slack",
                                "channel_id": ch_id,
                                "context": context,
                                "metadata": {"workspace_id": workspace_id},
                            }
                        )

                except Exception as e:
                    logger.error(f"Slack poll failed for user {user_id}: {e}")

    asyncio.run(_poll())


@celery_app.task(bind=True, max_retries=3)
def poll_gmail_for_suggestions(self):
    """
    Poll Gmail for unread emails and analyze them.
    Runs every 5 minutes via Celery Beat.
    """
    import asyncio

    async def _poll():
        from gaprio.db.connection import get_pool
        pool = await get_pool()

        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """SELECT DISTINCT u.id, c.access_token, c.refresh_token, c.workspace_id
                       FROM connections c
                       JOIN users u ON c.user_id = u.id
                       WHERE c.provider = 'google' AND c.refresh_token IS NOT NULL"""
                )
                connections = await cur.fetchall()

        async with httpx.AsyncClient(timeout=30) as client:
            for user_id, access_token, refresh_token, workspace_id in connections:
                try:
                    # List recent unread emails
                    resp = await client.get(
                        "https://gmail.googleapis.com/gmail/v1/users/me/messages",
                        headers={"Authorization": f"Bearer {access_token}"},
                        params={"q": "is:unread", "maxResults": 5}
                    )
                    messages = resp.json().get("messages", [])

                    if not messages:
                        continue

                    # Fetch snippets for each email
                    context_parts = ["Recent unread emails:\n"]
                    for msg_ref in messages[:5]:
                        msg_resp = await client.get(
                            f"https://gmail.googleapis.com/gmail/v1/users/me/messages/{msg_ref['id']}",
                            headers={"Authorization": f"Bearer {access_token}"},
                            params={"format": "metadata", "metadataHeaders": ["From", "Subject"]}
                        )
                        msg_data = msg_resp.json()
                        headers = {h["name"]: h["value"] for h in msg_data.get("payload", {}).get("headers", [])}
                        snippet = msg_data.get("snippet", "")
                        context_parts.append(
                            f"From: {headers.get('From', 'unknown')}\n"
                            f"Subject: {headers.get('Subject', 'No subject')}\n"
                            f"Preview: {snippet}\n"
                        )

                    context = "\n".join(context_parts)

                    await client.post(
                        f"{AI_ENGINE_URL}/analyze-context",
                        json={
                            "user_id": user_id,
                            "platform": "gmail",
                            "context": context,
                            "metadata": {"workspace_id": workspace_id},
                        }
                    )

                except Exception as e:
                    logger.error(f"Gmail poll failed for user {user_id}: {e}")

    asyncio.run(_poll())


@celery_app.task(bind=True, max_retries=3)
def poll_docs_for_suggestions(self):
    """
    Poll Google Docs for recently modified documents.
    Runs every 15 minutes via Celery Beat.
    """
    import asyncio
    from datetime import datetime, timedelta

    async def _poll():
        from gaprio.db.connection import get_pool
        pool = await get_pool()

        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """SELECT DISTINCT u.id, c.access_token, c.workspace_id
                       FROM connections c JOIN users u ON c.user_id = u.id
                       WHERE c.provider = 'google' AND c.access_token IS NOT NULL"""
                )
                connections = await cur.fetchall()

        two_hours_ago = (datetime.utcnow() - timedelta(hours=2)).isoformat() + "Z"

        async with httpx.AsyncClient(timeout=30) as client:
            for user_id, access_token, workspace_id in connections:
                try:
                    resp = await client.get(
                        "https://www.googleapis.com/drive/v3/files",
                        headers={"Authorization": f"Bearer {access_token}"},
                        params={
                            "q": f"mimeType='application/vnd.google-apps.document' and modifiedTime > '{two_hours_ago}'",
                            "fields": "files(id,name,modifiedTime,lastModifyingUser)",
                            "pageSize": 5,
                        }
                    )
                    files = resp.json().get("files", [])

                    if not files:
                        continue

                    context_parts = ["Recently modified Google Docs:\n"]
                    for f in files:
                        editor = f.get("lastModifyingUser", {}).get("displayName", "unknown")
                        context_parts.append(
                            f"Doc: {f.get('name', 'Untitled')}\n"
                            f"Last editor: {editor}\n"
                            f"Modified: {f.get('modifiedTime', 'unknown')}\n"
                        )

                    context = "\n".join(context_parts)

                    await client.post(
                        f"{AI_ENGINE_URL}/analyze-context",
                        json={
                            "user_id": user_id,
                            "platform": "google_docs",
                            "context": context,
                            "metadata": {"workspace_id": workspace_id},
                        }
                    )
                except Exception as e:
                    logger.error(f"Docs poll failed for user {user_id}: {e}")

    asyncio.run(_poll())
