"""
WebSocket connection manager for real-time push notifications.

Events emitted:
  new_suggestion:   {action_id, tool, description, confidence, source}
  action_executed:  {action_id, result}
  action_rejected:  {action_id}
  pipeline_update:  {thread_id, stage, status}
"""

import json
import logging
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)


class ConnectionManager:
    """
    Manages active WebSocket connections per user.

    Usage:
        manager = ConnectionManager()

        # In your WebSocket endpoint:
        await manager.connect(user_id, websocket)

        # From anywhere in the app:
        await manager.emit_to_user(user_id, "new_suggestion", data)
    """

    def __init__(self):
        # user_id → list of active WebSocket connections
        self._connections: dict[str, list[WebSocket]] = {}

    async def connect(self, user_id: str, websocket: WebSocket):
        """Accept and register a new WebSocket connection."""
        await websocket.accept()
        if user_id not in self._connections:
            self._connections[user_id] = []
        self._connections[user_id].append(websocket)
        logger.info(f"WebSocket connected: user={user_id} (total: {len(self._connections[user_id])})")

    def disconnect(self, user_id: str, websocket: WebSocket):
        """Remove a disconnected WebSocket."""
        if user_id in self._connections:
            self._connections[user_id] = [
                ws for ws in self._connections[user_id] if ws != websocket
            ]
            if not self._connections[user_id]:
                del self._connections[user_id]
        logger.info(f"WebSocket disconnected: user={user_id}")

    async def emit_to_user(self, user_id: str, event_type: str, data: Any):
        """Send an event to all connections for a specific user."""
        if user_id not in self._connections:
            return

        message = json.dumps({"type": event_type, "data": data})
        dead_connections = []

        for ws in self._connections[user_id]:
            try:
                await ws.send_text(message)
            except Exception:
                dead_connections.append(ws)

        # Clean up dead connections
        for ws in dead_connections:
            self.disconnect(user_id, ws)

    async def broadcast(self, event_type: str, data: Any):
        """Send an event to ALL connected users."""
        for user_id in list(self._connections.keys()):
            await self.emit_to_user(user_id, event_type, data)


# Global instance
_manager = ConnectionManager()


async def emit_to_user(user_id: str, event_type: str, data: Any):
    """Convenience function — emit an event to a user from anywhere."""
    await _manager.emit_to_user(user_id, event_type, data)


def get_manager() -> ConnectionManager:
    """Get the global ConnectionManager instance."""
    return _manager


async def websocket_endpoint(websocket: WebSocket, user_id: str):
    """
    WebSocket endpoint handler.

    Register this in your FastAPI app:
        @app.websocket("/ws/{user_id}")
        async def ws_route(websocket: WebSocket, user_id: str):
            await websocket_endpoint(websocket, user_id)
    """
    await _manager.connect(user_id, websocket)
    try:
        while True:
            # Keep connection alive, listen for client messages
            data = await websocket.receive_text()
            # Client can send ping/pong or commands
            if data == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
    except WebSocketDisconnect:
        _manager.disconnect(user_id, websocket)
    except Exception as e:
        logger.error(f"WebSocket error for user {user_id}: {e}")
        _manager.disconnect(user_id, websocket)