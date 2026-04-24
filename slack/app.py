"""
Slack Bolt app initialization.

Creates and configures the Slack app with Socket Mode for real-time messaging.
The app handles incoming events and routes them to the appropriate handlers.
"""

import logging
from typing import Any

from slack_bolt.async_app import AsyncApp
from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler

from gaprio.config import settings

logger = logging.getLogger(__name__)


def create_slack_app() -> AsyncApp:
    """
    Create and configure the async Slack Bolt app.
    
    Returns:
        Configured Slack Bolt AsyncApp instance
    """
    # Validate required settings
    if not settings.slack_bot_token:
        raise ValueError("SLACK_BOT_TOKEN is required")
    
    if not settings.slack_signing_secret:
        raise ValueError("SLACK_SIGNING_SECRET is required")
    
    # Create the async app
    app = AsyncApp(
        token=settings.slack_bot_token,
        signing_secret=settings.slack_signing_secret,
    )
    
    logger.info("Async Slack app created")
    return app


def create_socket_mode_handler(app: AsyncApp) -> AsyncSocketModeHandler:
    """
    Create an async Socket Mode handler for the app.
    
    Socket Mode allows the bot to receive events without exposing
    a public HTTP endpoint.
    
    Args:
        app: Slack Bolt AsyncApp instance
        
    Returns:
        AsyncSocketModeHandler instance
    """
    if not settings.slack_app_token:
        raise ValueError("SLACK_APP_TOKEN is required for Socket Mode")
    
    handler = AsyncSocketModeHandler(app, settings.slack_app_token)
    
    logger.info("Async Socket Mode handler created")
    return handler


def get_slack_client(app: AsyncApp) -> Any:
    """
    Get the Slack Web API client from the app.
    
    Args:
        app: Slack Bolt AsyncApp
        
    Returns:
        AsyncWebClient instance
    """
    return app.client
