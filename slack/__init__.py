"""
Slack module - Slack Bolt app and event handlers.

Components:
- app.py: Slack Bolt app initialization with Socket Mode
- handlers.py: Message and event handlers for bot interactions
"""

from gaprio.slack.app import create_slack_app, create_socket_mode_handler, get_slack_client
from gaprio.slack.handlers import register_handlers, SlackHandlers

__all__ = [
    "create_slack_app",
    "create_socket_mode_handler",
    "get_slack_client",
    "register_handlers",
    "SlackHandlers",
]

