"""Celery Beat schedule configuration."""

from celery.schedules import crontab
from tasks.monitoring import celery_app

celery_app.conf.beat_schedule = {
    "poll-slack-every-2-min": {
        "task": "tasks.monitoring.poll_slack_for_suggestions",
        "schedule": 120.0,  # every 2 minutes
    },
    "poll-gmail-every-5-min": {
        "task": "tasks.monitoring.poll_gmail_for_suggestions",
        "schedule": 300.0,  # every 5 minutes
    },
    "poll-docs-every-15-min": {
        "task": "tasks.monitoring.poll_docs_for_suggestions",
        "schedule": 900.0,  # every 15 minutes
    },
}