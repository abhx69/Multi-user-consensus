"""
Heartbeat and scheduled task management.

Manages lightweight state for:
- Periodic check timestamps
- Scheduled reminders
- Cron-style recurring tasks

The heartbeat system allows the agent to:
- Track when it last performed certain actions
- Schedule future reminders
- Manage recurring tasks (e.g., daily summaries)
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal

from gaprio.config import settings

logger = logging.getLogger(__name__)


class HeartbeatManager:
    """
    Manages heartbeat state and scheduled tasks.
    
    The heartbeat system provides lightweight persistence for:
    - Last-check timestamps for various operations
    - Scheduled reminders (one-time)
    - Recurring tasks (cron-like)
    
    State is stored in heartbeat-state.json for simplicity
    and easy manual inspection.
    
    Usage:
        heartbeat = HeartbeatManager()
        
        # Record when we last did something
        heartbeat.record_check("index_channels")
        
        # Schedule a reminder
        heartbeat.add_reminder(
            reminder_id="email_follow_up",
            message="Send follow-up email",
            trigger_at=datetime.now() + timedelta(minutes=5),
            channel="D123456"
        )
        
        # Check for due items
        due = heartbeat.get_due_items()
    """
    
    def __init__(self, data_dir: Path | None = None):
        """
        Initialize the heartbeat manager.
        
        Args:
            data_dir: Directory for state file (default from settings)
        """
        self.data_dir = Path(data_dir or settings.data_dir)
        self.state_file = self.data_dir / "heartbeat-state.json"
        self._load_state()
        
        logger.info("HeartbeatManager initialized")
    
    def _load_state(self) -> None:
        """Load state from file or create default."""
        if self.state_file.exists():
            try:
                with open(self.state_file, "r") as f:
                    self.state = json.load(f)
            except json.JSONDecodeError:
                logger.warning("Invalid heartbeat state file, creating new")
                self.state = self._default_state()
        else:
            self.state = self._default_state()
            self._save_state()
    
    def _default_state(self) -> dict[str, Any]:
        """Create default state structure."""
        return {
            "last_checks": {},      # operation_name -> timestamp
            "reminders": [],        # list of reminder objects
            "recurring_tasks": [],  # list of recurring task objects
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "version": "1.0",
            },
        }
    
    def _save_state(self) -> None:
        """Save state to file."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, "w") as f:
            json.dump(self.state, f, indent=2, default=str)
    
    # =========================================================================
    # Last Check Timestamps
    # =========================================================================
    
    def record_check(self, operation: str) -> None:
        """
        Record that an operation was just performed.
        
        Args:
            operation: Name of the operation (e.g., "index_channels")
        """
        self.state["last_checks"][operation] = datetime.now().isoformat()
        self._save_state()
        logger.debug(f"Recorded check: {operation}")
    
    def get_last_check(self, operation: str) -> datetime | None:
        """
        Get when an operation was last performed.
        
        Args:
            operation: Name of the operation
            
        Returns:
            Datetime of last check, or None if never
        """
        timestamp = self.state["last_checks"].get(operation)
        if timestamp:
            return datetime.fromisoformat(timestamp)
        return None
    
    def should_run(
        self,
        operation: str,
        interval: timedelta,
    ) -> bool:
        """
        Check if an operation should run based on interval.
        
        Args:
            operation: Name of the operation
            interval: Minimum time between runs
            
        Returns:
            True if the operation should run
        """
        last_check = self.get_last_check(operation)
        if last_check is None:
            return True
        return datetime.now() - last_check >= interval
    
    # =========================================================================
    # Reminders (One-time)
    # =========================================================================
    
    def add_reminder(
        self,
        reminder_id: str,
        message: str,
        trigger_at: datetime,
        channel: str | None = None,
        user_id: str | None = None,
        metadata: dict | None = None,
    ) -> None:
        """
        Schedule a one-time reminder.
        
        Args:
            reminder_id: Unique identifier for the reminder
            message: Text to send when triggered
            trigger_at: When to trigger the reminder
            channel: Channel to send to (defaults to DM)
            user_id: User who created the reminder
            metadata: Additional data to include
        """
        # Remove existing reminder with same ID
        self.state["reminders"] = [
            r for r in self.state["reminders"]
            if r["id"] != reminder_id
        ]
        
        reminder = {
            "id": reminder_id,
            "message": message,
            "trigger_at": trigger_at.isoformat(),
            "channel": channel,
            "user_id": user_id,
            "created_at": datetime.now().isoformat(),
            "metadata": metadata or {},
            "status": "pending",
        }
        
        self.state["reminders"].append(reminder)
        self._save_state()
        
        logger.info(f"Added reminder: {reminder_id} at {trigger_at}")
    
    def get_due_reminders(self) -> list[dict]:
        """
        Get all reminders that are due now.
        
        Returns:
            List of reminder objects that should be triggered
        """
        now = datetime.now()
        due = []
        
        for reminder in self.state["reminders"]:
            if reminder["status"] != "pending":
                continue
            
            trigger_at = datetime.fromisoformat(reminder["trigger_at"])
            if trigger_at <= now:
                due.append(reminder)
        
        return due
    
    def mark_reminder_complete(self, reminder_id: str) -> None:
        """
        Mark a reminder as completed.
        
        Args:
            reminder_id: ID of the reminder to mark complete
        """
        for reminder in self.state["reminders"]:
            if reminder["id"] == reminder_id:
                reminder["status"] = "completed"
                reminder["completed_at"] = datetime.now().isoformat()
                break
        
        self._save_state()
        logger.info(f"Marked reminder complete: {reminder_id}")
    
    def cancel_reminder(self, reminder_id: str) -> bool:
        """
        Cancel a pending reminder.
        
        Args:
            reminder_id: ID of the reminder to cancel
            
        Returns:
            True if reminder was found and cancelled
        """
        for reminder in self.state["reminders"]:
            if reminder["id"] == reminder_id and reminder["status"] == "pending":
                reminder["status"] = "cancelled"
                self._save_state()
                logger.info(f"Cancelled reminder: {reminder_id}")
                return True
        return False
    
    def get_pending_reminders(self) -> list[dict]:
        """Get all pending reminders."""
        return [
            r for r in self.state["reminders"]
            if r["status"] == "pending"
        ]
    
    # =========================================================================
    # Recurring Tasks (Cron-like)
    # =========================================================================
    
    def add_recurring_task(
        self,
        task_id: str,
        name: str,
        schedule: str,
        action: str,
        action_params: dict | None = None,
        channel: str | None = None,
    ) -> None:
        """
        Add a recurring task.
        
        Args:
            task_id: Unique identifier for the task
            name: Human-readable name
            schedule: Cron-like schedule (e.g., "10:00" for daily at 10am)
            action: Action type to perform
            action_params: Parameters for the action
            channel: Target channel for the action
        """
        # Remove existing task with same ID
        self.state["recurring_tasks"] = [
            t for t in self.state["recurring_tasks"]
            if t["id"] != task_id
        ]
        
        task = {
            "id": task_id,
            "name": name,
            "schedule": schedule,
            "action": action,
            "action_params": action_params or {},
            "channel": channel,
            "created_at": datetime.now().isoformat(),
            "last_run": None,
            "enabled": True,
        }
        
        self.state["recurring_tasks"].append(task)
        self._save_state()
        
        logger.info(f"Added recurring task: {name} ({schedule})")
    
    def get_recurring_tasks(self, enabled_only: bool = True) -> list[dict]:
        """
        Get all recurring tasks.
        
        Args:
            enabled_only: Only return enabled tasks
            
        Returns:
            List of recurring task objects
        """
        tasks = self.state["recurring_tasks"]
        if enabled_only:
            tasks = [t for t in tasks if t.get("enabled", True)]
        return tasks
    
    def update_recurring_task_run(self, task_id: str) -> None:
        """
        Record that a recurring task was just run.
        
        Args:
            task_id: ID of the task that was run
        """
        for task in self.state["recurring_tasks"]:
            if task["id"] == task_id:
                task["last_run"] = datetime.now().isoformat()
                break
        
        self._save_state()
    
    def disable_recurring_task(self, task_id: str) -> bool:
        """
        Disable a recurring task.
        
        Args:
            task_id: ID of the task to disable
            
        Returns:
            True if task was found and disabled
        """
        for task in self.state["recurring_tasks"]:
            if task["id"] == task_id:
                task["enabled"] = False
                self._save_state()
                logger.info(f"Disabled recurring task: {task_id}")
                return True
        return False
    
    # =========================================================================
    # Combined
    # =========================================================================
    
    def get_due_items(self) -> dict[str, list]:
        """
        Get all items that are due to be processed.
        
        Returns:
            Dict with "reminders" and "tasks" lists
        """
        return {
            "reminders": self.get_due_reminders(),
            "tasks": self._get_due_recurring_tasks(),
        }
    
    def _get_due_recurring_tasks(self) -> list[dict]:
        """
        Check which recurring tasks should run.
        
        This is a simple implementation that checks time-of-day.
        For production, consider using APScheduler or similar.
        """
        now = datetime.now()
        current_time = now.strftime("%H:%M")
        current_hour = now.hour
        
        due = []
        for task in self.get_recurring_tasks(enabled_only=True):
            schedule = task.get("schedule", "")
            
            # Simple time matching (e.g., "10:00")
            if ":" in schedule:
                scheduled_hour, scheduled_min = schedule.split(":")
                if int(scheduled_hour) == current_hour:
                    # Check if it was already run today
                    last_run = task.get("last_run")
                    if last_run:
                        last_run_date = datetime.fromisoformat(last_run).date()
                        if last_run_date == now.date():
                            continue
                    due.append(task)
        
        return due
    
    def cleanup_old_reminders(self, days: int = 30) -> int:
        """
        Remove completed/cancelled reminders older than N days.
        
        Args:
            days: Age threshold for cleanup
            
        Returns:
            Number of reminders removed
        """
        cutoff = datetime.now() - timedelta(days=days)
        original_count = len(self.state["reminders"])
        
        self.state["reminders"] = [
            r for r in self.state["reminders"]
            if r["status"] == "pending" or
            datetime.fromisoformat(r.get("completed_at", r["created_at"])) > cutoff
        ]
        
        removed = original_count - len(self.state["reminders"])
        if removed > 0:
            self._save_state()
            logger.info(f"Cleaned up {removed} old reminders")
        
        return removed
