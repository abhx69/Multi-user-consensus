"""
GaprioState — Shared state object for the LangGraph pipeline.

This TypedDict flows through all 7 nodes. Each node reads
from previous fields and writes to its own output fields.
"""

from typing import TypedDict, Any


class GaprioState(TypedDict, total=False):
    # ── Input (set at pipeline start) ──
    workspace_id: str
    source_platform: str          # "slack" | "gmail" | "asana"
    raw_messages: list[dict]      # all messages/emails/events to process
    participants: list[str]       # user IDs of everyone in the thread

    # ── Node 2: Work Detection output ──
    work_detected: bool
    detected_items: list[dict]    # [{type, title, confidence, assignees}]
    detection_reasoning: str

    # ── Node 3: Consensus Resolution output ──
    participant_intents: dict[str, str]   # user_id → their intent
    conflicts: list[dict]                 # [{users, conflicting_requests, resolution}]
    consensus_output: dict                # unified view of what needs to happen

    # ── Node 4: Orchestration Planning output ──
    action_plan: list[dict]       # [{tool, params, rationale, priority}]
    plan_summary: str             # human-readable summary
    memory_context: str           # knowledge graph + memory context

    # ── Node 5: Human Approval ──
    approval_status: str          # "pending" | "approved" | "rejected" | "edited"
    edit_instructions: str
    approved_by: str              # user_id who approved

    # ── Node 6: Execution output ──
    execution_results: list[dict] # [{tool, success, result, error}]

    # ── Node 7: Memory Update output ──
    memory_updates: list[dict]

    # ── Control ──
    error: str | None
    thread_id: str