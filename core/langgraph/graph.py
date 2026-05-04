"""
LangGraph Pipeline Assembly — connects all 7 nodes into a runnable graph.

Exports:
  run_pipeline(messages, workspace_id) → thread_id
  resume_pipeline(thread_id, decision) → results
"""

import logging
import uuid
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from gaprio.core.langgraph.state import GaprioState
from gaprio.core.langgraph.nodes import (
    ingest_messages_node,
    work_detection_node,
    consensus_resolution_node,
    orchestration_planning_node,
    human_approval_node,
    action_execution_node,
    memory_update_node,
)

logger = logging.getLogger(__name__)


def _should_continue_after_detection(state: GaprioState) -> str:
    """After work detection: continue if work found, else END."""
    if state.get("work_detected", False):
        return "consensus_resolution"
    return END


def _should_continue_after_approval(state: GaprioState) -> str:
    """After human approval: execute if approved, replan if edited, end if rejected."""
    status = state.get("approval_status", "rejected")
    if status == "approved":
        return "action_execution"
    elif status == "edited":
        return "orchestration_planning"  # Loop back to re-plan
    return END


def build_pipeline() -> StateGraph:
    """Build and compile the full 7-node LangGraph pipeline."""

    graph = StateGraph(GaprioState)

    # Add all 7 nodes
    graph.add_node("ingest_messages", ingest_messages_node)
    graph.add_node("work_detection", work_detection_node)
    graph.add_node("consensus_resolution", consensus_resolution_node)
    graph.add_node("orchestration_planning", orchestration_planning_node)
    graph.add_node("human_approval", human_approval_node)
    graph.add_node("action_execution", action_execution_node)
    graph.add_node("memory_update", memory_update_node)

    # Set entry point
    graph.add_edge(START, "ingest_messages")

    # Fixed edges
    graph.add_edge("ingest_messages", "work_detection")

    # Conditional: after detection, continue only if work found
    graph.add_conditional_edges(
        "work_detection",
        _should_continue_after_detection,
        {"consensus_resolution": "consensus_resolution", END: END},
    )

    graph.add_edge("consensus_resolution", "orchestration_planning")
    graph.add_edge("orchestration_planning", "human_approval")

    # Conditional: after approval, execute/replan/end
    graph.add_conditional_edges(
        "human_approval",
        _should_continue_after_approval,
        {
            "action_execution": "action_execution",
            "orchestration_planning": "orchestration_planning",
            END: END,
        },
    )

    graph.add_edge("action_execution", "memory_update")
    graph.add_edge("memory_update", END)

    return graph


# Compile with checkpointer (MemorySaver for dev, MySQL for production)
_checkpointer = MemorySaver()
_compiled_graph = build_pipeline().compile(checkpointer=_checkpointer)


async def run_pipeline(
    messages: list[dict],
    workspace_id: str,
    source_platform: str = "slack",
) -> str:
    """Start a new pipeline execution. Returns thread_id."""
    thread_id = str(uuid.uuid4())

    initial_state: GaprioState = {
        "workspace_id": workspace_id,
        "source_platform": source_platform,
        "raw_messages": messages,
        "participants": [],
        "thread_id": thread_id,
    }

    config = {"configurable": {"thread_id": thread_id}}

    logger.info(f"Starting pipeline {thread_id} with {len(messages)} messages")
    result = await _compiled_graph.ainvoke(initial_state, config=config)
    logger.info(f"Pipeline {thread_id} reached checkpoint (status: {result.get('approval_status', 'pending')})")

    return thread_id


async def resume_pipeline(thread_id: str, decision: dict) -> dict:
    """Resume a paused pipeline after human approval/rejection/edit."""
    config = {"configurable": {"thread_id": thread_id}}

    logger.info(f"Resuming pipeline {thread_id} with decision: {decision.get('status')}")
    result = await _compiled_graph.ainvoke(decision, config=config)

    return {
        "thread_id": thread_id,
        "approval_status": result.get("approval_status"),
        "execution_results": result.get("execution_results", []),
        "memory_updates": result.get("memory_updates", []),
    }