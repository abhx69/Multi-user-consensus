"""
LangGraph Pipeline Nodes — The 7 stages of Gaprio's consensus pipeline.

Each node is an async function: (state: GaprioState) -> dict
The returned dict is merged into the state automatically.
"""

import json
import logging
from typing import Any
from langgraph.types import interrupt
from gaprio.core.langgraph.state import GaprioState
from gaprio.core.llm.router import LLMRouter, TaskType

logger = logging.getLogger(__name__)
router = LLMRouter()


# ═══════════════════════════════════════════════════
# NODE 1: Ingest Messages (no LLM needed)
# ═══════════════════════════════════════════════════
async def ingest_messages_node(state: GaprioState) -> dict:
    """
    Normalize raw messages from any source into a standard format.
    Deduplicates messages and extracts participant list.
    """
    raw = state.get("raw_messages", [])
    seen_ids = set()
    cleaned = []
    participants = set()

    for msg in raw:
        msg_id = msg.get("id") or msg.get("ts") or msg.get("message_id", "")
        if msg_id in seen_ids:
            continue
        seen_ids.add(msg_id)

        cleaned.append({
            "id": msg_id,
            "user": msg.get("user") or msg.get("from") or "unknown",
            "text": msg.get("text") or msg.get("body") or msg.get("snippet", ""),
            "timestamp": msg.get("ts") or msg.get("date") or "",
            "platform": state.get("source_platform", "unknown"),
        })
        participants.add(msg.get("user") or msg.get("from") or "unknown")

    logger.info(f"Node 1: Ingested {len(cleaned)} messages from {len(participants)} participants")
    return {
        "raw_messages": cleaned,
        "participants": list(participants),
    }


# ═══════════════════════════════════════════════════
# NODE 2: Work Detection (LLM: Claude 3.5 Sonnet)
# ═══════════════════════════════════════════════════
async def work_detection_node(state: GaprioState) -> dict:
    """
    Read the conversation and detect implied work items.
    Uses few-shot examples to teach the LLM what "implied work" looks like.
    """
    messages = state.get("raw_messages", [])
    conversation = "\n".join([f"[{m['user']}]: {m['text']}" for m in messages])

    prompt = f"""You are Gaprio's work detection engine. Read this conversation and identify any implied work items — tasks, emails to send, meetings to schedule, documents to create.

CONVERSATION:
{conversation}

Respond with JSON:
{{
  "work_detected": true/false,
  "items": [
    {{
      "type": "task|email|meeting|document",
      "title": "Brief title",
      "description": "What needs to be done",
      "confidence": 0.0-1.0,
      "assignees": ["user_id or name"],
      "urgency": "low|medium|high"
    }}
  ],
  "reasoning": "Brief explanation of what you detected and why"
}}

Examples of implied work:
- "We should fix the login bug" → task with medium confidence
- "Can someone email the client?" → email with high confidence
- "Let's meet Thursday to discuss" → meeting with high confidence
- "I'll write up a doc about this" → document with medium confidence

If no work is detected, set work_detected to false and items to [].
Respond ONLY with valid JSON."""

    response = await router.generate(TaskType.WORK_DETECTION, prompt, temperature=0.2)

    try:
        data = json.loads(response.strip().strip("```json").strip("```"))
    except json.JSONDecodeError:
        import re
        match = re.search(r'\{.*\}', response, re.DOTALL)
        data = json.loads(match.group()) if match else {"work_detected": False, "items": [], "reasoning": "Parse error"}

    logger.info(f"Node 2: Detected {len(data.get('items', []))} work items")
    return {
        "work_detected": data.get("work_detected", False),
        "detected_items": data.get("items", []),
        "detection_reasoning": data.get("reasoning", ""),
    }


# ═══════════════════════════════════════════════════
# NODE 3: Consensus Resolution 🏆 DIFFERENTIATOR
# ═══════════════════════════════════════════════════
async def consensus_resolution_node(state: GaprioState) -> dict:
    """
    For GROUP conversations: map each participant's intent, find conflicts, synthesize.
    For single-user: just confirm intent.
    Uses chain-of-thought prompting for multi-step reasoning.
    """
    participants = state.get("participants", [])
    messages = state.get("raw_messages", [])
    detected_items = state.get("detected_items", [])
    conversation = "\n".join([f"[{m['user']}]: {m['text']}" for m in messages])

    prompt = f"""You are Gaprio's consensus resolution engine. You have detected the following work items from a group conversation:

PARTICIPANTS: {', '.join(participants)}

CONVERSATION:
{conversation}

DETECTED WORK ITEMS:
{json.dumps(detected_items, indent=2)}

Your task: Use chain-of-thought reasoning to:

Step 1: Identify each participant's intent (what do they want to happen?)
Step 2: Check for conflicts (do any requests contradict each other?)
Step 3: Synthesize a unified plan that satisfies ALL participants

Respond with JSON:
{{
  "participant_intents": {{
    "user_id": "what this person wants"
  }},
  "conflicts": [
    {{
      "users": ["user1", "user2"],
      "issue": "Description of conflict",
      "resolution": "How to resolve it"
    }}
  ],
  "consensus": {{
    "summary": "Unified understanding of what needs to happen",
    "actions_needed": ["action 1", "action 2"],
    "priority_order": ["most urgent first"]
  }}
}}

Respond ONLY with valid JSON."""

    response = await router.generate(TaskType.CONSENSUS, prompt, temperature=0.3)

    try:
        data = json.loads(response.strip().strip("```json").strip("```"))
    except json.JSONDecodeError:
        import re
        match = re.search(r'\{.*\}', response, re.DOTALL)
        data = json.loads(match.group()) if match else {}

    logger.info(f"Node 3: Resolved consensus for {len(participants)} participants")
    return {
        "participant_intents": data.get("participant_intents", {}),
        "conflicts": data.get("conflicts", []),
        "consensus_output": data.get("consensus", {}),
    }


# ═══════════════════════════════════════════════════
# NODE 4: Orchestration Planning (LLM: GPT-4o)
# ═══════════════════════════════════════════════════
async def orchestration_planning_node(state: GaprioState) -> dict:
    """
    Take consensus + knowledge graph + memory → generate action plan with MCP tools.
    """
    consensus = state.get("consensus_output", {})
    workspace_id = state.get("workspace_id", "")

    # Query knowledge graph for additional context
    kg_context = ""
    try:
        from gaprio.db.knowledge_graph_queries import search_nodes
        topic = consensus.get("summary", "")[:100]
        if topic:
            nodes = await search_nodes(workspace_id, topic, limit=5)
            if nodes:
                kg_context = "Knowledge Graph Context:\n" + "\n".join([
                    f"- [{n['node_type']}] {n['title']}: {n['summary']}" for n in nodes
                ])
    except Exception as e:
        logger.warning(f"Knowledge graph query failed: {e}")

    # Get available MCP tools
    from gaprio.api import _agent
    tools_desc = ""
    if _agent:
        tools_desc = "\n".join([
            f"- {name}: {tool.description}" for name, tool in _agent.tools.items()
        ])

    prompt = f"""You are Gaprio's orchestration planner. Create a concrete action plan using the available tools.

CONSENSUS:
{json.dumps(consensus, indent=2)}

{kg_context}

AVAILABLE TOOLS:
{tools_desc}

Create an action plan with specific tool calls. For each action:
- Use the EXACT tool name from the list above
- Provide ALL required parameters
- Include a rationale for why this action is needed

Respond with JSON:
{{
  "action_plan": [
    {{
      "tool": "exact_tool_name",
      "params": {{"param1": "value1"}},
      "rationale": "Why this action is needed",
      "priority": 1
    }}
  ],
  "plan_summary": "Human-readable summary of the plan"
}}

Respond ONLY with valid JSON."""

    response = await router.generate(TaskType.ORCHESTRATION, prompt, temperature=0.2)

    try:
        data = json.loads(response.strip().strip("```json").strip("```"))
    except json.JSONDecodeError:
        import re
        match = re.search(r'\{.*\}', response, re.DOTALL)
        data = json.loads(match.group()) if match else {"action_plan": [], "plan_summary": "Planning failed"}

    logger.info(f"Node 4: Created plan with {len(data.get('action_plan', []))} actions")
    return {
        "action_plan": data.get("action_plan", []),
        "plan_summary": data.get("plan_summary", ""),
        "memory_context": kg_context,
    }


# ═══════════════════════════════════════════════════
# NODE 5: Human Approval (interrupt — PAUSE)
# ═══════════════════════════════════════════════════
async def human_approval_node(state: GaprioState) -> dict:
    """
    PAUSE the pipeline and wait for human approval.
    Uses LangGraph's interrupt() to save state and halt execution.
    Three outcomes: approved → Node 6, edited → Node 4, rejected → END
    """
    plan = state.get("action_plan", [])
    summary = state.get("plan_summary", "")

    logger.info(f"Node 5: Waiting for human approval on {len(plan)} actions")

    # This call PAUSES the graph. State is checkpointed to MySQL.
    # The graph resumes when resume_pipeline() is called with a decision.
    decision = interrupt({
        "action_plan": plan,
        "plan_summary": summary,
        "message": "Please review the action plan and approve, edit, or reject.",
    })

    # After resume, 'decision' contains the user's response
    status = decision.get("status", "rejected")
    logger.info(f"Node 5: Received approval decision: {status}")

    return {
        "approval_status": status,
        "edit_instructions": decision.get("edit_instructions", ""),
        "approved_by": decision.get("user_id", ""),
    }


# ═══════════════════════════════════════════════════
# NODE 6: Action Execution (parallel via asyncio)
# ═══════════════════════════════════════════════════
async def action_execution_node(state: GaprioState) -> dict:
    """
    Execute all approved actions in PARALLEL.
    Handles partial success — if 2 of 3 tools succeed, continue.
    """
    import asyncio
    plan = state.get("action_plan", [])

    from gaprio.api import _agent
    if not _agent:
        return {"execution_results": [{"tool": "none", "success": False, "error": "Agent not initialized"}]}

    async def execute_single(action: dict) -> dict:
        tool_name = action.get("tool", "")
        params = action.get("params", {})

        if tool_name not in _agent.tools:
            return {"tool": tool_name, "success": False, "error": f"Unknown tool: {tool_name}"}

        try:
            tool = _agent.tools[tool_name]
            result = await tool.handler(**params)
            return {"tool": tool_name, "success": True, "result": str(result)[:500]}
        except Exception as e:
            logger.error(f"Tool {tool_name} failed: {e}")
            return {"tool": tool_name, "success": False, "error": str(e)}

    # Execute ALL actions in parallel
    results = await asyncio.gather(*[execute_single(a) for a in plan], return_exceptions=True)

    execution_results = []
    for r in results:
        if isinstance(r, Exception):
            execution_results.append({"tool": "unknown", "success": False, "error": str(r)})
        else:
            execution_results.append(r)

    successes = sum(1 for r in execution_results if r["success"])
    logger.info(f"Node 6: Executed {successes}/{len(plan)} actions successfully")
    return {"execution_results": execution_results}


# ═══════════════════════════════════════════════════
# NODE 7: Memory Update (LLM: Llama 3 8B — cheap)
# ═══════════════════════════════════════════════════
async def memory_update_node(state: GaprioState) -> dict:
    """
    Learn from this approved plan:
    - Extract patterns for team memory
    - Update knowledge graph with execution results
    - Log to long-term memory (MEMORY.md)
    """
    plan = state.get("action_plan", [])
    results = state.get("execution_results", [])
    workspace_id = state.get("workspace_id", "")

    # Extract pattern for team memory
    successful_tools = [r["tool"] for r in results if r.get("success")]
    if successful_tools:
        pattern_summary = f"Team executed: {', '.join(successful_tools)}"

        prompt = f"""Summarize what happened in this pipeline execution for long-term memory:

Action Plan: {json.dumps(plan, indent=2)}
Execution Results: {json.dumps(results, indent=2)}

Write a 2-3 sentence summary of what was accomplished, suitable for storing as a memory entry.
Respond with just the summary text, no JSON."""

        try:
            summary = await router.generate(TaskType.MEMORY, prompt, temperature=0.3)

            # Update knowledge graph with new execution edges
            from gaprio.core.knowledge_graph.extractor import KnowledgeGraphExtractor
            extractor = KnowledgeGraphExtractor()

            for result in results:
                if result.get("success") and result["tool"].startswith("asana_"):
                    # If we created an Asana task, extract it into the graph
                    await extractor.extract_from_asana_task(
                        {"name": result.get("result", ""), "gid": ""},
                        workspace_id
                    )

            logger.info(f"Node 7: Memory updated with {len(successful_tools)} successful actions")
            return {"memory_updates": [{"type": "pipeline_execution", "summary": summary.strip()}]}

        except Exception as e:
            logger.error(f"Memory update failed: {e}")

    return {"memory_updates": []}