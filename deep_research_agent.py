"""Core Deep Research Agent built with deepagents, LangGraph, and Gemini."""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend

# Load environment variables (GEMINI_API_KEY, LANGSMITH_*, etc.) at startup
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "gemini-3.5-flash-lite"

RESEARCHER_SUBAGENT = {
    "name": "researcher",
    "description": (
        "Gathers pros, cons, Italian municipal regulations, and urban mobility "
        "statistics on the research topic and writes structured findings to /notes.md."
    ),
    "system_prompt": (
        "You are an expert Market & Urban Mobility Researcher. "
        "When given a topic, gather comprehensive, balanced findings covering:\n"
        "1. Pros & opportunities (economic, environmental, commuter convenience)\n"
        "2. Cons & risks (safety, clutter, unit economics, vandalism)\n"
        "3. Italian municipal regulations (e.g., Codice della Strada, helmet/plate/insurance "
        "mandates, municipal tender caps, speed limits in ZTL/pedestrian zones)\n"
        "4. Urban mobility statistics (fleet sizes, modal shift, trip lengths, adoption trends)\n\n"
        "Write your structured, well-cited findings to `/notes.md` using `write_file`. "
        "Keep `/notes.md` clear, factual, and organized with markdown headings."
    ),
}

CRITIC_SUBAGENT = {
    "name": "critic",
    "description": (
        "Reviews /notes.md and report drafts, identifies missing data, bias, or "
        "weak arguments, and provides actionable revision feedback."
    ),
    "system_prompt": (
        "You are a rigorous Research Critic & Editorial Reviewer. "
        "Read `/notes.md` (and any draft if provided) using `read_file`. "
        "Identify:\n"
        "- Missing data, unverified claims, or gaps in Italian regulatory context\n"
        "- Potential bias or one-sided framing\n"
        "- Weak arguments or unclear structure\n\n"
        "Provide concise, actionable revision feedback so the lead analyst can "
        "produce a balanced, executive-ready final report."
    ),
}

MAIN_SYSTEM_PROMPT = (
    "You are a Lead Market Research Analyst. Your job is to produce a thorough, "
    "executive-ready market research report saved to `/final_report.md`.\n\n"
    "Follow this workflow:\n"
    "1. Use `write_todos` to create a clear, step-by-step research plan.\n"
    "2. Delegate initial research to the `researcher` subagent so it writes `/notes.md`.\n"
    "3. Delegate a quality review to the `critic` subagent to inspect `/notes.md` "
    "and identify gaps, bias, or weak arguments.\n"
    "4. Read `/notes.md` with `read_file`, incorporate the critic's feedback, "
    "and synthesize a polished, well-structured report into `/final_report.md` "
    "using `write_file`.\n"
    "5. Update your todos as you progress. If a human reviewer rejects `/final_report.md` "
    "with feedback, revise the report and call `write_file` again for approval."
)


def create_llm() -> ChatGoogleGenerativeAI:
    """Create the configured Gemini chat model."""
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    return ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=api_key,
        temperature=0.2,
    )


def build_agent(checkpointer: MemorySaver | None = None):
    """Build and compile the deep research agent with HITL and filesystem backend."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    saver = checkpointer if checkpointer is not None else MemorySaver()
    backend = FilesystemBackend(root_dir=OUTPUTS_DIR, virtual_mode=True)

    return create_deep_agent(
        model=create_llm(),
        system_prompt=MAIN_SYSTEM_PROMPT,
        subagents=[RESEARCHER_SUBAGENT, CRITIC_SUBAGENT],
        backend=backend,
        interrupt_on={"write_file": True},
        checkpointer=saver,
    )


def extract_interrupt_details(state_snapshot: Any) -> list[dict[str, Any]]:
    """Extract pending HITL action requests from a LangGraph state snapshot."""
    actions: list[dict[str, Any]] = []
    if not state_snapshot or not getattr(state_snapshot, "tasks", None):
        return actions

    for task in state_snapshot.tasks:
        for intr in getattr(task, "interrupts", ()) or ():
            val = getattr(intr, "value", intr)
            if isinstance(val, dict) and "action_requests" in val:
                actions.extend(val["action_requests"])
            elif isinstance(val, dict):
                actions.append(val)
    return actions


def run_cli() -> None:
    """Interactive CLI runner with Human-in-the-Loop approval."""
    print("=" * 60)
    print("Market Research Deep Agent (CLI)")
    print("=" * 60)
    topic = input("Enter research topic [Electric scooter sharing in Milan]: ").strip()
    if not topic:
        topic = "Electric scooter sharing in Milan"

    agent = build_agent()
    thread_id = f"cli-{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": thread_id}}

    print(f"\nStarting research on: {topic}\nThread ID: {thread_id}\n")
    current_input: Any = {
        "messages": [{"role": "user", "content": f"Conduct market research on: {topic}"}]
    }

    while True:
        for event in agent.stream(current_input, config=config, stream_mode="updates"):
            for node_name, update in event.items():
                if node_name == "__interrupt__":
                    continue
                if isinstance(update, dict) and "todos" in update:
                    todos = update["todos"]
                    done = sum(1 for t in todos if t.get("status") == "completed")
                    print(f"[Plan] {done}/{len(todos)} todos completed")

        state = agent.get_state(config)
        if not state.next:
            break

        actions = extract_interrupt_details(state)
        print("\n" + "-" * 60)
        print("HUMAN-IN-THE-LOOP APPROVAL REQUIRED")
        print("-" * 60)
        for idx, act in enumerate(actions, start=1):
            name = act.get("name", "unknown_tool")
            args = act.get("args", {})
            file_path = args.get("file_path", "")
            content = args.get("content", "")
            print(f"Action #{idx}: {name} -> {file_path}")
            if content:
                preview = content[:600] + ("..." if len(content) > 600 else "")
                print(f"\nPreview:\n{preview}\n")

        decision = input("Approve this action? [a]pprove / [r]eject: ").strip().lower()
        if decision.startswith("a"):
            decisions = [{"type": "approve"} for _ in (actions or [1])]
            current_input = Command(resume={"decisions": decisions})
        else:
            feedback = input("Enter revision feedback for the agent: ").strip()
            if not feedback:
                feedback = "Please revise and improve the report."
            decisions = [
                {"type": "reject", "message": feedback} for _ in (actions or [1])
            ]
            current_input = Command(resume={"decisions": decisions})

    print("\n" + "=" * 60)
    print("Research complete! Generated files in:", OUTPUTS_DIR)
    for fname in ("notes.md", "final_report.md"):
        fpath = OUTPUTS_DIR / fname
        if fpath.exists():
            print(f" - {fpath}")
    print("=" * 60)


if __name__ == "__main__":
    run_cli()
