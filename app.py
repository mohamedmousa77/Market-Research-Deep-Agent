"""Clean Light Streamlit UI for the Market Research Deep Agent."""

from __future__ import annotations

import uuid
from typing import Any

import streamlit as st
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from deep_research_agent import (
    OUTPUTS_DIR,
    build_agent,
    extract_interrupt_details,
)

st.set_page_config(
    page_title="Market Research Deep Agent",
    page_icon="📊",
    layout="wide",
)


def init_session_state() -> None:
    if "checkpointer" not in st.session_state:
        st.session_state.checkpointer = MemorySaver()
    if "agent" not in st.session_state:
        st.session_state.agent = build_agent(checkpointer=st.session_state.checkpointer)
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = f"ui-{uuid.uuid4().hex[:8]}"
    if "todos" not in st.session_state:
        st.session_state.todos = []
    if "activity" not in st.session_state:
        st.session_state.activity = []
    if "running" not in st.session_state:
        st.session_state.running = False
    if "completed" not in st.session_state:
        st.session_state.completed = False


def reset_session() -> None:
    st.session_state.checkpointer = MemorySaver()
    st.session_state.agent = build_agent(checkpointer=st.session_state.checkpointer)
    st.session_state.thread_id = f"ui-{uuid.uuid4().hex[:8]}"
    st.session_state.todos = []
    st.session_state.activity = []
    st.session_state.running = False
    st.session_state.completed = False


def run_agent_step(input_payload: Any) -> None:
    """Stream agent updates until completion or HITL interrupt."""
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    agent = st.session_state.agent

    for event in agent.stream(input_payload, config=config, stream_mode="updates"):
        for node_name, update in event.items():
            if node_name == "__interrupt__":
                st.session_state.activity.append("⏸️ Paused for human approval")
                continue

            if isinstance(update, dict):
                if "todos" in update:
                    st.session_state.todos = update["todos"]

                for msg in update.get("messages", []) or []:
                    tool_calls = getattr(msg, "tool_calls", None) or []
                    for tc in tool_calls:
                        tc_name = tc.get("name", "")
                        if tc_name == "task":
                            subagent = tc.get("args", {}).get("subagent_type", "subagent")
                            st.session_state.activity.append(
                                f"🤖 Delegated task to **{subagent}**"
                            )
                        elif tc_name:
                            st.session_state.activity.append(f"🛠️ Called `{tc_name}`")

    state = agent.get_state(config)
    st.session_state.completed = not bool(state.next)


init_session_state()

st.title("📊 Market Research Deep Agent")
st.caption(
    "Powered by `deepagents`, `LangGraph`, and `Gemini 3.5 Flash-Lite` "
    "with Human-in-the-Loop approval."
)

with st.sidebar:
    st.header("Research Controls")
    topic = st.text_input(
        "Research Topic",
        value="Electric scooter sharing in Milan",
        placeholder="Enter a market research topic...",
    )
    col1, col2 = st.columns(2)
    with col1:
        start_clicked = st.button("🚀 Start", type="primary", use_container_width=True)
    with col2:
        if st.button("🔄 Reset", use_container_width=True):
            reset_session()
            st.rerun()

    st.divider()
    st.subheader("📋 Live Todo Plan")
    if st.session_state.todos:
        for todo in st.session_state.todos:
            status = todo.get("status", "pending")
            icon = {"completed": "✅", "in_progress": "🔄", "pending": "⏳"}.get(
                status, "⏳"
            )
            st.markdown(f"{icon} {todo.get('content', '')}")
    else:
        st.info("Todo plan will appear here once research starts.")

    st.divider()
    st.subheader("🤖 Subagent Activity")
    if st.session_state.activity:
        for item in st.session_state.activity[-12:]:
            st.markdown(f"- {item}")
    else:
        st.caption("No activity yet.")

if start_clicked and topic.strip():
    st.session_state.activity.append(f"🔎 Started research: **{topic.strip()}**")
    with st.spinner("Running research and critic subagents..."):
        run_agent_step(
            {"messages": [{"role": "user", "content": f"Conduct market research on: {topic.strip()}"}]}
        )
    st.rerun()

# Check if paused for Human-in-the-Loop approval
config = {"configurable": {"thread_id": st.session_state.thread_id}}
state = st.session_state.agent.get_state(config)

if state.next:
    actions = extract_interrupt_details(state)
    st.warning("⏸️ **Human-in-the-Loop Approval Required** before writing to disk.")
    for idx, act in enumerate(actions, start=1):
        args = act.get("args", {})
        file_path = args.get("file_path", "unknown")
        content = args.get("content", "")
        with st.expander(f"Action #{idx}: `{act.get('name', 'write_file')}` → `{file_path}`", expanded=True):
            if content:
                st.markdown(content)

    feedback = st.text_area(
        "Revision Feedback (required if rejecting)",
        placeholder="e.g., Add more data on Milan's 2024 municipal fleet cap and helmet rules...",
    )
    c1, c2 = st.columns(2)
    with c1:
        if st.button("✅ Approve & Write File", type="primary", use_container_width=True):
            decisions = [{"type": "approve"} for _ in (actions or [1])]
            with st.spinner("Resuming agent..."):
                run_agent_step(Command(resume={"decisions": decisions}))
            st.rerun()
    with c2:
        if st.button("❌ Reject with Feedback", use_container_width=True):
            msg = feedback.strip() or "Please revise and improve the report."
            decisions = [{"type": "reject", "message": msg} for _ in (actions or [1])]
            st.session_state.activity.append(f"✍️ Rejected with feedback: {msg}")
            with st.spinner("Revising draft based on your feedback..."):
                run_agent_step(Command(resume={"decisions": decisions}))
            st.rerun()

# Output Previews
tab_report, tab_notes = st.tabs(["📄 Final Report (final_report.md)", "📝 Research Notes (notes.md)"])

with tab_report:
    report_path = OUTPUTS_DIR / "final_report.md"
    if report_path.exists():
        report_text = report_path.read_text(encoding="utf-8")
        st.download_button(
            "⬇️ Download final_report.md",
            data=report_text,
            file_name="final_report.md",
            mime="text/markdown",
        )
        st.markdown(report_text)
    else:
        st.info("`outputs/final_report.md` has not been generated yet.")

with tab_notes:
    notes_path = OUTPUTS_DIR / "notes.md"
    if notes_path.exists():
        notes_text = notes_path.read_text(encoding="utf-8")
        st.download_button(
            "⬇️ Download notes.md",
            data=notes_text,
            file_name="notes.md",
            mime="text/markdown",
        )
        st.markdown(notes_text)
    else:
        st.info("`outputs/notes.md` has not been generated yet.")
