# Market Research Deep Agent

A multi-agent market research assistant built with **deepagents**, **LangGraph**, **Gemini (`gemini-3.5-flash-lite`)**, and **Streamlit**, featuring Human-in-the-Loop (HITL) approval before writing the final report.

## Architecture

- **Lead Analyst (`deep_research_agent.py`)**: Orchestrates the research workflow using `create_deep_agent` with built-in planning (`write_todos`) and filesystem tools (`write_file`, `read_file`, `ls`) rooted in `outputs/`.
- **Specialized Subagents**:
  - `researcher`: Gathers pros, cons, Italian municipal regulations, and urban mobility statistics, saving structured findings to `outputs/notes.md`.
  - `critic`: Reviews `outputs/notes.md` and drafts for missing data, bias, or weak arguments, returning actionable revision feedback.
- **Human-in-the-Loop (`interrupt_on`)**: Uses LangGraph's `MemorySaver` checkpointer with `thread_id` and `interrupt_on={"write_file": True}` to pause before writing `final_report.md`. You can **Approve** or **Reject with feedback** (which loops back to revise the draft).
- **Interfaces**:
  - **CLI**: `python deep_research_agent.py`
  - **Streamlit Light UI**: `streamlit run app.py` (with `.streamlit/config.toml`)

## Setup

1. **Activate the virtual environment** (from `market-research-agent/`):
   ```bash
   source .venv/bin/activate
   ```
2. **Install dependencies** (already installed in `.venv`):
   ```bash
   pip install -r requirements.txt
   ```
3. **Configure `.env`**:
   Ensure `GEMINI_API_KEY` (and optional `LANGSMITH_TRACING`, `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT`) are set in `.env`.

## Running

### CLI Runner
```bash
python deep_research_agent.py
```

### Streamlit UI
```bash
streamlit run app.py
```
