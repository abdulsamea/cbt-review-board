import os
import sqlite3
from pathlib import Path
from typing import List

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from dotenv import load_dotenv

from graph.state import (
    ProjectState,
    BlackboardNote,
    IntentSignal,
)
from graph.agents import (
    drafting_agent_node,
    safety_agent_node,
    critic_agent_node,
    hil_node,
    finalize_node,
)

load_dotenv()

# blacboard helper functions
def _has_unresolved_blockers(state: ProjectState) -> bool:
    """Check if any unresolved blocking notes exist."""
    return any(
        note["severity"] == "blocker" and not note["resolved"]
        for note in state.get("blackboard_notes", [])
    )


def _emit_intent(
    state: ProjectState,
    *,
    from_agent: str,
    to_agent: str,
    intent: str,
    reason: str,
) -> None:
    """Append an intent signal to the blackboard."""
    state.setdefault("intent_signals", []).append(
        {
            "from_agent": from_agent,
            "to_agent": to_agent,
            "intent": intent,
            "reason": reason,
        }
    )

def route_initial_entry(state: ProjectState) -> str:
    """
    Entry router for new or resumed threads.
    """
    state["active_node"] = "Drafting"

    if state.get("human_decision") == "Approve":
        print("Conditional Entry: State indicates pre-approval. Routing to Finalize.")
        return "Finalize"
    print("Conditional Entry: Starting new or revised process. Routing to Drafting.")
    return "Drafting"


def route_safety_check(state: ProjectState) -> str:
    """
    Routes after Safety agent evaluation.
    Emits explicit intent if revision is required.
    """
    SAFETY_THRESHOLD = 0.70

    if state["safety_metric"] < SAFETY_THRESHOLD:
        _emit_intent(
            state,
            from_agent="Safety",
            to_agent="Drafting",
            intent="revise_for_safety",
            reason="Safety score below threshold",
        )
        return "Drafting"

    return "Critic"


def route_critic_check(state: ProjectState) -> str:
    """
    Routes after Critic agent evaluation.
    Uses blackboard notes AND metrics.
    """
    EMPATHY_THRESHOLD = 0.60

    # Stop if unresolved blockers exist
    if _has_unresolved_blockers(state):
        _emit_intent(
            state,
            from_agent="Critic",
            to_agent="Drafting",
            intent="revise_structure",
            reason="Unresolved blocker notes present",
        )
        return "Drafting"

    # Empathy failure
    if state["empathy_metric"] < EMPATHY_THRESHOLD:
        _emit_intent(
            state,
            from_agent="Critic",
            to_agent="Drafting",
            intent="revise_for_empathy",
            reason="Empathy score below threshold",
        )
        return "Drafting"

    iteration = state.get("iteration_count", 0)

    # First acceptable draft → human review
    if iteration == 1:
        _emit_intent(
            state,
            from_agent="Critic",
            to_agent="Supervisor",
            intent="human_review_required",
            reason="First acceptable draft",
        )
        return "HIL_Node"

    # 3. Halt Condition
    # Max Iteration Check (for subsequent revisions)
    # If acceptable AND max iterations reached, FORCE Critic -> Finalize (this bypasses Human in Loop).
    # # Max iterations is set to 20 (this includes all iterations for a thread, considering Human in loop reviews)
    if iteration >= 20:
        return "Finalize"

    return "HIL_Node"


def route_human_decision(state: ProjectState) -> str:
    """
    Routes based on explicit human decision.
    """
    decision = state.get("human_decision")

    if decision == "Approve":
        print("Router (HIL): Decision is 'Approve'. Moving to Finalize.")
        _emit_intent(
            state,
            from_agent="Supervisor",
            to_agent="Finalize",
            intent="ready_to_finalize",
            reason="Human approved output",
        )
        return "Finalize"

    if decision == "Reject":
        print("Router (HIL): Decision is 'Reject'. Moving to Drafting for revision.")
        _emit_intent(
            state,
            from_agent="Supervisor",
            to_agent="Drafting",
            intent="revise_structure",
            reason="Human requested revision",
        )
        return "Drafting"
    else:
        # default to HIL node for review.
        print(f"Router (HIL): Unexpected decision or flag reset ({decision}). Halting.")
        return "HIL_Node"


_BASE_DIR = Path(__file__).resolve().parent.parent
SQLITE_DB_PATH = str(_BASE_DIR / "cbt_review_board.sqlite")


def _ensure_sqlite_file(path: str) -> None:
    db_path = Path(path)
    parent = db_path.parent

    parent.mkdir(parents=True, exist_ok=True)

    if db_path.exists():
        con = sqlite3.connect(str(db_path))
        con.execute("PRAGMA journal_mode = WAL;")
        con.close()
    else:
        con = sqlite3.connect(str(db_path))
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS _langgraph_db_init_check (
                id INTEGER PRIMARY KEY,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        con.commit()
        con.close()


def get_checkpointer() -> SqliteSaver:
    _ensure_sqlite_file(SQLITE_DB_PATH)

    conn = sqlite3.connect(
        SQLITE_DB_PATH,
        check_same_thread=False,  # required for FastAPI threads
    )

    return SqliteSaver(conn)


def compile_supervisor_graph():
    """
    Compiles the CBT Review Board LangGraph workflow
    with a true blackboard state.
    """
    workflow = StateGraph(ProjectState)

    # Nodes
    workflow.add_node("Drafting", drafting_agent_node)
    workflow.add_node("Safety", safety_agent_node)
    workflow.add_node("Critic", critic_agent_node)
    workflow.add_node("HIL_Node", hil_node)
    workflow.add_node("Finalize", finalize_node)

    # Entry
    workflow.set_conditional_entry_point(
        route_initial_entry,
        {
            "Drafting": "Drafting",
            "Finalize": "Finalize",
        },
    )

    # Edges
    workflow.add_edge("Drafting", "Safety")

    workflow.add_conditional_edges(
        "Safety",
        route_safety_check,
        {
            "Drafting": "Drafting",
            "Critic": "Critic",
        },
    )

    workflow.add_conditional_edges(
        "Critic",
        route_critic_check,
        {
            "Drafting": "Drafting",
            "HIL_Node": "HIL_Node",
            "Finalize": "Finalize",
        },
    )

    workflow.add_conditional_edges(
        "HIL_Node",
        route_human_decision,
        {
            "Drafting": "Drafting",
            "Finalize": "Finalize",
        },
    )

    workflow.add_edge("Finalize", END)

    return workflow.compile(checkpointer=get_checkpointer())

cbt_review_graph = compile_supervisor_graph()
