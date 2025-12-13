import os
import sqlite3
from pathlib import Path
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from graph.state import ProjectState
from graph.agents import (
    drafting_agent_node,
    safety_agent_node,
    critic_agent_node,
    hil_node,
    finalize_node,
)
from dotenv import load_dotenv

load_dotenv()


def route_initial_entry(state: ProjectState) -> str:
    """
    Determines the first node to run when a thread is first invoked.
    If the state somehow indicates completion (e.g., if you are passing a mock approved state),
    it routes to Finalize, otherwise, it begins the Drafting process.
    """
    human_decision = state.get("human_decision")
    
    # In a resume scenario, human_decision is set to 'Approve' or 'Reject.
    # If the thread is new, human_decision is 'REVIEW_REQUIRED'.
    
    if human_decision == "Approve":
        print("Conditional Entry: State indicates pre-approval. Routing to Finalize.")
        return "Finalize"
    
    # If the decision is 'Reject' or 'REVIEW_REQUIRED', start the process.
    print("Conditional Entry: Starting new or revised process. Routing to Drafting.")
    return "Drafting"

# Conditional Edge Logic
def route_safety_check(state: ProjectState) -> str:
    SAFETY_THRESHOLD = 0.85
    if state.get("safety_metric", 0.0) < SAFETY_THRESHOLD:
        print(
            f"Router (Safety): Below threshold ({state.get('safety_metric', 0.0):.2f}). Looping back to Drafting."
        )
        return "Drafting"
    return "Critic"

def route_critic_check(state: ProjectState) -> str:
    """Routes based on Empathy/Critic Metric and Iteration Count."""
    EMPATHY_THRESHOLD = 0.70
    
    # 1. Empathy Check: If too low, always loop back.
    if state.get("empathy_metric", 0.0) < EMPATHY_THRESHOLD:
        print(f"Router (Critic): Below threshold ({state.get('empathy_metric', 0.0):.2f}). Looping back to Drafting.")
        return "Drafting" 
    
    current_iteration = state.get("iteration_count", 0)
    
    # 2. Halt Condition 1: First Draft Review
    # If the first iteration (count=1) passed metrics, send it to the Human.
    if current_iteration == 1:
        print("Router (Critic): First draft passed metrics. Moving to HIL_Node for initial review.")
        return "HIL_Node"
        
    # 3. Halt Condition 2: Max Iteration Check (for subsequent revisions)
    # If acceptable AND max iterations reached, FORCE Critic -> Finalize (this bypasses Human in Loop).
    # Max iterations is set to 20 (this includes all iterations for a thread, considering Human in loop reviews)
    if current_iteration >= 20:
        print("Router (Critic): Max iterations reached. Moving to Finalize.")
        return "Finalize" 
        
    # default to HIL node for review.
    print("Router (Critic): Metrics acceptable, but not yet finalized. Moving to HIL_Node for human review.")
    return "HIL_Node"


def route_human_decision(state: ProjectState) -> str:
    """
    Routes the graph based on the human's decision ('Approve' or 'Reject').
    This function is executed immediately after the human input is received.
    """
    human_decision = state.get("human_decision")
    print('------------------------------------------------------ ')
    print(human_decision)

    if human_decision == "Approve":
        print("Router (HIL): Decision is 'Approve'. Moving to Finalize.")
        # Finalize node leads directly to END
        return "Finalize"
    
    elif human_decision == "Reject":
        print("Router (HIL): Decision is 'Reject'. Moving to Drafting for revision.")
        # Drafting node starts the revision loop
        return "Drafting"
        
    else:
        # default to HIL node for review.
        print(f"Router (HIL): Unexpected decision or flag reset ({human_decision}). Halting.")
        return "HIL_Node"

# Checkpointer Setup (Enforced SQLite Persistence)
_BASE_DIR = Path(__file__).resolve().parent.parent
SQLITE_DB_PATH = str(_BASE_DIR / "cbt_review_board.sqlite")


def _ensure_sqlite_file(path: str) -> None:
    """
    Ensures the sqlite file exists and is writable. (This function is good and stays.)
    """
    db_path = Path(path)
    parent = db_path.parent
    try:
        parent.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        raise RuntimeError(f"Failed to create parent directory '{parent}': {e}") from e

    if db_path.exists():
        try:
            con = sqlite3.connect(str(db_path))
            con.execute("PRAGMA journal_mode = WAL;")
            con.close()
        except Exception as e:
            raise RuntimeError(
                f"Existing DB file '{db_path}' is not accessible: {e}"
            ) from e
    else:
        try:
            con = sqlite3.connect(str(db_path))
            con.execute(
                """CREATE TABLE IF NOT EXISTS _langgraph_db_init_check (id INTEGER PRIMARY KEY, created_at TEXT DEFAULT CURRENT_TIMESTAMP)"""
            )
            con.commit()
            con.close()
        except Exception as e:
            raise RuntimeError(f"Unable to create DB file '{db_path}': {e}") from e


def get_checkpointer() -> SqliteSaver:
    """
    Initializes and returns the SQLite Checkpointer by using a direct sqlite3 connection
    Throws a RuntimeError on failure.
    """
    print(f"Attempting to initialize SQLite Checkpointer at: {SQLITE_DB_PATH}")
    try:
        _ensure_sqlite_file(SQLITE_DB_PATH)

        # check_same_thread=False is essential for FastAPI/uvicorn (multithreading)
        conn = sqlite3.connect(SQLITE_DB_PATH, check_same_thread=False)

        # INSTANTIATE SQLITESAVER DIRECTLY WITH THE CONNECTION
        checkpointer = SqliteSaver(conn)

        print("SQLite Checkpointer initialized successfully. Persistence is ON.")
        return checkpointer

    except Exception as e:
        error_msg = (
            f"CRITICAL ERROR: Failed to initialize SQLite Checkpointer. Persistence is OFF. Details: {e}"
        )
        print(error_msg)
        raise RuntimeError(error_msg) from e


# Graph Compilation
def compile_supervisor_graph():
    """Compiles the entire CBT Review Board LangGraph workflow."""
    
    workflow = StateGraph(ProjectState)

    # Add Nodes (Agents)
    workflow.add_node("Drafting", drafting_agent_node)
    workflow.add_node("Safety", safety_agent_node)
    workflow.add_node("Critic", critic_agent_node)
    workflow.add_node("HIL_Node", hil_node)
    workflow.add_node("Finalize", finalize_node)

    # Set Entry Point and Edges
    workflow.set_conditional_entry_point(
        route_initial_entry,
        {"Drafting": "Drafting", "Finalize": "Finalize"}
    )

    workflow.add_edge("Drafting", "Safety")

    workflow.add_conditional_edges(
        "Safety",
        route_safety_check,
        {"Drafting": "Drafting", "Critic": "Critic"},
    )

    workflow.add_conditional_edges(
        "Critic",
        route_critic_check,
        {"Drafting": "Drafting", "HIL_Node": "HIL_Node", "Finalize": "Finalize"},
    )

    workflow.add_conditional_edges(
        "HIL_Node",
        route_human_decision,
        {"Drafting": "Drafting", "Finalize": "Finalize"},
    )

    workflow.add_edge("Finalize", END)

    # Compile with checkpointer based persistence
    app = workflow.compile(checkpointer=get_checkpointer())
    return app


# Initialize the production graph for direct import
cbt_review_graph = compile_supervisor_graph()