# graph/supervisor.py (Final fix for _GeneratorContextManager error)

import os
import sqlite3
from pathlib import Path
from langgraph.graph import StateGraph, END, START
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

# --- Conditional Edge Logic (Router Functions remain the same) ---
def route_safety_check(state: ProjectState) -> str:
    # ... (content remains the same)
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
    
    # 2. Max Iteration Check: If acceptable AND max iterations reached, FORCE HIL.
    if state.get("iteration_count", 0) >= 3:
        print("Router (Critic): Max iterations reached. Forcing HIL_Node.")
        return "HIL_Node" 
        
    # 3. Default Path: If it passed metrics but is NOT at the iteration limit.
    # NOTE: In your original design, this line should never be hit if iteration_count >= 3.
    print("Router (Critic): Both metrics approved. Moving to Finalize.")
    return "Finalize"

def route_human_decision(state: ProjectState) -> str:
    # ... (content remains the same)
    human_action = state.get("human_decision", "Reject")

    if human_action == "Approve":
        print("Router (HIL): Human Approved. Moving to Finalize.")
        return "Finalize"

    print("Router (HIL): Human Rejected/Revise. Looping back to Drafting.")
    return "Drafting"

# --- Checkpointer Setup (Enforced SQLite Persistence) ---

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
    (THE FIX). Throws a RuntimeError on failure.
    """
    print(f"Attempting to initialize SQLite Checkpointer at: {SQLITE_DB_PATH}")
    try:
        # 1. Ensure file existence and permissions are correct
        _ensure_sqlite_file(SQLITE_DB_PATH)

        # 2. CREATE DIRECT CONNECTION OBJECT (THE FIX)
        # check_same_thread=False is essential for FastAPI/uvicorn (multithreading)
        conn = sqlite3.connect(SQLITE_DB_PATH, check_same_thread=False)

        # 3. INSTANTIATE SQLITESAVER DIRECTLY WITH THE CONNECTION
        checkpointer = SqliteSaver(conn) # <-- Fixed: No more .from_conn_string()

        print("SQLite Checkpointer initialized successfully. Persistence is ON.")
        return checkpointer

    except Exception as e:
        # CRITICAL: Throw an error as requested.
        error_msg = (
            f"CRITICAL ERROR: Failed to initialize SQLite Checkpointer. Persistence is OFF. Details: {e}"
        )
        print(error_msg)
        raise RuntimeError(error_msg) from e


# --- Graph Compilation ---
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
    workflow.set_entry_point("Drafting")

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

    # Compile with Persistence
    app = workflow.compile(checkpointer=get_checkpointer())
    return app


# Initialize the production graph for direct import
cbt_review_graph = compile_supervisor_graph()