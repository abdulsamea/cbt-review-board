import asyncio
import json
from pathlib import Path
import sqlite3
import threading
import uvicorn
import uuid
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Any, Dict, Literal, Optional, Union
from graph.schemas import HumanDecision
from graph.state import CriticNotes, SafetyReport
from graph.supervisor import cbt_review_graph
import msgpack

from utils import make_json_safe

app_api = FastAPI()

_BASE_DIR = Path(__file__).resolve().parent
DB_PATH = _BASE_DIR / "cbt_review_board.sqlite"

app_api.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://localhost:\d+|http://127\.0\.0\.1:\d+",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory tracking for background graph runs
active_threads: Dict[str, threading.Thread] = {}
thread_errors: Dict[str, str] = {}

# Data Models APIs

class StartSessionRequest(BaseModel):
    user_prompt: str
    thread_id: Optional[str] = None 
    model_choice: str = "openai"

class ResumeSessionRequest(BaseModel):
    thread_id: str
    suggested_content: str = Field(description="Specific feedback or new instructions for the drafting agent.")
    human_decision: HumanDecision

class SessionStatus(BaseModel):
    thread_id: str
    is_complete: bool
    status: Literal["running", "halted", "complete", "revising"]
    current_draft: Optional[str] = Field(description="The current draft awaiting human review.")
    final_cbt_plan: Optional[str] = Field(description="The final approved output.")
    safety_metric: Optional[float]
    empathy_metric: Optional[float]
    model_choice: str
    active_node: Optional[str] = Field(default=None, description="Last known active node in the graph.")
    active_node_label: Optional[str] = Field(default=None, description="User-friendly label for the active node.")

def create_initial_state(user_prompt: str, thread_id: str, model_choice: str) -> Dict[str, Any]:
    """Initializes ProjectState with mandatory fields as a dict."""
    return {
        "user_intent": user_prompt,
        "thread_id": thread_id,
        "current_draft": "",
        "draft_history": [],
        "iteration_count": 0,
        "model_choice": model_choice,
        "active_node": "Drafting",
        "safety_metric": 0.0,
        "empathy_metric": 0.0,
        "critic_notes": CriticNotes(empathy_revision="", structure_revision=""),
        "safety_report": SafetyReport(flagged_lines=[], safety_score=0.0),
        "next_node": model_choice,  
        "human_decision": "REVIEW_REQUIRED" 
    }


def get_state_from_checkpoint(checkpoint: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Safely extracts the state data from a LangGraph checkpoint dictionary."""
    if not checkpoint:
        raise HTTPException(status_code=500, detail="CRITICAL: Checkpoint not found. Graph failed to save state.")
        
    state_data = checkpoint.get('channel_values')
    
    if state_data is None:
        state_data = checkpoint.get('values', checkpoint) 
        
    # NOTE: Since state_data is likely a channel value wrapper, we need to extract the actual state dictionary
    # Assuming ProjectState is a dict, we extract it from the LangGraph wrapper (usually '__root__')
    if isinstance(state_data, dict) and "__root__" in state_data:
        root_data = state_data["__root__"].get("value")
        if isinstance(root_data, dict):
             return root_data
        # Handle string serialization if state is serialized as a JSON string
        elif isinstance(root_data, str):
            try:
                return json.loads(root_data)
            except json.JSONDecodeError:
                pass
        
    # Final validation/error for unhandled structures
    if not isinstance(state_data, dict) or 'thread_id' not in state_data:
          raise HTTPException(
              status_code=500, 
              detail=f"CRITICAL: Checkpoint data is malformed. Retrieved dictionary lacks necessary state keys. Full checkpoint keys: {list(checkpoint.keys())}"
           )

    return state_data # Return the state_data if it's already the dict (older format)


def _derive_status_view(
    thread_id: str,
    state: Optional[Dict[str, Any]],
    thread_alive: bool,
    default_model_choice: Optional[str] = None,
) -> Dict[str, Any]:
    """Derives a user-friendly status dictionary from the raw state."""
    node_labels = {
        "Drafting": "Drafting Team",
        "Safety": "Safety Team",
        "Critic": "Clinical Critic Team",
        "HIL_Node": "Human Review",
        "Finalize": "Finalize",
        "END": "Finalized"
    }

    human_action = state.get("human_decision") if state else None
    current_draft = state.get("current_draft") if state else None
    final_cbt_plan = None
    is_complete = False
    
    status: Literal["running", "halted", "complete", "revising"] = "running"
    if thread_alive:
         status = "running"
         if human_action == "Reject":
             status = "revising"
    else:
        status = "halted"

    if state.get("active_node") == "END" or human_action == "Approve":
        status = "complete"
        is_complete = True
        final_cbt_plan = state.get("current_draft") # Assuming current_draft holds final output upon completion

    active_node = state.get("active_node", "HIL_Node") if state else None
    
    return {
        "thread_id": thread_id,
        "is_complete": is_complete,
        "status": status,
        "current_draft": current_draft,
        "final_cbt_plan": final_cbt_plan,
        "safety_metric": state.get("safety_metric") if state else None,
        "empathy_metric": state.get("empathy_metric") if state else None,
        "model_choice": (state.get("model_choice") if state else default_model_choice) or "openai",
        "active_node": active_node,
        "active_node_label": node_labels.get(active_node) if active_node else None,
        "error": thread_errors.get(thread_id),
    }


def run_graph_in_background(initial_state: Dict[str, Any], config: Dict[str, Any], thread_id: str) -> None:
    """Run the LangGraph in a background thread and track errors. (As provided)"""
    thread_errors.pop(thread_id, None)
    try:
        # NOTE: This is synchronous invoke, suitable for background threading.
        cbt_review_graph.invoke(initial_state, config=config)
    except Exception as e:
        thread_errors[thread_id] = str(e)
        print(f"Background graph execution for {thread_id} stopped: {e}")
    finally:
        active_threads.pop(thread_id, None)

def execute_graph_in_background(thread_id: str, state_to_invoke: Dict[str, Any]):
    """
    Common function to check if thread is running and execute the graph in a new thread.
    """
    config = {"configurable": {"thread_id": thread_id}}

    existing_thread = active_threads.get(thread_id)
    if existing_thread and existing_thread.is_alive():
        raise HTTPException(status_code=400, detail=f"Session {thread_id} is already running.")

    background_thread = threading.Thread(
        target=run_graph_in_background,
        args=(state_to_invoke, config, thread_id),
        daemon=True,
    )
    active_threads[thread_id] = background_thread
    background_thread.start()

def _prepare_and_invoke_session(
    thread_id: str,
    initial_state_or_resume_req: Union[Dict[str, Any], ResumeSessionRequest],
    default_model_choice: Optional[str] = None
) -> SessionStatus:
    """
    Unified logic to prepare state (initial or resumed) and start background execution.
    """
    state_to_invoke: Dict[str, Any]
    
    if isinstance(initial_state_or_resume_req, dict):
        # Case 1: Start Session
        state_to_invoke = initial_state_or_resume_req
        
    elif isinstance(initial_state_or_resume_req, ResumeSessionRequest):
        # Case 2: Resume Session
        req = initial_state_or_resume_req
        config = {"configurable": {"thread_id": req.thread_id}}
        
        # Load latest state from checkpointer
        checkpoint = cbt_review_graph.checkpointer.get(config)
        if not checkpoint:
            raise HTTPException(status_code=404, detail=f"Session {req.thread_id} not found.")
            
        current_state_data = get_state_from_checkpoint(checkpoint)
        state_to_invoke = dict(current_state_data) # Create mutable copy
        
        # Apply human input logic
        if req.human_decision == 'Approve':
            state_to_invoke['human_decision'] = 'Approve' 
            state_to_invoke['user_intent'] = state_to_invoke.get('user_intent', '').split("REVISION INSTRUCTION")[0].strip()

        elif req.human_decision == 'Reject':
            state_to_invoke['user_intent'] = (
                f"REVISION INSTRUCTION (Based on Rejected Draft): {req.suggested_content}"
            )
            state_to_invoke['human_decision'] = 'Reject' 
            state_to_invoke['active_node'] = 'Drafting' # Explicitly set node for quick stream update
            state_to_invoke['status'] = 'revising'
            
    else:
        raise ValueError("Invalid input type for _prepare_and_invoke_session")

    # Start graph execution
    execute_graph_in_background(thread_id, state_to_invoke)

    # Return the latest state immediately
    status_view = _derive_status_view(thread_id, state=state_to_invoke, thread_alive=True, default_model_choice=default_model_choice)

    return SessionStatus(**status_view)

@app_api.post("/start_session", response_model=SessionStatus)
async def start_session(req: StartSessionRequest):
    """Starts a LangGraph CBT session and executes it up to the first 'review' point."""
    
    thread_id = req.thread_id or str(uuid.uuid4())
    initial_state = create_initial_state(req.user_prompt, thread_id, req.model_choice)
    
    return _prepare_and_invoke_session(
        thread_id=thread_id,
        initial_state_or_resume_req=initial_state,
        default_model_choice=req.model_choice
    )


@app_api.post("/resume_session", response_model=SessionStatus)
async def resume_session(req: ResumeSessionRequest):
    """Resumes a halted session with human input and runs the graph in background."""
    
    return _prepare_and_invoke_session(
        thread_id=req.thread_id,
        initial_state_or_resume_req=req
    )

@app_api.get("/stream_session_info")
async def stream_session_info(thread_id: str, poll_interval: float = 0.75):
    """
    Streams session info (phase, status, metrics) in real time for the frontend.
    Uses Server-Sent Events (text/event-stream) to push updates until the graph halts or completes.
    """
    config = {"configurable": {"thread_id": thread_id}}

    async def event_generator():
        last_payload: Optional[Dict[str, Any]] = None
        while True:
            checkpoint = cbt_review_graph.checkpointer.get(config)
            state: Optional[Dict[str, Any]] = None
            
            if checkpoint:
                try:
                    state = get_state_from_checkpoint(checkpoint)
                except HTTPException:
                    state = None
            
            thread_ref = active_threads.get(thread_id)
            thread_alive = bool(thread_ref and thread_ref.is_alive())
            payload = _derive_status_view(thread_id, state or {}, thread_alive)

            # Send update only if the payload content has changed
            if payload != last_payload:
                yield f"data: {json.dumps(payload)}\n\n"
                last_payload = payload

            # Exit condition: Status is terminal AND the background thread is confirmed dead
            if payload["status"] in ("complete", "halted") and not thread_alive:
                break

            await asyncio.sleep(poll_interval)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app_api.get("/checkpoints/{checkpoint_id}")
def get_checkpoint(checkpoint_id: str):
    if not DB_PATH.exists():
        raise HTTPException(status_code=500, detail="SQLite database not found")

    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT checkpoint FROM checkpoints WHERE checkpoint_id = ?",
            (checkpoint_id,),
        )

        row = cursor.fetchone()
        conn.close()

        if row is None:
            raise HTTPException(status_code=404, detail="Checkpoint not found")

        blob = row[0]

        decoded = msgpack.unpackb(blob, raw=False)

        safe_payload = make_json_safe(decoded)

        return {
            "checkpoint_id": checkpoint_id,
            "checkpoint": safe_payload,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app_api.get("/threads/{thread_id}/checkpoints")
def get_all_checkpoints_for_thread(thread_id: str):
    """
    Returns all checkpoints for a given thread_id in chronological order.
    """
    if not DB_PATH.exists():
        raise HTTPException(status_code=500, detail="SQLite database not found")

    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT checkpoint_id, checkpoint
            FROM checkpoints
            WHERE thread_id = ?
            ORDER BY checkpoint_id ASC
            """,
            (thread_id,),
        )

        rows = cursor.fetchall()
        conn.close()

        if not rows:
            raise HTTPException(
                status_code=404,
                detail=f"No checkpoints found for thread_id: {thread_id}",
            )

        checkpoints = []
        for checkpoint_id, blob in rows:
            decoded = msgpack.unpackb(blob, raw=False)
            safe_payload = make_json_safe(decoded)

            checkpoints.append(
                {
                    "checkpoint_id": checkpoint_id,
                    "checkpoint": safe_payload,
                }
            )

        return {
            "thread_id": thread_id,
            "total_checkpoints": len(checkpoints),
            "checkpoints": checkpoints,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app_api, host="0.0.0.0", port=8000)