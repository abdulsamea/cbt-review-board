# api.py

from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
import uvicorn
import os
import time
from typing import Optional, Dict, Any

from graph.supervisor import compile_supervisor_graph, get_checkpointer # get_checkpointer is primarily for debug/info
from graph.state import ProjectState
from dotenv import load_dotenv

load_dotenv()

# --- FastAPI Setup ---
app = FastAPI(title="CBT Review Board API", version="1.0")

# --- Schemas for API Payload ---
class UserRequest(BaseModel):
    user_intent: str
    model_choice: str = 'openai'
    thread_id: Optional[str] = None

# --- Graph Initialization ---
# The checkpointer is initialized inside compile_supervisor_graph(),
# which will enforce the SQLite connection or raise an error on startup.
@app.on_event("startup")
async def startup_event():
    print("Initializing LangGraph application...")
    try:
        # Compiling the graph triggers the checkpointer initialization
        app.state.graph_app = compile_supervisor_graph() 
        print("LangGraph application and SQLite checkpointer initialized successfully.")
    except Exception as e:
        print(f"\nFATAL ERROR: Could not initialize persistence. Shutting down. Details: {e}")
        # In a production environment, you might log this error severity and force exit.
        # Here we let the app start but fail on the first request if persistence is critical.
        # (The compile() call will typically stop the process, fulfilling the requirement).
        pass 


# --- API Endpoint to Start a New Review ---
@app.post("/start_review")
async def start_review(request: UserRequest):
    """Starts a new LangGraph review thread and returns the result up to pause/end."""
    
    # Ensure the graph app was successfully initialized
    if not hasattr(app.state, 'graph_app'):
         raise HTTPException(status_code=503, detail="Service not initialized due to persistence failure.")
         
    # 1. Use existing thread_id or generate a new one
    thread_id = request.thread_id or f"session_{int(time.time())}"
    
    # 2. Define Initial State
    initial_state: ProjectState = {
        "user_intent": request.user_intent,
        "model_choice": request.model_choice,
        "current_draft": "",
        "draft_history": [],
        "iteration_count": 0,
        "safety_metric": 0.0,
        "empathy_metric": 0.0,
        "safety_report": {},
        "critic_notes": {},
        "final_output": None
    }

    # 3. Compile Configuration
    # NOTE: The checkpointer is already compiled into the app, so we only need the thread_id
    config = {"configurable": {"thread_id": thread_id}}
    
    # 4. Invoke the Graph to Run until it pauses or ends
    try:
        # Use stream() to run the graph and collect all updates until it pauses or ends.
        for _ in app.state.graph_app.stream(
            input=initial_state,
            config=config,
            stream_mode="updates"
        ):
            # We just consume the stream; the checkpointer saves the state
            pass 
            
    except Exception as e:
        print(f"Graph execution error for thread {thread_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Graph execution failed: {e}")

    # 5. Get final persistent state from checkpointer
    final_state = app.state.graph_app.get_state(config)
    
    # 6. Check for HIL pause or END state to set status
    status = "running"
    if final_state and final_state.next == ('__end__',):
        status = "approved"
    elif final_state and final_state.next and final_state.next[0] == 'hil_node':
        status = "paused_for_review"

    return {
        "thread_id": thread_id,
        "status": status,
        "current_draft": final_state.values.get("current_draft"),
        "metrics": {
            "safety_metric": final_state.values.get("safety_metric"),
            "empathy_metric": final_state.values.get("empathy_metric")
        },
        "next_nodes": final_state.next 
    }


# --- Endpoint to Resume a Paused Thread (e.g., after Human Approval) ---
@app.post("/resume_review/{thread_id}")
async def resume_review(thread_id: str, user_action: str = Body(..., embed=True)):
    """Resumes a thread paused by the Human-in-the-Loop node based on user action ('approve' or 'reject')."""
    
    if not hasattr(app.state, 'graph_app'):
         raise HTTPException(status_code=503, detail="Service not initialized due to persistence failure.")
         
    config = {"configurable": {"thread_id": thread_id}}
    app_instance = app.state.graph_app
    
    state = app_instance.get_state(config)
    
    if not state:
        raise HTTPException(status_code=404, detail="Thread not found.")
        
    if state.next != ('hil_node',):
        raise HTTPException(status_code=400, detail=f"Thread is not paused for human review. Current state: {state.next}")

    # 1. Update the state with the human's decision 
    if user_action.lower() == "approve":
        state.values['human_decision'] = "Approve" # Used by the router
        start_node = "HIL_Node" # Resume from the node that pauses
    elif user_action.lower() == "reject":
        state.values['human_decision'] = "Reject" # Used by the router
        start_node = "HIL_Node"
    else:
        raise HTTPException(status_code=400, detail="Invalid action. Must be 'approve' or 'reject'.")

    # 2. Continue the graph execution
    final_output = app_instance.invoke(state.values, config=config, recursion_limit=100)
    
    final_status = final_output.get("final_output", {}).get("final_status", "REVISED")
    
    return {
        "thread_id": thread_id,
        "status": "resumed",
        "final_draft": final_output.get("current_draft"),
        "final_status": final_status
    }


# --- Main Runner for Development ---
if __name__ == "__main__":
    print("Starting FastAPI server on http://127.0.0.1:8000")
    # Add reload=True for development environment
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)