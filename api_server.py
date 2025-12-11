import uvicorn
import uuid
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Literal, Optional, Union
from graph.state import ProjectState, CriticNotes, SafetyReport
from graph.supervisor import cbt_review_graph 

app_api = FastAPI()

app_api.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Data Models for Communication (UPDATED) ---

class StartSessionRequest(BaseModel):
    user_prompt: str
    thread_id: Optional[str] = None 
    model_choice: str = "openai" 

class ResumeSessionRequest(BaseModel):
    thread_id: str
    # CHANGED: Use suggested_content for specific revision instructions
    suggested_content: str = Field(description="Specific feedback or new instructions for the drafting agent.")
    human_decision: Literal['Approve', 'Reject'] # The decision flag

class SessionStatus(BaseModel):
    thread_id: str
    is_complete: bool
    status: Literal["running", "halted", "complete", "revising"]
    current_draft: Optional[str] = Field(description="The current draft awaiting human review.")
    final_cbt_plan: Optional[str] = Field(description="The final approved output.")
    safety_metric: Optional[float]
    empathy_metric: Optional[float]
    model_choice: str
    
# --- Helper Function to Create Initial State (omitted for brevity) ---

def create_initial_state(user_prompt: str, thread_id: str, model_choice: str) -> ProjectState:
    """Initializes ProjectState with mandatory fields."""
    return ProjectState(
        user_intent=user_prompt,
        thread_id=thread_id,
        current_draft="",
        draft_history=[],
        iteration_count=0,
        model_choice=model_choice,
        safety_metric=0.0,
        empathy_metric=0.0,
        critic_notes=CriticNotes(empathy_revision="", structure_revision=""),
        safety_report=SafetyReport(flagged_lines=[], safety_score=0.0),
        next_node=model_choice,  
        human_decision="REVIEW_REQUIRED" 
    )

# --- Core Checkpoint Retrieval Function (THE FINAL FIX) ---
def get_state_from_checkpoint(checkpoint: Optional[Dict[str, Any]]) -> ProjectState:
    """
    Safely extracts the state data from a LangGraph checkpoint dictionary.
    Prioritizes 'channel_values' based on the latest environment feedback.
    """
    if not checkpoint:
        raise HTTPException(status_code=500, detail="CRITICAL: Checkpoint not found. Graph failed to save state.")
        
    # 1. Try 'channel_values' key (Based on recent error traceback)
    state_data = checkpoint.get('channel_values')
    
    # 2. Fallback: If 'channel_values' is missing
    if state_data is None:
        state_data = checkpoint.get('values', checkpoint) 
        
    # 3. Final validation
    if not isinstance(state_data, dict) or 'thread_id' not in state_data:
         raise HTTPException(
            status_code=500, 
            detail=f"CRITICAL: Checkpoint data is malformed. Retrieved dictionary lacks necessary state keys. Full checkpoint keys: {list(checkpoint.keys())}"
         )

    return state_data


# --- API Endpoints ---

@app_api.post("/start_session", response_model=SessionStatus)
async def start_session(req: StartSessionRequest):
    """Starts a LangGraph session and executes it up to the first 'halt' point."""
    
    thread_id = req.thread_id or str(uuid.uuid4())
    initial_state = create_initial_state(req.user_prompt, thread_id, req.model_choice)
    config = {"configurable": {"thread_id": thread_id}}
    
    try:
        cbt_review_graph.invoke(initial_state, config=config)
    except Exception as e:
        print(f"Graph execution stopped (expected halt): {e}")

    # 1. Fetch the checkpoint
    current_checkpoint = cbt_review_graph.checkpointer.get(config) 
    
    # 2. Safely extract the state data
    current_state: ProjectState = get_state_from_checkpoint(current_checkpoint)

    # 3. Map ProjectState (TypedDict) to SessionStatus (Pydantic)
    return SessionStatus(
        thread_id=thread_id,
        is_complete=False,
        status="halted", 
        current_draft=current_state.get('current_draft'),
        final_cbt_plan=None,
        safety_metric=current_state.get('safety_metric'),
        empathy_metric=current_state.get('empathy_metric'),
        model_choice=current_state.get('model_choice')
    )


@app_api.post("/resume_session", response_model=SessionStatus)
async def resume_session(req: ResumeSessionRequest):
    """Resumes a halted session with human input."""
    
    config = {"configurable": {"thread_id": req.thread_id}}
    checkpoint = cbt_review_graph.checkpointer.get(config)
    
    # 1. Safely load and update the state
    current_state_data = get_state_from_checkpoint(checkpoint)
    
    # Create a mutable copy (dict) of the state for modification
    current_state: ProjectState = dict(current_state_data)
    
    # 2. Modify the state based on human input for resumption (CRUCIAL LOGIC CHANGE HERE)
    
    if req.human_decision == 'Approve':
        # --- APPROVAL PATH: Finalize the Draft ---
        
        # 1. Set the decision flag to trigger the 'Finalize' route
        current_state['human_decision'] = 'Approve' 
        
        # 2. MITIGATION: Clear the user_intent to ensure it doesn't contain a residual revision instruction
        current_state['user_intent'] = current_state.get('user_intent', '').split("REVISION INSTRUCTION")[0].strip()

    elif req.human_decision == 'Reject':
        # If rejected, the agent MUST take the suggestion as the new primary intent/revision context.
        # The Drafting Agent uses 'user_intent' for the core task.
        current_state['user_intent'] = (
            f"REVISION INSTRUCTION (Based on Rejected Draft): {req.suggested_content}"
        )
        # Note: The 'current_draft' field already holds the previous rejected draft, 
        # which the Drafting Agent uses for revision context in agents.py.

        current_state['human_decision'] = 'Reject' 

    # 3. Resume execution
    try:
        cbt_review_graph.invoke(current_state, config=config)
    except Exception as e:
        print(f"Resume execution stopped: {e}")

    # 4. Fetch the final state and return status
    final_checkpoint = cbt_review_graph.checkpointer.get(config)
    final_state: ProjectState = get_state_from_checkpoint(final_checkpoint)
    
    # 5. Determine status and final output based on the last decision/state
    human_action = final_state.get('human_decision')
    
    is_complete: bool
    status: Literal["running", "halted", "complete", "revising"]
    final_cbt_plan: Optional[str] = None
    current_draft = final_state.get('current_draft')
    
    if human_action == 'Approve':
        status = "complete"
        is_complete = True
        final_cbt_plan = current_draft
    elif human_action == 'Reject':
        # The agent ran a revision cycle and halted again for review
        status = "halted" 
        is_complete = False
    else:
        status = "halted" 
        is_complete = False


    return SessionStatus(
        thread_id=req.thread_id,
        is_complete=is_complete,
        status=status, 
        current_draft=current_draft,
        final_cbt_plan=final_cbt_plan,
        safety_metric=final_state.get('safety_metric'),
        empathy_metric=final_state.get('empathy_metric'),
        model_choice=final_state.get('model_choice')
    )


if __name__ == "__main__":
    uvicorn.run(app_api, host="0.0.0.0", port=8000)