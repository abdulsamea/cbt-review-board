from typing import Literal, TypedDict, List
from pydantic import BaseModel, Field

# 1. Structured Feedback Models (Used by Critic and Safety Agents)
class CriticNotes(BaseModel):
    empathy_revision: str = Field(description="Specific feedback on tone/empathy, must be concise.")
    structure_revision: str = Field(description="Specific feedback on CBT structure adherence.")

class SafetyReport(BaseModel):
    flagged_lines: List[int] = Field(description="List of line numbers flagged for potential harm or medical advice.")
    safety_score: float = Field(description="Safety rating from 0.0 (unsafe) to 1.0 (safe).")

# 2. The Core LangGraph State (The Blackboard)
class ProjectState(TypedDict):
    user_intent: str
    thread_id: str # Crucial for checkpointing/HIL
    current_draft: str
    draft_history: List[str]
    iteration_count: int
    model_choice: str
    
    # Reducers / Metrics
    safety_metric: float # Updated by Safety Team
    empathy_metric: float # Updated by Critic Team

    # Structured Feedback
    critic_notes: CriticNotes
    safety_report: SafetyReport
    
    # Control variables for routing/HIL
    next_node:  Literal['openai', 'groq', 'ollama'] # Conditional edge result (used by Supervisor)
    human_decision: str # Input from HIL ('Approve' or 'Reject')