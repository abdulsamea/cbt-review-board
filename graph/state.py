from typing import TypedDict, List, Optional
from pydantic import BaseModel, Field
from graph.schemas import GraphNode, HumanDecision, ModelChoice

class CriticNotes(BaseModel):
    """Structured feedback provided by the Critic agent."""

    empathy_revision: str = Field(
        description="Specific feedback on tone and empathy. Must be concise."
    )
    structure_revision: str = Field(
        description="Specific feedback on CBT structure adherence."
    )


class SafetyReport(BaseModel):
    """Safety analysis output from the Safety agent."""

    flagged_lines: List[int] = Field(
        description="Line numbers flagged for potential harm or medical advice."
    )
    safety_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Safety rating from 0.0 (unsafe) to 1.0 (safe).",
    )

# Core LangGraph State (Blackboard)
class ProjectState(TypedDict):
    """
    Shared mutable state passed between LangGraph nodes.
    Acts as the system blackboard.
    """

    # --- Identity / Session ---
    user_intent: str
    thread_id: str  # Used for checkpointing and HIL persistence
    model_choice: ModelChoice

    # --- Draft Lifecycle ---
    current_draft: str
    draft_history: List[str]
    iteration_count: int

    # --- Execution Flow ---
    active_node: GraphNode
    next_node: GraphNode  # Set by supervisor routing logic

    # --- Metrics / Reducers ---
    safety_metric: float     # Updated by Safety agent
    empathy_metric: float    # Updated by Critic agent

    # --- Structured Agent Outputs ---
    critic_notes: Optional[CriticNotes]
    safety_report: Optional[SafetyReport]

    # --- Human-in-the-Loop Control ---
    human_decision: Optional[HumanDecision]
