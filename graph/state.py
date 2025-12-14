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
from typing import TypedDict, List, Dict, Optional, Literal
from graph.schemas import HumanDecision, ModelChoice
from graph.state import CriticNotes, SafetyReport

GraphNode = Literal[
    "Drafting",
    "Safety",
    "Critic",
    "HIL_Node",
    "Finalize",
    "END",
]

AgentName = Literal["Drafting", "Safety", "Critic", "Supervisor"]

class BlackboardNote(TypedDict):
    agent: AgentName
    iteration: int
    severity: Literal["info", "warning", "blocker"]
    message: str
    resolved: bool

class IntentSignal(TypedDict):
    from_agent: AgentName
    to_agent: AgentName
    intent: Literal[
        "revise_for_safety",
        "revise_for_empathy",
        "revise_structure",
        "human_review_required",
        "ready_to_finalize",
    ]
    reason: str

class ProjectState(TypedDict):
    """
    Shared mutable state passed between LangGraph nodes.
    Acts as a true multi-agent blackboard.
    """

    # Identity / Session
    user_intent: str
    thread_id: str
    model_choice: ModelChoice

    # Draft Lifecycle
    current_draft: str
    draft_history: List[str]
    iteration_count: int

    # Execution Flow
    active_node: GraphNode
    next_node: Optional[GraphNode]

    # Metrics
    safety_metric: float
    empathy_metric: float

    # Structured Agent Outputs
    critic_notes: Optional[CriticNotes]
    safety_report: Optional[SafetyReport]

    #  BLACKBOARD
    blackboard_notes: List[BlackboardNote]      # additive, persistent
    intent_signals: List[IntentSignal]          # explicit agent → agent intent

    # Human-in-the-Loop
    human_decision: Optional[HumanDecision]
