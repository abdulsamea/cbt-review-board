from typing import Literal


ModelChoice = Literal["openai", "groq", "ollama"]

GraphNode = Literal[
    "Drafting",
    "Safety",
    "Critic",
    "HIL_Node",
    "Finalize",
]

HumanDecision = Literal["Approve", "Reject"]
