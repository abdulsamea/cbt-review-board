import os
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool
from graph.state import ProjectState
from graph.agents import drafting_agent_node, safety_agent_node, critic_agent_node, hil_node
from dotenv import load_dotenv

load_dotenv()

# Define Conditional Edge Logic
def route_safety_check(state: ProjectState) -> str:
    # Autonomous loop check (Self-correction)
    SAFETY_THRESHOLD = 0.95 
    if state["safety_metric"] < SAFETY_THRESHOLD:
        return "Drafting" # Loop back for revision
    return "Critic"

def route_critic_check(state: ProjectState) -> str:
    # Autonomous loop check (Internal Debate)
    EMPATHY_THRESHOLD = 0.70
    if state["empathy_metric"] < EMPATHY_THRESHOLD:
        return "Drafting" # Loop back for revision
    
    # Human-in-the-Loop Trigger
    if state["iteration_count"] >= 1 and state["safety_metric"] >= 0.95:
        return "HIL_Node" # Ready for human review
    
    return "Drafting"

def route_human_decision(state: ProjectState) -> str:
    # Routing based on external human input
    if state["human_decision"] == "Approve":
        return "Finalize"
    return "Drafting" # Reject/Revise loops back

# Checkpointer Setup (for Persistence)
def get_checkpointer():
    DB_URI = os.getenv("POSTGRES_CONNECTION_STRING")
    if not DB_URI:
        # Fallback for testing, but NOT production
        from langgraph.checkpoint.memory import MemorySaver
        return MemorySaver()
    
    # Production-ready PostgreSQL checkpointer
    pool = ConnectionPool(conninfo=DB_URI, max_size=10)
    return PostgresSaver(pool)


# Graph Compilation
def compile_supervisor_graph():
    workflow = StateGraph(ProjectState)

    # Add Nodes (Agents)
    workflow.add_node("Drafting", drafting_agent_node)
    workflow.add_node("Safety", safety_agent_node)
    workflow.add_node("Critic", critic_agent_node)
    workflow.add_node("HIL_Node", hil_node)
    workflow.add_node("Finalize", lambda s: {"next_node": "END"}) # Dummy node for final saving

    # Set Entry Point and Initial Edges
    workflow.set_entry_point("Drafting")
    workflow.add_edge("Drafting", "Safety") # Linear transition
    
    # Add Conditional Edges (Supervisor Logic)
    workflow.add_conditional_edges(
        "Safety", # From Safety Node
        route_safety_check, # Conditional router function
        {"Drafting": "Drafting", "Critic": "Critic"},
    )
    workflow.add_conditional_edges(
        "Critic", # From Critic Node
        route_critic_check, # Conditional router function
        {"Drafting": "Drafting", "HIL_Node": "HIL_Node"},
    )
    workflow.add_conditional_edges(
        "HIL_Node", # From HIL Node (after resume)
        route_human_decision, # Conditional router function
        {"Approve": "Finalize", "Reject": "Drafting"},
    )
    
    # Set End Point
    workflow.add_edge("Finalize", END)
    
    # Compile with Persistence
    app = workflow.compile(checkpointer=get_checkpointer())
    return app

# Initialize the production graph
cbt_review_graph = compile_supervisor_graph()