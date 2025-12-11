# main.py

import os
import time
from typing import Dict, Any, Optional
from graph.supervisor import compile_supervisor_graph, get_checkpointer
from graph.state import ProjectState
from dotenv import load_dotenv

load_dotenv()

# --- Main Execution Logic ---

def run_supervisor_agent(user_prompt: str, thread_id: str, model_choice: str):
    """Initializes and runs the supervisor graph for a given user intent."""
    
    print(f"--- Setting up Thread (ID: {thread_id}) using model: {model_choice} ---")
    
    # 1. Initialize Graph (The checkpointer is compiled inside supervisor.py)
    # This call will crash the script if SQLite persistence fails, as intended.
    app = compile_supervisor_graph()

    # 2. Define Initial State
    initial_state: ProjectState = {
        "user_intent": user_prompt,
        "model_choice": model_choice,
        "current_draft": "",
        "draft_history": [],
        "iteration_count": 0,
        "safety_metric": 0.0,
        "empathy_metric": 0.0,
        "safety_report": {},
        "critic_notes": {},
        "human_decision": "N/A", # Initialize the new required field
        "final_output": None
    }

    # 3. Invoke the Graph and Stream Results
    print(f"\n--- Starting Graph Execution ---")

    config = {"configurable": {"thread_id": thread_id}}

    try:
        # Use stream to show progress as nodes execute
        for state in app.stream(
            input=initial_state,
            config=config,
            stream_mode="updates" 
        ):
            # Print the output of the last executed node
            node_name = list(state.keys())[0]
            print(f"  [NODE] -> {node_name}")

            # --- Handle Human-in-the-Loop Interruption ---
            if node_name == "HIL_Node":
                print("\n\n!!! GRAPH PAUSED FOR HUMAN INTERVENTION (HIL) !!!")
                print(f"Review Thread ID: {thread_id}")
                
                # Auto-approve for non-interactive testing purposes
                print("--- Simulating HIL Approval and Resuming ---")
                
                # Get the last saved state
                last_state = app.get_state(config)
                
                # Manually inject the human decision and resume
                # NOTE: This is done manually here to test the resume logic.
                last_state.values['human_decision'] = "Approve" 
                
                # Resume execution from the last node that ran (HIL_Node)
                # We need to tell the graph to start at the HIL_Node again so it hits the router.
                final_output = app.invoke(
                    last_state.values, 
                    config=config, 
                    # The next step must be HIL_Node to hit the router
                    recursion_limit=100
                )
                
                # Exit the stream loop after the manual invoke finishes
                break 

    except Exception as e:
        print(f"\n\n!!! GRAPH EXECUTION FAILED: {e}")
        return

    # 4. Final Result Reporting
    final_checkpoint_state = app.get_state(config)
    
    if final_checkpoint_state and final_checkpoint_state.next == ('__end__',):
        print("\n\n############################################")
        print("✅ PROJECT FINALIZED AND APPROVED.")
        print(f"Total Iterations: {final_checkpoint_state.values.get('iteration_count')}")
        print("FINAL APPROVED DRAFT:")
        print(final_checkpoint_state.values.get("current_draft"))
        print("############################################")
    else:
        print("\n\n--- Graph stopped or paused (Not finalized) ---")
        if final_checkpoint_state:
             print(f"Next expected node: {final_checkpoint_state.next}")


# --- Run the Project ---

if __name__ == "__main__":
    thread_id = f"session_{int(time.time())}"
    
    user_prompt = (
        "I need a short, practical CBT exercise to manage my anxiety "
        "when I feel overwhelmed by work deadlines. Focus on a cognitive "
        "reframing technique."
    )
    
    run_supervisor_agent(user_prompt, thread_id, model_choice="openai")