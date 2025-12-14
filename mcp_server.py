import time
from dotenv import load_dotenv
from graph.supervisor import cbt_review_graph
from graph.state import CriticNotes, SafetyReport
from mcp.server.fastmcp import FastMCP

load_dotenv()


def run_cbt_workflow(
    user_intent: str,
    model_choice: str = "openai",
    thread_id: str = None,
) -> dict:
    """
    Execute the full CBT Review Board LangGraph workflow.
    
    This function automatically handles HIL by auto-approving when the graph
    reaches the HIL node, ensuring seamless execution without halting.

    Args:
        user_intent (str): The main user request or problem statement.
        model_choice (str, optional): Backend model identifier. Default "openai".
        thread_id (str or None, optional): Optional explicit thread identifier.

    Returns:
        dict: Contains keys "thread_id", "current_draft", "final_output",
              "safety_metric", "empathy_metric".
    """
    resolved_thread_id = thread_id or f"mcp_session_{int(time.time())}"

    initial_state = {
        "user_intent": user_intent,
        "thread_id": resolved_thread_id,
        "model_choice": model_choice,
        "current_draft": "",
        "draft_history": [],
        "iteration_count": 0,
        "active_node": "Drafting",
        "next_node": None,
        "safety_metric": 0.0,
        "empathy_metric": 0.0,
        "safety_report": SafetyReport(flagged_lines=[], safety_score=0.0),
        "critic_notes": CriticNotes(empathy_revision="", structure_revision=""),
        "blackboard_notes": [],
        "intent_signals": [],
        "human_decision": None,  # None ensures normal workflow through all nodes
    }

    config = {"configurable": {"thread_id": resolved_thread_id}}
    
    # Use stream with updates mode to monitor node execution
    final_values = None
    hil_detected = False
    
    try:
        # Stream execution and catch HIL node
        for event in cbt_review_graph.stream(
            initial_state, config=config, stream_mode="updates", recursion_limit=100
        ):
            # Check each node update
            for node_name, node_state in event.items():
                if isinstance(node_state, dict):
                    active_node = node_state.get("active_node")
                    human_decision = node_state.get("human_decision")
                    
                    # If HIL_Node just executed and no decision set, auto-approve
                    if node_name == "HIL_Node" and human_decision is None and not hil_detected:
                        print("MCP: Auto-approving at HIL node for seamless execution")
                        hil_detected = True
                        # Update state with approval and continue
                        updated_state = {**node_state, "human_decision": "Approve"}
                        # Continue execution from this point
                        try:
                            final_values = cbt_review_graph.invoke(
                                updated_state, config=config, recursion_limit=100
                            )
                            break
                        except Exception as e:
                            print(f"Error continuing after HIL: {e}")
                            # Try update_state method
                            try:
                                cbt_review_graph.update_state(config, {"human_decision": "Approve"})
                                final_values = cbt_review_graph.invoke(
                                    None, config=config, recursion_limit=100
                                )
                                break
                            except Exception:
                                pass
                    
                    # If we've reached END, we're done
                    if active_node == "END" or node_name == "__end__":
                        final_values = node_state
                        break
            
            if final_values is not None:
                break
                
    except Exception as e:
        print(f"Error in streaming: {e}")
    
    # Fallback: check checkpoint if we didn't get final values
    if final_values is None:
        checkpoint = cbt_review_graph.checkpointer.get(config)
        if checkpoint:
            channel_values = checkpoint.get("channel_values", {})
            if isinstance(channel_values, dict):
                # Handle different checkpoint formats
                if "__root__" in channel_values:
                    root_data = channel_values["__root__"].get("value")
                    state = root_data if isinstance(root_data, dict) else channel_values
                else:
                    state = channel_values
                
                # If stuck at HIL, auto-approve and continue
                if isinstance(state, dict):
                    active_node = state.get("active_node")
                    human_decision = state.get("human_decision")
                    
                    if active_node == "HIL_Node" and human_decision is None:
                        print("MCP: Auto-approving at HIL node (checkpoint fallback)")
                        updated_state = {**state, "human_decision": "Approve"}
                        final_values = cbt_review_graph.invoke(
                            updated_state, config=config, recursion_limit=100
                        )
                    elif active_node == "END":
                        final_values = state
                    else:
                        final_values = state
            else:
                final_values = channel_values
        else:
            # Last resort: try regular invoke
            try:
                final_values = cbt_review_graph.invoke(
                    initial_state, config=config, recursion_limit=100
                )
            except Exception as e:
                print(f"Error in final invoke: {e}")
                # Return empty state as fallback
                final_values = {}

    return {
        "thread_id": resolved_thread_id,
        "current_draft": final_values.get("current_draft", ""),
        "final_output": final_values.get("final_output"),
        "safety_metric": final_values.get("safety_metric", 0.0),
        "empathy_metric": final_values.get("empathy_metric", 0.0),
    }


server = FastMCP("cbt-review-board")


@server.tool()
async def cerina_foundry_cbt_protocol(
    prompt: str,
    model_choice: str = "openai",
    thread_id: str = None,
) -> dict:
    """
    MCP Tool: cerina_foundry_cbt_protocol

    Runs the clinically-informed CBT Review Board workflow to create safe,
    refined CBT-style protocols (e.g., sleep hygiene guides, anxiety reframing steps, etc.).
    
    This tool automatically handles the Human-in-the-Loop step by auto-approving,
    ensuring seamless execution without requiring manual intervention.

    Args:
        prompt (str): The human user intent or request to process.
        model_choice (str, optional): Backend model identifier for LangGraph LLM calls. 
            Options: "openai", "groq", "ollama". Defaults to "openai".
        thread_id (str or None, optional): Explicit identifier to ensure continuity across sessions.

    Returns:
        dict: {
            "thread_id": str,
            "draft": str|None,
            "metrics": {"safety_metric": float, "empathy_metric": float},
            "final_output": str|None
        }
    """
    result = run_cbt_workflow(
        user_intent=prompt,
        model_choice=model_choice,
        thread_id=thread_id,
    )

    return {
        "thread_id": result["thread_id"],
        "draft": result["current_draft"],
        "metrics": {
            "safety_metric": result["safety_metric"],
            "empathy_metric": result["empathy_metric"],
        },
        "final_output": result.get("final_output"),
    }


if __name__ == "__main__":
    server.run()
