import time

from dotenv import load_dotenv

from graph.supervisor import compile_supervisor_graph

from mcp.server.fastmcp import FastMCP

load_dotenv()


def run_cbt_workflow(
    user_intent: str,
    model_choice: str = "openai",
    thread_id: str = None,
) -> dict:
    """
    Execute the full CBT Review Board LangGraph workflow.

    This function is called internally by the MCP tool wrapper.
    It sets up the initial state, executes the LangGraph, and returns
    the final output, metrics, and thread identifier.

    Args:
        user_intent (str): The main user request or problem statement.
        model_choice (str, optional): Backend model identifier. Default "openai".
        thread_id (str or None, optional): Optional explicit thread identifier.

    Returns:
        dict: Contains keys "thread_id", "current_draft", "final_output",
              "safety_metric", "empathy_metric".
    """
    app = compile_supervisor_graph()
    resolved_thread_id = thread_id or f"mcp_session_{int(time.time())}"

    initial_state = {
        "user_intent": user_intent,
        "model_choice": model_choice,
        "current_draft": "",
        "draft_history": [],
        "iteration_count": 0,
        "safety_metric": 0.0,
        "empathy_metric": 0.0,
        "safety_report": {},
        "critic_notes": {},
        "human_decision": "Approve",
    }

    config = {"configurable": {"thread_id": resolved_thread_id}}
    final_values = app.invoke(initial_state, config=config, recursion_limit=100)

    return {
        "thread_id": resolved_thread_id,
        "current_draft": final_values.get("current_draft"),
        "final_output": final_values.get("final_output"),
        "safety_metric": final_values.get("safety_metric"),
        "empathy_metric": final_values.get("empathy_metric"),
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

    Args:
        prompt (str): The human user intent or request to process.
        model_choice (str, optional): Backend model identifier for LangGraph LLM calls. If none is provided, default to openai
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
