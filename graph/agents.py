from typing import Dict, Any
from graph.state import ProjectState, CriticNotes, SafetyReport
from graph.llm_config import get_llm_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from graph.tools.nhs_cbt_manual_retriever import get_nhs_manual_retriever_tool


from nltk.sentiment.vader import SentimentIntensityAnalyzer

sid = SentimentIntensityAnalyzer()

load_dotenv()

# Terms that constitute unauthorized or unsafe medical/clinical advice.
PROHIBITED_TERMS = {
    "take this medication", 
    "discontinue treatment", 
    "contact your doctor immediately",
    "prescription", 
    "diagnosis",
    "dosage",
    "cure for"
}

def get_llm_with_nhs_tool(model_choice, output_schema=None):
    """
    Returns an LLM chain configured with the NHS CBT manual retriever tool.
    """
    nhs_tool = get_nhs_manual_retriever_tool()
    return get_llm_chain(
        model_choice,
        output_schema=output_schema,
        tools=[nhs_tool],
    )



# Drafting Team Agent Node
def drafting_agent_node(state: ProjectState) -> Dict[str, Any]:
    print(f"--- Running Drafting Team (Iteration: {state.get('iteration_count', 0) + 1}) ---")

    current_draft = state.get("current_draft", "")
    critic_notes = state.get("critic_notes")
    safety_report = state.get("safety_report")
    current_intent = state["user_intent"]

    critic_feedback = critic_notes.notes if critic_notes and hasattr(critic_notes, 'notes') else 'None'
    safety_feedback = safety_report.feedback if safety_report and hasattr(safety_report, 'feedback') else 'None'

    is_human_revision = current_intent.startswith("REVISION INSTRUCTION:")

    context = (
        "You are a CBT exercise creator. Your task is to generate a comprehensive "
        "Cognitive Behavioral Therapy (CBT) exercise based on the user's intent, "
        "adhering to strict safety, empathy, and clinical best practices. "
        "The final output must be engaging and non-directive.\n\n"
        "IMPORTANT:\n"
        "- Use the NHS Talking Therapies manual as an authoritative guideline.\n"
        "- Follow stepped care, assessment boundaries, and non-clinical framing.\n"
        "- Do NOT quote the manual verbatim.\n"
        "- Do NOT provide diagnosis or treatment instructions."
    )

    if is_human_revision:
        context += (
            "\n\n--- CRITICAL REVISION TASK: HUMAN OVERRIDE ---"
            "\nThe previous draft was REJECTED. Your ONLY task is to apply the human instruction "
            "to the existing draft while remaining NHS-compliant."
        )
        task_instruction = f"REVISE THE DRAFT using the instruction: {current_intent}"

    elif state.get("iteration_count", 0) > 0:
        context += (
            "\n\n--- INTERNAL REVISION TASK ---"
            f"\nCRITIC NOTES: {critic_feedback}"
            f"\nSAFETY REPORT: {safety_feedback}"
            "\nAddress all issues while remaining compliant with NHS CBT guidance."
        )
        task_instruction = f"REVISE the draft for user intent: {state['user_intent']}"

    else:
        task_instruction = f"GENERATE the initial draft for user intent: {state['user_intent']}"

    drafting_prompt = ChatPromptTemplate.from_messages([
        ("system", context),
        ("human", "Core Task/Instruction: {task_instruction}"),
    ])

    llm_chain = get_llm_chain(state["model_choice"])

    rendered_prompt = drafting_prompt.invoke({"task_instruction": task_instruction})
    new_draft = llm_chain.invoke(rendered_prompt).content

    draft_history = state.get("draft_history", [])
    if current_draft:
        draft_history.append(current_draft)

    return {
        "current_draft": new_draft,
        "draft_history": draft_history,
        "iteration_count": state.get("iteration_count", 0) + 1,
        "model_choice": state["model_choice"],
        "active_node": "Drafting",
        "human_decision": "REVIEW_REQUIRED",
    }

# Safety Team Agent Node
def safety_agent_node(state: ProjectState) -> Dict[str, Any]:
    print("--- Running Safety Team ---")
    draft = state["current_draft"]

    safety_check_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a safety reviewer for CBT content.\n"
         "- Use the NHS Talking Therapies manual as the safety authority.\n"
         "- Identify clinical overreach, risk escalation, or medical advice.\n"
         "Respond ONLY with the required JSON schema."
        ),
        ("human", "Draft to review: {draft_content}"),
    ])

    llm_chain = get_llm_with_nhs_tool(
        state["model_choice"],
        output_schema=SafetyReport
    )

    rendered_prompt = safety_check_prompt.invoke({"draft_content": draft})
    safety_output: SafetyReport = llm_chain.invoke(rendered_prompt)

    normalized_draft = draft.lower()
    safety_violations = []

    for term in PROHIBITED_TERMS:
        if term in normalized_draft:
            safety_violations.append(f"Contains prohibited phrase: '{term}'")

    if safety_violations:
        safety_output.safety_score = min(safety_output.safety_score, 0.2)
        safety_output.feedback = (safety_output.feedback or []) + safety_violations

    return {
        "safety_report": safety_output,
        "safety_metric": safety_output.safety_score,
        "model_choice": state["model_choice"],
        "active_node": "Safety",
    }


# Clinical Critic Team Agent Node
def critic_agent_node(state: ProjectState) -> Dict[str, Any]:
    print("--- Running Clinical Critic Team ---")
    draft = state["current_draft"]

    critic_check_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a CBT clinical critic.\n"
         "- Use the NHS Talking Therapies manual as the reference standard.\n"
         "- Evaluate empathy, tone, CBT structure, and appropriateness.\n"
         "Respond ONLY with the required JSON schema."
        ),
        ("human", "Draft to critique: {draft_content}"),
    ])

    llm_chain = get_llm_with_nhs_tool(
        state["model_choice"],
        output_schema=CriticNotes
    )

    rendered_prompt = critic_check_prompt.invoke({"draft_content": draft})
    critic_output: CriticNotes = llm_chain.invoke(rendered_prompt)

    sentiment_scores = sid.polarity_scores(draft)
    empathy_metric = (sentiment_scores['compound'] + 1) / 2

    print(
        f"NLTK Sentiment: {sentiment_scores['compound']:.2f} "
        f"-> Empathy Metric: {empathy_metric:.2f}"
    )

    return {
        "critic_notes": critic_output,
        "empathy_metric": empathy_metric,
        "model_choice": state["model_choice"],
        "active_node": "Critic",
    }


# Finalize Node
def finalize_node(state: ProjectState) -> Dict[str, Any]:
    """Finalizes the project and prepares the final approved output."""
    print("--- Running Finalize Node: Approval Granted ---")
    
    final_output = {
        "final_status": "APPROVED",
        "final_draft": state["current_draft"],
        "total_iterations": state.get("iteration_count", 0),
        "safety_score": state.get("safety_metric"),
        "empathy_score": state.get("empathy_metric"),
        "final_report": state.get("safety_report"),
    }
    
    print("\n\n PROJECT FINALIZED AND APPROVED.")
    
    return {"current_draft": state["current_draft"], "final_output": final_output, "active_node": "Finalize"}


# Human-in-the-Loop Node
def hil_node(state: ProjectState) -> Dict[str, Any]:
    """Pauses the graph execution and waits for a human decision (Approve/Reject)."""
    print("--- Running HIL Node: Awaiting Human Review ---")
    
    return {"next_node": "HIL_Node", "active_node": "HIL_Node"}