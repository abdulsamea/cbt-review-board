from typing import Dict, Any
from graph.state import ProjectState, CriticNotes, SafetyReport
from graph.llm_config import get_llm_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

from nltk.sentiment.vader import SentimentIntensityAnalyzer
# Initialize NLTK Sentiment Analyzer
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


# Drafting Team Agent Node
def drafting_agent_node(state: ProjectState) -> Dict[str, Any]:
    """Generates the initial CBT exercise draft or revises it based on feedback."""
    print(f"--- Running Drafting Team (Iteration: {state.get('iteration_count', 0) + 1}) ---")
    
    current_draft = state.get("current_draft", "")
    critic_notes = state.get("critic_notes", {})
    safety_report = state.get("safety_report", {})

    # Prepare Context and Prompt
    context = (
        f"You are a CBT exercise creator. Your task is to generate a comprehensive "
        f"Cognitive Behavioral Therapy (CBT) exercise based on the user's intent, "
        f"adhering to strict safety, empathy, and clinical best practices. "
        f"The final output must be engaging and non-directive."
    )
    
    # Add revision context if this is not the first iteration
    if state.get("iteration_count", 0) > 0:
        context += (
            f"\n\n--- REVISION HISTORY/FEEDBACK ---"
            f"\nPrevious Draft Summary: {current_draft[:200]}..."
            f"\n\nCRITIC NOTES (Focus on empathy/structure): {critic_notes.get('notes', 'None')}"
            f"\n\nSAFETY REPORT (Focus on compliance/risk): {safety_report.get('feedback', 'None')}"
            f"\n\nCRITICAL TASK: Address all negative feedback, increase empathy, and remove any clinical safety issues. "
            f"Generate a significantly improved draft."
        )

    # Use the structured intent for the prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", context),
        ("human", f"User Intent: {state['user_intent']}"),
    ])
    
    # LLM Call
    llm_chain = get_llm_chain(state["model_choice"]) 
    new_draft = llm_chain.invoke(prompt).content

    # Update History
    draft_history = state.get("draft_history", [])
    if current_draft:
        draft_history.append(current_draft)

    return {
        "current_draft": new_draft,
        "draft_history": draft_history,
        "iteration_count": state.get("iteration_count", 0) + 1,
        "model_choice": state["model_choice"],
    }


# Safety Team Agent Node
def safety_agent_node(state: ProjectState) -> Dict[str, Any]:
    """
    Checks the draft for safety compliance using LLM for nuance and NLTK for deterministic rule-checking.
    (Replaces spaCy with NLTK)
    """
    print("--- Running Safety Team ---")
    draft = state["current_draft"]
    
    #  LLM Check (For scoring/critique)
    llm_chain = get_llm_chain(state["model_choice"], output_schema=SafetyReport)
    safety_output: SafetyReport = llm_chain.invoke(
        f"Review the following CBT draft for any medical advice, clinical overreach, "
        f"or inappropriate language. Rate safety from 0.0 (unsafe) to 1.0 (perfectly safe). "
        f"Draft: {draft}"
    )
    
    # NLTK Rule-Based Check (safety check)
    normalized_draft = draft.lower()
    safety_violations = []
    
    # Check for prohibited multi-word phrases (e.g., "take this medication")
    for term in PROHIBITED_TERMS:
        if term in normalized_draft:
            safety_violations.append(f"Contains prohibited phrase: '{term}'")

    # Apply severe penalty if rules are broken
    if safety_violations:
        print(f"!!! SAFETY VIOLATIONS FOUND: {safety_violations}")
        # Set a low score to force a revision
        safety_output.safety_score = min(safety_output.safety_score, 0.2) 
        
        # Add the rule violation to the LLM report's feedback list
        if not safety_output.feedback:
            safety_output.feedback = []
            
        safety_output.feedback.extend(safety_violations)
        
    # Final Metric Determination
    safety_metric = safety_output.safety_score
    
    return {
        "safety_report": safety_output,
        "safety_metric": safety_metric,
        "model_choice": state["model_choice"],
    }


# Clinical Critic Team Agent Node
def critic_agent_node(state: ProjectState) -> Dict[str, Any]:
    """Critiques the draft for empathy, tone, and clinical structure."""
    print("--- Running Clinical Critic Team ---")
    draft = state["current_draft"]

    # LLM Check (For structured feedback on tone, empathy, and structure)
    llm_chain = get_llm_chain(state["model_choice"], output_schema=CriticNotes)
    critic_output: CriticNotes = llm_chain.invoke(
        f"Critique the tone, empathy, and CBT structure of this draft. "
        f"Provide actionable feedback for revision. Draft: {draft}"
    )
    
    # NLTK Sentiment Check (Deterministic empathy/tone metric)
    
    # Use VADER (Valence Aware Dictionary and sentiment Reasoner) for a simple, fast sentiment score
    sentiment_scores = sid.polarity_scores(draft)
    
    # We use the 'compound' score as a proxy for positive/empathetic tone. 
    # Scores range from -1 (Extremely Negative) to +1 (Extremely Positive).
    empathy_score_raw = sentiment_scores['compound']
    
    # Normalize score to 0.0-1.0 metric (LangGraph expects this range)
    # Formula: (score + 1) / 2
    empathy_metric = (empathy_score_raw + 1) / 2
    
    # Final Metric Assignment
    print(f"NLTK VADER Sentiment Score: {empathy_score_raw:.2f} -> Empathy Metric: {empathy_metric:.2f}")

    # If the LLM critic score is low (e.g., critic_output.score is not provided, use a threshold)
    # We use the deterministic NLTK score as the final routing metric.

    return {
        "critic_notes": critic_output,
        "empathy_metric": empathy_metric,
        "model_choice": state["model_choice"],
    }


# Finalize Node
def finalize_node(state: ProjectState) -> Dict[str, Any]:
    """Finalizes the project and prepares the final approved output."""
    print("--- Running Finalize Node: Approval Granted ---")
    
    # In a production system, this node would save the final draft 
    # to a persistent application database (not the checkpointer).
    
    final_output = {
        "final_status": "APPROVED",
        "final_draft": state["current_draft"],
        "total_iterations": state.get("iteration_count", 0),
        "safety_score": state.get("safety_metric"),
        "empathy_score": state.get("empathy_metric"),
        "final_report": state.get("safety_report"),
    }
    
    print("\n\n PROJECT FINALIZED AND APPROVED.")
    
    # This node needs to return the updated state, even though it's the end of the graph.
    return {"current_draft": state["current_draft"], "final_output": final_output}


# Human-in-the-Loop Node
def hil_node(state: ProjectState) -> Dict[str, Any]:
    """Pauses the graph execution and waits for a human decision (Approve/Reject)."""
    print("--- Running HIL Node: Awaiting Human Review ---")
    # LangGraph will automatically pause the thread here due to the `interrupt()` mechanism in supervisor.py
    
    return {"next_node": "HIL_Node"} # Does not change the state flow, just signals the node name