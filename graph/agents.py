# graph/agents.py (FIXED: Prompt Template Invocation)

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
    critic_notes = state.get("critic_notes")
    safety_report = state.get("safety_report")
    current_intent = state["user_intent"] # Grab the current user_intent
    
    critic_feedback = critic_notes.notes if critic_notes and hasattr(critic_notes, 'notes') else 'None'
    safety_feedback = safety_report.feedback if safety_report and hasattr(safety_report, 'feedback') else 'None'

    # --- NEW LOGIC: Determine Revision Source ---
    is_human_revision = current_intent.startswith("REVISION INSTRUCTION:")
    
    # 1. Prepare Base Context
    context = (
        f"You are a CBT exercise creator. Your task is to generate a comprehensive "
        f"Cognitive Behavioral Therapy (CBT) exercise based on the user's intent, "
        f"adhering to strict safety, empathy, and clinical best practices. "
        f"The final output must be engaging and non-directive."
    )
    
    task_instruction: str
    
    if is_human_revision:
        # --- PRIORITY 1: Human Rejection Feedback ---
        revision_instruction = current_intent
        context += (
            f"\n\n--- CRITICAL REVISION TASK: HUMAN OVERRIDE ---"
            f"\nThe previous draft was REJECTED. Your ONLY task is to address the following "
            f"human instruction and apply changes directly to the EXISTING DRAFT. "
            f"\n\nEXISTING DRAFT TO REVISE:\n{current_draft[:200]}..."
            f"\n\n--- HUMAN INSTRUCTION ---\n{revision_instruction}"
            f"\n\nNOTE: IGNORE prior LLM-generated feedback (Critic/Safety) if it conflicts with the human's request."
        )
        task_instruction = f"REVISE THE DRAFT using the instruction: {revision_instruction}"
        
    elif state.get("iteration_count", 0) > 0:
        # --- PRIORITY 2: Internal LLM Feedback (Safety/Critic) ---
        context += (
            f"\n\n--- INTERNAL REVISION TASK ---"
            f"\nPrevious Draft Summary: {current_draft[:200]}..."
            f"\n\nCRITIC NOTES (Focus on empathy/structure): {critic_feedback}" 
            f"\n\nSAFETY REPORT (Focus on compliance/risk): {safety_feedback}"
            f"\n\nCRITICAL TASK: Address all negative feedback, increase empathy, and remove any clinical safety issues. "
            f"Generate a significantly improved draft."
        )
        task_instruction = f"REVISE THE DRAFT based on internal feedback for user intent: {state['user_intent']}"
        
    else:
        # --- PRIORITY 3: Initial Draft ---
        task_instruction = f"GENERATE the initial draft for user intent: {state['user_intent']}"


    # Use the structured intent for the prompt
    drafting_prompt = ChatPromptTemplate.from_messages([
        ("system", context),
        # Pass the original user intent/the human revision instruction here
        ("human", "Core Task/Instruction: {task_instruction}"), 
    ])
    
    # LLM Call
    llm_chain = get_llm_chain(state["model_choice"]) 
    
    # Render the prompt template with the actual task instruction
    rendered_prompt = drafting_prompt.invoke({"task_instruction": task_instruction})
    
    # Invoke the chain with the rendered messages
    new_draft = llm_chain.invoke(rendered_prompt).content

    # Update History
    draft_history = state.get("draft_history", [])
    if current_draft:
        draft_history.append(current_draft)

    return {
        "current_draft": new_draft,
        "draft_history": draft_history,
        "iteration_count": state.get("iteration_count", 0) + 1,
        "model_choice": state["model_choice"],
        "active_node": "Drafting",
        # IMPORTANT: When a human rejected the draft, 'user_intent' was set to "REVISION INSTRUCTION:...",
        # We must reset 'human_decision' so the HIL router correctly pauses again for the next review.
        "human_decision": "REVIEW_REQUIRED", 
        # Optional: You might want to reset the user_intent to the ORIGINAL prompt after the revision is done, 
        # but leaving it as the revision instruction ensures logging clarity.
    }

# Safety Team Agent Node
def safety_agent_node(state: ProjectState) -> Dict[str, Any]:
    """
    Checks the draft for safety compliance using LLM for nuance and NLTK for deterministic rule-checking.
    """
    print("--- Running Safety Team ---")
    draft = state["current_draft"]
    
    # LLM Check Prompt
    safety_check_prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are a safety reviewer. Review the draft for medical advice, clinical overreach, "
         "or inappropriate language. Rate safety from 0.0 (unsafe) to 1.0 (perfectly safe). "
         "Respond ONLY with the required JSON schema."
        ),
        ("human", "Draft to review: {draft_content}"),
    ])
    
    # --- FIX: Render the Prompt Template before invoking ---
    llm_chain = get_llm_chain(state["model_choice"], output_schema=SafetyReport)
    
    rendered_prompt = safety_check_prompt.invoke({"draft_content": draft})
    safety_output: SafetyReport = llm_chain.invoke(rendered_prompt)
    # --- END FIX ---
    
    # NLTK Rule-Based Check (safety check)
    normalized_draft = draft.lower()
    safety_violations = []
    
    for term in PROHIBITED_TERMS:
        if term in normalized_draft:
            safety_violations.append(f"Contains prohibited phrase: '{term}'")

    # Apply severe penalty if rules are broken
    if safety_violations:
        print(f"!!! SAFETY VIOLATIONS FOUND: {safety_violations}")
        safety_output.safety_score = min(safety_output.safety_score, 0.2) 
        
        if not safety_output.feedback:
            safety_output.feedback = []
            
        safety_output.feedback.extend(safety_violations)
        
    # Final Metric Determination
    safety_metric = safety_output.safety_score
    
    return {
        "safety_report": safety_output,
        "safety_metric": safety_metric,
        "model_choice": state["model_choice"],
        "active_node": "Safety",
    }


# Clinical Critic Team Agent Node
def critic_agent_node(state: ProjectState) -> Dict[str, Any]:
    """Critiques the draft for empathy, tone, and clinical structure."""
    print("--- Running Clinical Critic Team ---")
    draft = state["current_draft"]

    # LLM Check Prompt
    critic_check_prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "Critique the tone, empathy, and CBT structure of this draft. "
         "Provide actionable feedback for revision. Respond ONLY with the required JSON schema."
        ),
        ("human", "Draft to critique: {draft_content}"),
    ])
    
    # --- FIX: Render the Prompt Template before invoking ---
    llm_chain = get_llm_chain(state["model_choice"], output_schema=CriticNotes)
    
    rendered_prompt = critic_check_prompt.invoke({"draft_content": draft})
    critic_output: CriticNotes = llm_chain.invoke(rendered_prompt)
    # --- END FIX ---
    
    # NLTK Sentiment Check (Deterministic empathy/tone metric)
    sentiment_scores = sid.polarity_scores(draft)
    empathy_score_raw = sentiment_scores['compound']
    empathy_metric = (empathy_score_raw + 1) / 2
    
    print(f"NLTK VADER Sentiment Score: {empathy_score_raw:.2f} -> Empathy Metric: {empathy_metric:.2f}")

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