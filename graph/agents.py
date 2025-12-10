from graph.state import ProjectState, CriticNotes, SafetyReport
from graph.llm_config import get_llm_chain
from langchain_core.messages import SystemMessage, HumanMessage
import spacy
from typing import Dict, Any

# Utilities Setup
nlp = spacy.load("en_core_web_sm") 
# TODO: Wrap spaCy in a LangChain Tool for proper integration

# Drafting Agent Node.
def drafting_agent_node(state: ProjectState) -> Dict[str, Any]:
    # Versioning: Update draft_history
    draft_history = state.get("draft_history", [])
    if state["current_draft"]:
        draft_history.append(state["current_draft"])
    
   
    # The prompt incorporates feedback for self-correction
    feedback = f"Prior Safety Report: {state['safety_report'].dict() if state.get('safety_report') else 'None'}. "
    feedback += f"Prior Critic Notes: {state['critic_notes'].dict() if state.get('critic_notes') else 'None'}."
    
    prompt = [
        SystemMessage(f"You are a CBT Exercise Drafter. Create a safe, structured, and empathetic CBT exercise based on the user intent. Adhere strictly to the required Pydantic schema. Previous revision feedback: {feedback}"),
        HumanMessage(content=f"User Goal: {state['user_intent']}"),
    ]
    
    # LLM Call
    llm_chain = get_llm_chain("openai")
    new_draft = llm_chain.invoke(prompt).content

    # Using spaCy for grammar / style check
    doc = nlp(new_draft)
    if sum(1 for token in doc if token.dep_ == "auxpass") > 5:
        print("Drafting Agent detected passive voice overuse. Simple self-correction applied.")
    
    return {
        "current_draft": new_draft,
        "draft_history": draft_history,
        "iteration_count": state.get("iteration_count", 0) + 1,
    }

# Safety Team Agent Node
def safety_agent_node(state: ProjectState) -> Dict[str, Any]:
    draft = state["current_draft"]
    
    llm_chain = get_llm_chain("openai", output_schema=SafetyReport)
    safety_output = llm_chain.invoke(f"Review the following CBT draft for any medical advice, unsafe language, or self-harm triggers. Draft: {draft}")
    
    # 2.spaCy check for medical terms (Deterministic Pre-Processing/Validation)
    doc = nlp(draft)
    medical_entities = [ent.text for ent in doc.ents if ent.label_ in ["DRUG", "MEDICAL_CONDITION"]]
    
    if medical_entities:
        safety_output.safety_score = min(safety_output.safety_score, 0.5) # Enforce penalty
        safety_output.rationale += f" **Hard Flag:** Detected unverified medical entities: {', '.join(medical_entities)}"

    return {
        "safety_report": safety_output,
        "safety_metric": safety_output.safety_score,
    }

# Clinical Critic Team Agent Node
def critic_agent_node(state: ProjectState) -> Dict[str, Any]:
    draft = state["current_draft"]

    # Empathy & Structure Check
    llm_chain = get_llm_chain("openai", output_schema=CriticNotes)
    critic_output = llm_chain.invoke(f"Critique the tone, empathy, and CBT structure of this draft. Draft: {draft}")
    
    # 2. NLTK Sentiment Check (NLP Utility for Empathy Metric)
    # Quick, deterministic check to anchor LLM's empathy score
    from nltk.sentiment import SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()
    sentiment = sia.polarity_scores(draft)
    empathy_score = 0.5 + (sentiment['pos'] - sentiment['neg']) * 0.5
    
    return {
        "critic_notes": critic_output,
        "empathy_metric": empathy_score,
    }

# Human-in-the-Loop Node
def hil_node(state: ProjectState) -> Dict[str, Any]:
    # This node is where the graph PAUSES.
    from langgraph.types import interrupt, Command
    
    if state["human_decision"]:
        # Execution resumed with human input
        print(f"Graph resumed with decision: {state['human_decision']}")
        return {"next_node": state["human_decision"]} # e.g., 'Approve' or 'Reject'
    
    # Pause execution and save the state for the UI to pick up
    interrupt_payload = {
        "thread_id": state["thread_id"],
        "draft": state["current_draft"],
        "safety_report": state["safety_report"].model_dump(),
        "critic_notes": state["critic_notes"].model_dump(),
    }
    
    # The interrupt function pauses execution and saves the state to the checkpointer
    # The external caller (FastAPI) receives this payload and waits for a resume command.
    # The resume value is passed back into the node.
    decision = interrupt(interrupt_payload) 
    
    # The line above is a placeholder. In the real async system, the graph's runtime
    # will handle the pause and resume logic via the checkpointer and Command.
    # We return the pause state to the FastAPI runtime to handle the HIL logic.
    return {"next_node": "HIL_PAUSED"} # Fictional state for the API