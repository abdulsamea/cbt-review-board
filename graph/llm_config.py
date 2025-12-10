from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_community.chat_models import ChatOllama
from pydantic import BaseModel
from langchain_core.runnables import Runnable
import os
from dotenv import load_dotenv

load_dotenv()

def get_llm_chain(model_choice: str = 'openai' | 'groq' | 'ollama', output_schema: BaseModel = None) -> Runnable:
    """
    Initializes a swappable LLM chain based on the model_choice, with optional structured output.
    
    Args:
        model_choice: The desired model key ('openai', 'groq', 'ollama').
        output_schema: Optional Pydantic BaseModel for structured output.
    """
    llm = None
    
    # 1. use OpenAI for high clinical quality
    if model_choice == "openai":
        # Using GPT-4o-mini for a balance of quality and speed
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3, api_key=os.getenv("OPENAI_API_KEY")) 
    
    # 2. High-Speed Open LLM (Groq for near-local performance)
    elif model_choice == "groq":
        # Using Mixtral 8x7B for high-quality open-source reasoning
        llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0.3, api_key=os.getenv("GROQ_API_KEY"))
    
    # 3. Local/Hugging Face LLM (True Local Development/Open Source)
    elif model_choice == "ollama":
        # Requires an Ollama instance running locally (e.g., 'ollama run mistral')
        llm = ChatOllama(model="mistral", base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"), temperature=0.3)
    
    else:
        raise ValueError(f"Unknown model choice: {model_choice}. Options are 'openai', 'groq', or 'ollama'.")

    # Structured Output and Logic Control (Pydantic enforcement)
    if output_schema:
        # LangChain handles the appropriate method (JSON mode, tool calling) for the model
        return llm.with_structured_output(output_schema)
    
    return llm