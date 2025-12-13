from typing import Optional, Type
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_community.chat_models import ChatOllama
from pydantic import BaseModel
from langchain_core.runnables import Runnable
import os
from dotenv import load_dotenv
from graph.schemas import ModelChoice

load_dotenv()

def get_llm_chain(
    model_choice: ModelChoice = 'openai',
    output_schema: Optional[Type[BaseModel]] = None,
    tools: list | None = None,
) -> Runnable:
    """
    Initializes a swappable LLM chain based on the model_choice,
    with optional structured output.
    """

    if model_choice == "openai":
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.7,
            api_key=os.getenv("OPENAI_API_KEY"),
        )

    elif model_choice == "groq":
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.7,
            api_key=os.getenv("GROQ_API_KEY"),
        )

    else:  # model_choice == "ollama"
        llm = ChatOllama(
            model="mistral",
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=0.3,
        )

    if tools:
        llm = llm.bind_tools(tools)

    if output_schema:
        return llm.with_structured_output(output_schema)

    return llm
