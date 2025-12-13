from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.tools import create_retriever_tool

MANUAL_URL = "https://www.england.nhs.uk/wp-content/uploads/2018/06/nhs-talking-therapies-manual-v7.1-updated.pdf"

_vectorstore = None
_retriever = None
_retriever_tool = None

def _build_vectorstore():
    global _vectorstore
    if _vectorstore is not None:
        return _vectorstore

    pdf_path = Path("data/nhs_talking_therapies_manual_v7.1.pdf")
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    if not pdf_path.exists():
        import requests
        resp = requests.get(MANUAL_URL)
        resp.raise_for_status()
        pdf_path.write_bytes(resp.content)

    loader = PyPDFLoader(str(pdf_path))
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
        add_start_index=True,
    )
    splits = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings()
    _vectorstore = InMemoryVectorStore.from_documents(splits, embeddings)
    return _vectorstore

def get_nhs_manual_retriever():
    """Return the underlying retriever for the NHS manual."""
    global _retriever
    if _retriever is not None:
        return _retriever
    vectorstore = _build_vectorstore()
    _retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
    return _retriever

def get_nhs_manual_retriever_tool():
    """Return a LangChain tool that uses the NHS manual retriever."""
    global _retriever_tool
    if _retriever_tool is not None:
        return _retriever_tool

    retriever = get_nhs_manual_retriever()
    _retriever_tool = create_retriever_tool(
        retriever,
        name="nhs_talking_therapies_manual",
        description=(
            "Search the NHS Talking Therapies for anxiety and depression manual "
            "(version 7.1, updated) for service model, stepped care, assessment, "
            "risk, and intervention procedures."
        ),
    )
    return _retriever_tool
