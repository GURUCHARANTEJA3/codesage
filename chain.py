import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def get_qa_chain(vectorstore, k=4, temperature=0.2):
    api_key = st.secrets.get("GROQ_API_KEY")
    if not api_key:
        return None

    # THE ONLY MODEL IN YOUR LIST WITH "DAY 1" SMARTNESS
    # This 120B model provides the deep, clear info you are looking for.
    model_id = "openai/gpt-oss-120b"
    
    try:
        # We use max_tokens=800 to stay under your account's 1000-token limit
        # while still allowing a deep, structured answer.
        llm = ChatGroq(
            api_key=api_key, 
            model_name=model_id, 
            temperature=temperature,
            max_tokens=800
        )
    except Exception:
        # Fallback to the 20B version if the big one is busy
        llm = ChatGroq(api_key=api_key, model_name="openai/gpt-oss-20b", max_tokens=600)

    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    # --- THE "SAGE" PROMPT (Restoring Day 1 Quality) ---
    system_prompt = """You are CodeSage, a Senior Software Architect and Mentor. 
    Analyze the project codebase and provide a DEEP, TECHNICAL, and CLEAR explanation.

    Structure your answer as follows:
    1. PURPOSE: What is the core logic and goal of this project?
    2. ARCHITECTURE: How do the components and files interact?
    3. DATA FLOW: How does information move through the system?
    4. CITATIONS: Clearly state which files you are referencing.

    Tone: Professional, helpful, and high-level. Be as detailed as possible within your token limit."""

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])

    def format_docs(docs):
        # We take 800 characters per file to give the 120B model deep info to read
        return "\n\n".join(f"--- FILE: {d.metadata.get('file_path')} ---\n{d.page_content[:800]}" for d in docs)

    # The Modern LCEL Chain (Compatible with Python 3.13)
    rag_chain = (
        RunnablePassthrough.assign(
            context=lambda x: format_docs(retriever.invoke(x["input"]))
        )
        | qa_prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain