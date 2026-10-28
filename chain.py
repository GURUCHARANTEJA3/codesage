import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def get_qa_chain(vectorstore, k=6, temperature=0.3):
    api_key = st.secrets.get("GROQ_API_KEY")
    if not api_key:
        return None

    # THE "DAY 1" SAGE MODEL
    model_id = "llama-3.3-70b-versatile"
    
    llm = ChatGroq(
        api_key=api_key, 
        model_name=model_id, 
        temperature=temperature
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    # --- STEP 1: CONTEXTUALIZE QUESTION ---
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])
    
    contextualize_chain = contextualize_q_prompt | llm | StrOutputParser()

    # --- STEP 2: THE "SAGE" PROMPT (Day 1 Deep Quality) ---
    system_prompt = """You are CodeSage, a world-class Software Architect. 
    Provide DEEP, CLEAR, and PROFESSIONAL explanations based on the context.

    1. DEEP DIVE: Explain the 'Why' behind the logic.
    2. ARCHITECTURE: Describe how components interact.
    3. CITATIONS: Cite file paths clearly.

    CONTEXT:
    {context}"""

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])

    # --- STEP 3: FORMATTING ---
    def format_docs(docs):
        # We take a slightly smaller slice (1200 chars) to prevent the "Length Error"
        return "\n\n".join(f"--- FILE: {d.metadata.get('file_path')} ---\n{d.page_content[:1200]}" for d in docs)

    # --- STEP 4: THE MASTER CHAIN ---
    def get_context(input_dict):
        standalone_query = contextualize_chain.invoke(input_dict)
        docs = retriever.get_relevant_documents(standalone_query)
        return format_docs(docs)

    rag_chain = (
        RunnablePassthrough.assign(context=get_context)
        | qa_prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain