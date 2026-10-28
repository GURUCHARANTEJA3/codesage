import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def get_qa_chain(vectorstore, k=6, temperature=0.4):
    api_key = st.secrets.get("GROQ_API_KEY")
    if not api_key:
        return None

    # We use the most powerful model available on the Personal Tier
    # This model can handle HUGE, DEEP answers.
    model_id = "llama-3.3-70b-versatile"
    
    try:
        llm = ChatGroq(
            api_key=api_key, 
            model_name=model_id,
            temperature=temperature,
            max_tokens=4000 # SET TO 4000 FOR MASSIVE, DEEP ANSWERS
        )
    except Exception:
        # Fallback to the 8B if the 70B is busy
        llm = ChatGroq(api_key=api_key, model_name="llama-3.1-8b-instant", max_tokens=2000)

    # We take 6 snippets so the AI has 3x more information to read
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    def format_docs(docs):
        # Full snippets for deep analysis
        return "\n\n".join(f"FILE: {d.metadata.get('file_path')}\n{d.page_content}" for d in docs)

    # THE ULTIMATE "DEEP ANALYSIS" PROMPT
    template = """You are CodeSage, a Distinguished Principal Engineer. 
    The user wants an EXHAUSTIVE, DEEP, and COMPLETE analysis of the project.

    Please provide a massive, detailed report covering:
    1. ARCHITECTURAL BLUEPRINT: Explain the design patterns and the high-level system flow.
    2. DEEP LOGIC DIVE: For every file mentioned in the context, explain the exact logic, the variables, and the algorithms used.
    3. DATA JOURNEY: How does data enter, move through, and exit the system?
    4. TECHNICAL EXPERTISE: Highlight the most sophisticated parts of this codebase. Why is this project impressive?
    5. RECOMMENDATIONS: How would you scale this? What are the potential bottlenecks?

    Format this with clear headers, bold text, and extensive code explanations. 
    Do not be brief. Be as detailed as possible.

    CONTEXT:
    {context}

    QUESTION: 
    {question}

    PRINCIPAL ENGINEER'S DEEP ANALYSIS:"""
    
    prompt = ChatPromptTemplate.from_template(template)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain