import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def get_qa_chain(vectorstore, k=6, temperature=0.1):
    api_key = st.secrets.get("GROQ_API_KEY")
    if not api_key:
        st.error("API Key not found.")
        return None

    # These are the actual model IDs for the brands you saw
    # We try the best Meta model, then the Alibaba model, then the stable Llama
    models_to_try = [
        "llama-3.3-70b-versatile",   # Meta Flagship
        "qwen-2.5-coder-32b",        # Alibaba's Coding Model (If available)
        "llama-3.1-8b-instant",      # Meta Stable
        "llama3-8b-8192",             # Meta Legacy
        "meta-llama/llama-prompt-guard-2-22m"
    ]
    
    llm = None
    for model_id in models_to_try:
        try:
            temp_llm = ChatGroq(
                groq_api_key=api_key, 
                model_name=model_id,
                temperature=temperature
            )
            # Test it - this is the "handshake"
            temp_llm.invoke("Hi")
            llm = temp_llm
            break
        except Exception:
            continue

    if not llm:
        st.error("❌ NO CHAT MODELS FOUND. Your account might only have access to 'Safety Models' like Prompt Guard. Please try creating a NEW Groq API Key.")
        return None

    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    def format_docs(docs):
        return "\n\n".join(f"File: {d.metadata.get('file_path')}\n{d.page_content}" for d in docs)

    template = """You are CodeSage, an expert AI assistant.
    Context: {context}
    Question: {question}
    Answer:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    rag_chain = ({"context": retriever | format_docs, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser())
    return rag_chain