import streamlit as st
from groq import Groq
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def get_qa_chain(vectorstore, k=2, temperature=0.1):
    api_key = st.secrets.get("GROQ_API_KEY")
    if not api_key:
        st.error("API Key missing.")
        return None

    # THE ULTIMATE STABLE LIST
    # We use 'llama3-8b-8192' first because it is the most common ID in Groq history.
    models_to_try = [
        "llama3-8b-8192",
        "llama-3.2-3b-preview",
        "llama-3.2-1b-preview"
    ]
    
    llm = None
    working_model = ""
    
    # Try the stable list
    for model_id in models_to_try:
        try:
            temp_llm = ChatGroq(api_key=api_key, model_name=model_id, temperature=temperature, max_tokens=400)
            temp_llm.invoke("Hi") # Test
            llm = temp_llm
            working_model = model_id
            break
        except Exception:
            continue

    # DEBUG: If the stable list fails, we find out what you DO have
    if not llm:
        try:
            client = Groq(api_key=api_key)
            all_available = [m.id for m in client.models.list().data]
            st.warning(f"Searching for models... Found: {all_available}")
            
            # Filter for anything that isn't safety or whisper
            chat_models = [m for m in all_available if "guard" not in m and "whisper" not in m and "canopy" not in m]
            if chat_models:
                working_model = chat_models[0]
                llm = ChatGroq(api_key=api_key, model_name=working_model, temperature=temperature, max_tokens=400)
        except Exception as e:
            st.error(f"Critical Error: {e}")
            return None

    if not llm:
        st.error("No compatible chat models found in your Groq account.")
        return None

    st.toast(f"Connected to: {working_model}")

    # --- RAG Logic ---
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    def format_docs(docs):
        # Extremely small chunks to prevent Error 400 (Context too long)
        return "\n\n".join(f"File: {d.metadata.get('file_path')}\n{d.page_content[:500]}" for d in docs)

    template = """You are CodeSage. Answer based on the code provided. 
    Be short. Cite file path.
    
    Context: {context}
    Question: {question}
    Answer:"""
    
    prompt = ChatPromptTemplate.from_template(template)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain