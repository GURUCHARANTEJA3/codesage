import streamlit as st
from groq import Groq # We use the base client to check availability
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def get_qa_chain(vectorstore, k=3, temperature=0.1):
    api_key = st.secrets.get("GROQ_API_KEY")
    if not api_key:
        st.error("API Key missing.")
        return None

    # --- DYNAMIC MODEL DISCOVERY ---
    try:
        client = Groq(api_key=api_key)
        available_models = client.models.list().data
        
        # We look for a model that is a 'Chat' model (not Prompt Guard or Whisper)
        # We prefer Llama 3.3, then 3.1, then 3.2
        chat_model_ids = [m.id for m in available_models if "prompt-guard" not in m.id.lower() and "whisper" not in m.id.lower()]
        
        if not chat_model_ids:
            st.error("Your Groq account currently has no chat models available (only safety/audio models).")
            return None
            
        # Pick the best available one automatically
        # Priority: 3.3 -> 3.1 -> any other
        model_id = chat_model_ids[0]
        for m in chat_model_ids:
            if "3.3" in m:
                model_id = m
                break
        
        st.toast(f"Connected to Groq using model: {model_id}")
        
    except Exception as e:
        st.error(f"Failed to fetch models from Groq: {e}")
        return None
    # -------------------------------

    try:
        llm = ChatGroq(
            api_key=api_key, 
            model_name=model_id,
            temperature=temperature
        )
    except Exception as e:
        st.error(f"LLM Error: {e}")
        return None

    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    def format_docs(docs):
        # Limit context to prevent 'Length' errors
        return "\n\n".join(f"File: {d.metadata.get('file_path')}\n{d.page_content[:800]}" for d in docs)

    template = """You are CodeSage. Answer the question using the provided code.
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