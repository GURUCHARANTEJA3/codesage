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

    # --- FILTERED MODEL DISCOVERY ---
    try:
        client = Groq(api_key=api_key)
        # We get the list but FILTER OUT the ones that cause errors (Canopy, Prompt-Guard, etc.)
        all_models = client.models.list().data
        
        # We only want Meta (Llama), Mistral, or Gemma models
        allowed_keywords = ["llama", "mixtral", "gemma", "70b", "8b"]
        blocked_keywords = ["canopy", "guard", "whisper", "vision", "preview"]
        
        chat_model_ids = [
            m.id for m in all_models 
            if any(k in m.id.lower() for k in allowed_keywords)
            and not any(b in m.id.lower() for b in blocked_keywords)
        ]
        
        # Priority order: We want Llama 3.1 8B first for high free-tier limits
        model_id = "llama-3.1-8b-instant" # Safe default
        
        if chat_model_ids:
            # If our preferred model is in the list, use it. Otherwise use the first allowed one.
            if "llama-3.1-8b-instant" in chat_model_ids:
                model_id = "llama-3.1-8b-instant"
            elif "llama3-8b-8192" in chat_model_ids:
                model_id = "llama3-8b-8192"
            else:
                model_id = chat_model_ids[0]
                
        st.toast(f"Connected to: {model_id}")
    except Exception:
        # Fallback if discovery fails
        model_id = "llama3-8b-8192"
    # -------------------------------

    try:
        # max_tokens=500 is a safe 'middle ground' for all free tiers
        llm = ChatGroq(
            api_key=api_key, 
            model_name=model_id,
            temperature=temperature,
            max_tokens=500
        )
    except Exception as e:
        st.error(f"LLM Setup Error: {e}")
        return None

    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    def format_docs(docs):
        # We take only 2 snippets and limit length to avoid the 400 error
        return "\n\n".join(f"File: {d.metadata.get('file_path')}\n{d.page_content[:700]}" for d in docs)

    template = """You are CodeSage. Answer the question using the code context below.
    BE CONCISE. Cite the file path.
    
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