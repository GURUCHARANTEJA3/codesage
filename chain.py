import streamlit as st
from groq import Groq # We use the base client to check availability
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def get_qa_chain(vectorstore, k=2, temperature=0.1):
    api_key = st.secrets.get("GROQ_API_KEY")
    if not api_key:
        st.error("API Key missing.")
        return None

    # --- DYNAMIC DISCOVERY WITH TOKEN LIMITS ---
    try:
        client = Groq(api_key=api_key)
        available_models = [m.id for m in client.models.list().data if "prompt-guard" not in m.id.lower()]
        
        # Priority list: We want Llama 3.1 8B first because it has HIGHER limits than Qwen
        model_id = available_models[0]
        for m in available_models:
            if "llama-3.1-8b-instant" in m:
                model_id = m
                break
    except Exception as e:
        return None

    try:
        # CRITICAL FIX: We add max_tokens=300 to stay under the 1000 limit
        llm = ChatGroq(
            api_key=api_key, 
            model_name=model_id,
            temperature=temperature,
            max_tokens=300 
        )
    except Exception as e:
        return None
    # -------------------------------

    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    def format_docs(docs):
        # We take only 2 snippets and only 600 chars to keep the input small
        return "\n\n".join(f"File: {d.metadata.get('file_path')}\n{d.page_content[:600]}" for d in docs)

    # We tell the AI to be EXTREMELY brief to save tokens
    template = """You are CodeSage. Answer the question using the code context below.
    RULES: Be extremely brief. Max 3 sentences. Cite the file path.
    
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