import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def get_qa_chain(vectorstore, k=2, temperature=0.1): # Only 2 chunks instead of 6
    api_key = st.secrets.get("GROQ_API_KEY")
    if not api_key:
        st.error("API Key missing.")
        return None

    # WE MUST USE THE 8B MODEL. 
    # The 70B model on the free tier is too restrictive for RAG.
    model_id = "llama-3.1-8b-instant"
    
    try:
        llm = ChatGroq(
            api_key=api_key, 
            model_name=model_id,
            temperature=temperature
        )
    except Exception as e:
        st.error(f"LLM Error: {e}")
        return None

    # k=2 is the safest possible setting for a free account
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    def format_docs(docs):
        formatted = []
        for d in docs:
            # ONLY TAKE 500 CHARACTERS PER FILE. 
            # This is a very tight limit to prevent Error 400.
            content = d.page_content[:500] 
            file_path = d.metadata.get('file_path', 'Unknown')
            formatted.append(f"FILE: {file_path}\nCODE: {content}")
        return "\n\n".join(formatted)

    # Ultra-short prompt to save space
    template = """You are CodeSage. Answer based on this code:
    {context}

    Question: {question}
    Answer:"""
    
    prompt = ChatPromptTemplate.from_template(template)

    # Simple LCEL Chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain