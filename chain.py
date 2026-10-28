import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def get_qa_chain(vectorstore, k=6, temperature=0.1):
    # Try to get the key from multiple sources
    api_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
    
    if not api_key:
        st.error("🚨 DATABASE ERROR: The API Key is not being picked up by Streamlit. Please re-enter it in the Secrets dashboard.")
        return None

    # Use the most basic, high-availability model for testing
    model_id = "llama-3.1-8b-instant"
    
    try:
        # We manually pass the api_key here to ensure it's used
        llm = ChatGroq(
            groq_api_key=api_key, 
            model_name=model_id,
            temperature=temperature,
            max_retries=2
        )
        # Verify connection
        llm.invoke("Hi")
    except Exception as e:
        st.error(f"❌ GROQ CONNECTION FAILED: {str(e)}")
        st.info("Tip: If you see 'Unauthorized', your API key is wrong. If you see 'Rate Limit', wait 60 seconds.")
        return None

    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    def format_docs(docs):
        return "\n\n".join(f"File: {d.metadata.get('file_path')}\n{d.page_content}" for d in docs)

    template = """You are CodeSage, an expert AI assistant. Use the code context to answer.
    Context: {context}
    Question: {question}
    Answer:"""
    
    prompt_template = ChatPromptTemplate.from_template(template)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt_template
        | llm
        | StrOutputParser()
    )
    
    return rag_chain