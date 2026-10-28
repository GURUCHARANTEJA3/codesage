import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def get_qa_chain(vectorstore, k=6, temperature=0.1):
    groq_api_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
    
    if not groq_api_key:
        st.error("Groq API Key not found.")
        return None

    # List of models to try in order of quality
    # Groq changes these frequently, so we try the whole list!
    models_to_try = [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "llama3-8b-8192",
        "gemma2-9b-it"
    ]
    
    llm = None
    
    # Try each model until one works
    for model in models_to_try:
        try:
            llm = ChatGroq(
                groq_api_key=groq_api_key,
                model_name=model,
                temperature=temperature,
                max_retries=1
            )
            # Test if the model works with a tiny dummy call
            llm.invoke("test") 
            break # If it works, stop the loop
        except Exception:
            continue # If it fails, try the next model in the list

    if llm is None:
        st.error("Could not connect to any Groq models. Please check your API key and Tier.")
        return None

    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    # This combines the retrieved documents into one string
    def format_docs(docs):
        return "\n\n".join(f"--- {d.metadata['file_path']} ---\n{d.page_content}" for d in docs)

    # The Prompt
    template = """You are CodeSage, an expert AI code assistant. 
    Use the retrieved context to answer the question. 
    If you don't know the answer, say you don't know. 
    CITE file paths. Use markdown for code blocks.

    Context:
    {context}

    Question: {question}
    Answer:"""
    
    prompt = ChatPromptTemplate.from_template(template)

    # The Chain (LCEL style)
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain