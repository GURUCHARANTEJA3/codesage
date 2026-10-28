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

    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="llama-3.1-8b-instant",
        temperature=temperature
    )

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
    # This manually builds the RAG logic without needing 'langchain.chains'
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain