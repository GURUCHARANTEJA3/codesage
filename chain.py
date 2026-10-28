import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def get_qa_chain(vectorstore, k=6, temperature=0.1):
    # 1. Securely fetch the API Key
    api_key = st.secrets.get("GROQ_API_KEY")
    
    if not api_key:
        st.error("🚨 API KEY MISSING: Please add GROQ_API_KEY to your Streamlit Secrets.")
        return None

    # 2. List of models to try (Priority order)
    models_to_try = [
        "llama-3.3-70b-versatile", 
        "llama3-8b-8192",
        "llama-3.1-8b-instant"
    ]
    
    llm = None
    # 3. Try each model until one works
    for model_id in models_to_try:
        try:
            temp_llm = ChatGroq(
                groq_api_key=api_key, 
                model_name=model_id,
                temperature=temperature
            )
            # Test connection with a simple string
            temp_llm.invoke("Ping")
            llm = temp_llm
            break 
        except Exception:
            continue

    if not llm:
        st.error("❌ CONNECTION FAILED: Groq is currently unavailable. Check your API key.")
        return None

    # 4. Set up the retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    # 5. Helper function to format the code snippets
    def format_docs(docs):
        return "\n\n".join(f"--- File: {d.metadata.get('file_path')} ---\n{d.page_content}" for d in docs)

    # 6. The RAG Prompt Template
    template = """You are CodeSage, an expert AI code assistant. 
    Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, say you don't know. 
    ALWAYS cite the file path. Use markdown for code blocks.

    Context:
    {context}

    Question: {question}
    Answer:"""
    
    prompt_template = ChatPromptTemplate.from_template(template)

    # 7. The LCEL Chain (Connects everything)
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt_template
        | llm
        | StrOutputParser()
    )
    
    return rag_chain