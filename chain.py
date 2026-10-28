import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def get_qa_chain(vectorstore, k=6, temperature=0.1):
    groq_api_key = st.secrets.get("GROQ_API_KEY")
    
    if not groq_api_key:
        st.error("Groq API Key not found in Streamlit Secrets.")
        return None

    # THE ONLY MODELS GROQ SUPPORTS FOR FREE RIGHT NOW
    models_to_try = ["llama-3.2-3b-preview", "llama-3.1-8b-instant"]
    
    llm = None
    for model_id in models_to_try:
        try:
            test_llm = ChatGroq(
                api_key=groq_api_key,
                model_name=model_id,
                temperature=temperature
            )
            # Verify if it works
            test_llm.invoke("Hi")
            llm = test_llm
            break
        except Exception:
            continue

    if not llm:
        st.error("No working Groq models found. Check your API key.")
        return None

    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    def format_docs(docs):
        return "\n\n".join(f"File: {d.metadata.get('file_path')}\n{d.page_content}" for d in docs)

    # Simplified Prompt to avoid warnings
    template = """You are CodeSage, an expert AI assistant. Use the code context to answer.
    If you don't know, say you don't know. Cite the file paths.

    Context:
    {context}

    Question: {question}
    Answer:"""
    
    prompt_template = ChatPromptTemplate.from_template(template)

    # The modern LCEL chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt_template
        | llm
        | StrOutputParser()
    )
    
    return rag_chain