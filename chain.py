import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def get_qa_chain(vectorstore, k=1, temperature=0.1): # Only 1 single chunk
    api_key = st.secrets.get("GROQ_API_KEY")
    if not api_key:
        st.error("API Key missing.")
        return None

    # Use the 8B model - it has the highest limits
    model_id = "llama-3.1-8b-instant"
    
    try:
        llm = ChatGroq(
            api_key=api_key, 
            model_name=model_id,
            temperature=temperature
        )
    except Exception as e:
        return None

    # k=1 is the absolute minimum (only one piece of code)
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    def format_docs(docs):
        # Extreme truncation: Only 400 characters
        if not docs: return ""
        content = docs[0].page_content[:400] 
        file_path = docs[0].metadata.get('file_path', 'code')
        return f"File: {file_path}\nContent: {content}"

    # Minimalist prompt with NO history placeholders
    template = """Answer briefly based ONLY on this:
    {context}
    Question: {question}"""
    
    prompt = ChatPromptTemplate.from_template(template)

    # This chain is now "Stateless" (It doesn't look at old messages)
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain