import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def get_qa_chain(vectorstore, k=6, temperature=0.1):
    # Fetch the key
    api_key = st.secrets.get("GROQ_API_KEY")
    
    if not api_key:
        st.error("API Key not found in Streamlit Secrets!")
        return None

    try:
        # Using the absolute most stable model ID currently available
        llm = ChatGroq(
            groq_api_key=api_key, 
            model_name="llama-3.1-8b-instant",
            temperature=temperature
        )
        # Verify it works
        llm.invoke("Hi")
    except Exception as e:
        st.error(f"GROQ REJECTED YOUR KEY: {str(e)}")
        return None

    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    def format_docs(docs):
        return "\n\n".join(f"File: {d.metadata.get('file_path')}\n{d.page_content}" for d in docs)

    template = """You are CodeSage. Answer based on context.
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