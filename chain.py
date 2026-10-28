import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def get_qa_chain(vectorstore, k=6, temperature=0.1):
    # 1. Fetch API Key
    api_key = st.secrets.get("GROQ_API_KEY")
    if not api_key:
        st.error("API Key missing from Secrets.")
        return None

    # 2. Initialize Model (Using the most stable ID)
    try:
        llm = ChatGroq(
            api_key=api_key,
            model_name="llama-3.3-70b-versatile",
            temperature=temperature
        )
    except Exception as e:
        st.error(f"LLM Initialization Error: {e}")
        return None

    # 3. Define the Retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    # 4. Helper to format code snippets
    def format_docs(docs):
        return "\n\n".join(f"--- File: {d.metadata.get('file_path')} ---\n{d.page_content}" for d in docs)

    # 5. The Prompt (Uses 'context' and 'question')
    template = """You are CodeSage, an expert AI code assistant.
    Use the following pieces of context to answer the question.
    Cite file paths for every claim.

    Context:
    {context}

    Question: {question}
    Answer:"""
    
    prompt = ChatPromptTemplate.from_template(template)

    # 6. The LCEL Chain
    # We map 'user_input' to 'question' and 'retriever' to 'context'
    rag_chain = (
        {
            "context": retriever | format_docs, 
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain