import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def get_qa_chain(vectorstore, k=3, temperature=0.1):
    api_key = st.secrets.get("GROQ_API_KEY")
    if not api_key:
        st.error("API Key missing from Secrets.")
        return None

    # We use the 8B model. It is 10x faster and has much higher limits than the 70B.
    # NEVER use 'prompt-guard' here; it only gives numbers, not answers.
    model_id = "llama-3.1-8b-instant"
    
    try:
        llm = ChatGroq(
            api_key=api_key, 
            model_name=model_id,
            temperature=temperature
        )
    except Exception as e:
        st.error(f"LLM Config Error: {e}")
        return None

    # k=3 retrieves 3 snippets of code
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    def format_docs(docs):
        # We limit each snippet to 800 characters to prevent Error 400
        return "\n\n".join(f"File: {d.metadata.get('file_path')}\n{d.page_content[:800]}" for d in docs)

    template = """You are CodeSage. Answer the question using the provided code context.
    If the answer isn't in the code, say you don't know. Cite file paths.

    Context:
    {context}

    Question: {question}
    Answer:"""
    
    prompt = ChatPromptTemplate.from_template(template)

    # Simplified LCEL Chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain