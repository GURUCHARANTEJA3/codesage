import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def get_qa_chain(vectorstore, k=4, temperature=0.1):
    # 1. Fetch API Key from Streamlit Secrets
    api_key = st.secrets.get("GROQ_API_KEY")
    if not api_key:
        st.error("API Key missing.")
        return None

    # 2. These are the specific META models on Groq
    # We use Llama 3.3 and 3.1 because they have larger context windows
    models_to_try = [
        "llama-3.3-70b-versatile", # Meta's newest flagship
        "llama-3.1-8b-instant",    # Meta's most stable
        "llama3-70b-8192"          # Meta's legacy reliable
    ]
    
    llm = None
    for model_id in models_to_try:
        try:
            temp_llm = ChatGroq(
                groq_api_key=api_key, 
                model_name=model_id,
                temperature=temperature
            )
            # Test connection
            temp_llm.invoke("Hi")
            llm = temp_llm
            break
        except Exception:
            continue

    if not llm:
        st.error("No working Meta Llama models found.")
        return None

    # 3. Setup Retriever with k=4 to stay under the token limit
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    # 4. Format docs and CROP them (Prevents Error 400)
    def format_docs(docs):
        formatted = []
        for d in docs:
            # We take only the first 1000 chars to keep the prompt size safe
            content = d.page_content[:1000]
            file_name = d.metadata.get('file_path', 'Unknown')
            formatted.append(f"--- FILE: {file_name} ---\n{content}")
        return "\n\n".join(formatted)

    # 5. Compact Prompt (Reduces token usage)
    template = """You are CodeSage, an expert AI for Meta Llama 3. 
    Use the context below to answer accurately. 
    Cite file paths. Be technical and concise.

    Context:
    {context}

    Question: {question}
    Answer:"""
    
    prompt = ChatPromptTemplate.from_template(template)

    # 6. Build the Chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain