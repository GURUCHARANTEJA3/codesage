import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def get_qa_chain(vectorstore, k=4, temperature=0.1): # Lower temperature = higher accuracy
    api_key = st.secrets.get("GROQ_API_KEY")
    
    # Use the 120B model for maximum intelligence
    model_id = "openai/gpt-oss-120b"
    
    llm = ChatGroq(
        api_key=api_key, 
        model_name=model_id, 
        temperature=temperature,
        max_tokens=800
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    # --- THE "STRICT ACCURACY" PROMPT ---
    system_prompt = """You are CodeSage, a Senior Technical Lead. 
    Your mission is to provide 100% ACCURATE explanations based ONLY on the provided code context.

    STRICT RULES:
    1. SOURCE ONLY: Do not use outside knowledge. If the answer is not in the context, say: "I cannot find the specific logic for this in the current files."
    2. ARCHITECTURAL TRUTH: Explain how these specific files work together. 
    3. CITE EVERYTHING: Every time you explain a function, name the file it comes from.
    4. NO GUESSING: If you are not sure, describe what you DO see and ask for more context.

    CONTEXT FROM CODEBASE:
    {context}"""

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])

    def format_docs(docs):
        # We increase the content limit to 1200 chars for better accuracy
        return "\n\n".join(f"--- SOURCE: {d.metadata.get('file_path')} ---\n{d.page_content[:1200]}" for d in docs)

    # High-accuracy chain
    rag_chain = (
        RunnablePassthrough.assign(
            context=lambda x: format_docs(retriever.invoke(x["input"]))
        )
        | qa_prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain