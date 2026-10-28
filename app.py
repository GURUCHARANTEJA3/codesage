import streamlit as st
import shutil
import os
from ingest import clone_repository, get_repo_files
from chunking import chunk_docs
from vectorstore import get_vectorstore
from chain import get_qa_chain
from ui_components import render_sidebar, render_source_explorer

st.set_page_config(page_title="CodeSage", page_icon="🧠", layout="wide")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "repo_indexed" not in st.session_state:
    st.session_state.repo_indexed = False

repo_url, branch, index_btn, k, temp = render_sidebar()

# Main logic for indexing
if index_btn and repo_url:
    with st.status("Indexing repository...", expanded=True) as status:
        try:
            status.write("Cloning repository...")
            repo_path = clone_repository(repo_url, branch)
            
            status.write("Parsing files...")
            docs = get_repo_files(repo_path)
            
            status.write(f"Chunking {len(docs)} files...")
            chunks = chunk_docs(docs)
            
            status.write("Generating embeddings & building index...")
            vs = get_vectorstore(chunks, repo_url, branch)
            st.session_state.vectorstore = vs
            st.session_state.repo_indexed = True
            
            # Clean up temp clone
            shutil.rmtree(repo_path)
            status.update(label="Indexing complete!", state="complete", expanded=False)
            st.rerun()
        except Exception as e:
            st.error(f"Error: {str(e)}")

# UI Main Pane
st.title("🧠 CodeSage")
st.caption("Chat with any GitHub repository using RAG (Llama 3 + ChromaDB)")

if not st.session_state.repo_indexed:
    st.info("👈 Enter a GitHub URL and click **Index Repository** to get started.")
    # Example Quick Start
    st.markdown("""
    ### How it works:
    1. **Clone**: We shallow-clone the repo.
    2. **Embed**: We use local `all-MiniLM-L6-v2` to process code chunks.
    3. **RAG**: We use Groq's Llama-3-70B for fast, intelligent answers.
    """)
else:
    # Display Chat History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message:
                render_source_explorer(message["sources"])

    # Chat Input
    if user_input := st.chat_input("Ask a question about the code..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            try:
                # 1. Get the chain
                chain = get_qa_chain(st.session_state.vectorstore, k, temp)
                
                if chain:
                    with st.spinner("Analyzing codebase..."):
                        # 2. Run the chain (Passing user_input directly)
                        answer = chain.invoke(user_input)
                        
                        # 3. Get sources separately for the UI display
                        sources = st.session_state.vectorstore.similarity_search(user_input, k=k)
                        
                        st.markdown(answer)
                        render_source_explorer(sources)
                        
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": answer, 
                            "sources": sources
                        })
            except Exception as e:
                st.error(f"Application Error: {str(e)}")
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()