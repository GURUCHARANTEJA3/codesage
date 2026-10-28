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
# --- CHAT INTERFACE ---
st.title("🧠 CodeSage")

if not st.session_state.get("repo_indexed"):
    st.info("👈 Enter a GitHub URL and click **Index Repository** to get started.")
else:
    # Display Chat History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message:
                render_source_explorer(message["sources"])

    # Chat Input
    if user_prompt := st.chat_input("Ask CodeSage anything..."):
        # 1. Display user message
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        # 2. Generate Assistant Response
        with st.chat_message("assistant"):
            try:
                # Use the k and temp from the sidebar
                chain = get_qa_chain(st.session_state.vectorstore, k=k, temperature=temp)
                
                if chain:
                    with st.spinner("CodeSage is thinking..."):
                        # Format history for the chain
                        chat_history = []
                        for m in st.session_state.messages[:-1]:
                            from langchain_core.messages import HumanMessage, AIMessage
                            if m["role"] == "user":
                                chat_history.append(HumanMessage(content=m["content"]))
                            else:
                                chat_history.append(AIMessage(content=m["content"]))

                        # Invoke the chain
                        answer = chain.invoke({
                            "input": user_prompt,
                            "chat_history": chat_history
                        })
                        
                        # Get sources for display
                        sources = st.session_state.vectorstore.similarity_search(user_prompt, k=k)
                        
                        # Render result
                        st.markdown(answer)
                        render_source_explorer(sources)
                        
                        # Save to state
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": answer, 
                            "sources": sources
                        })
            except Exception as e:
                st.error(f"App Error: {str(e)}")
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()