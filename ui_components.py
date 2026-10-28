import streamlit as st

def render_sidebar():
    with st.sidebar:
        st.title("🤖 CodeSage Settings")
        repo_url = st.text_input("GitHub Repo URL", placeholder="https://github.com/user/repo")
        branch = st.text_input("Branch (optional)", placeholder="main")
        
        st.divider()
        
        with st.expander("Model Configuration"):
            k = st.slider("Retrieval K (chunks)", 1, 5, 2)
            temp = st.slider("Temperature", 0.0, 1.0, 0.1)
            
        st.divider()
        
        index_btn = st.button("Index Repository", use_container_width=True, type="primary")
        
        return repo_url, branch, index_btn, k, temp

def render_source_explorer(source_documents):
    with st.expander("📍 Referenced Code Sources"):
        for i, doc in enumerate(source_documents):
            rel_path = doc.metadata.get('file_path', 'Unknown')
            st.markdown(f"**Source {i+1}: `{rel_path}`**")
            st.code(doc.page_content, language=doc.metadata.get('file_ext', '').replace('.', ''))