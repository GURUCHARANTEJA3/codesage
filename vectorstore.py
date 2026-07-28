import os
import hashlib
import streamlit as st
from langchain_chroma import Chroma  # Using the new library path
from langchain_huggingface import HuggingFaceEmbeddings # Using the new library path

@st.cache_resource
def get_embedding_model():
    # Note: I updated this import path to the newer version as well
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def get_vectorstore(chunks, repo_url, branch="main"):
    # Create a unique ID for the repo to persist cache
    repo_hash = hashlib.md5(f"{repo_url}_{branch}".encode()).hexdigest()
    persist_directory = f"./chroma_db/{repo_hash}"
    
    embeddings = get_embedding_model()
    
    if os.path.exists(persist_directory) and not chunks:
        # Load existing
        return Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    
    # Create new
    texts = [c["content"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]
    
    vectorstore = Chroma.from_texts(
        texts=texts,
        metadatas=metadatas,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    
    # REMOVED: vectorstore.persist() - This is now automatic!
    
    return vectorstore