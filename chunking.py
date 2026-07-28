from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
EXTENSION_TO_LANG = {
    '.py': Language.PYTHON,
    '.js': Language.JS,
    '.ts': Language.TS,
    '.tsx': Language.TS,
    '.jsx': Language.JS,
    '.java': Language.JAVA,
    '.go': Language.GO,
    '.rs': Language.RUST,
    '.rb': Language.RUBY,
    '.cpp': Language.CPP,
    '.c': Language.CPP,
    '.md': Language.MARKDOWN,
}

def get_splitter(extension):
    lang = EXTENSION_TO_LANG.get(extension)
    if lang:
        return RecursiveCharacterTextSplitter.from_language(
            language=lang, chunk_size=1000, chunk_overlap=150
        )
    return RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

def chunk_docs(docs):
    all_chunks = []
    for doc in docs:
        splitter = get_splitter(doc["metadata"]["file_ext"])
        # Split text manually to retain/calculate line numbers if possible
        # For simplicity in this RAG, we use the standard splitter
        chunks = splitter.split_text(doc["content"])
        
        for i, chunk in enumerate(chunks):
            # We add a chunk index to help with source identification
            metadata = doc["metadata"].copy()
            metadata["chunk_id"] = i
            all_chunks.append({"content": chunk, "metadata": metadata})
            
    return all_chunks