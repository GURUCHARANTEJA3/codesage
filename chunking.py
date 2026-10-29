from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

def get_splitter(extension):
    # Mapping extensions to LangChain languages
    ext_map = {
        '.py': Language.PYTHON,
        '.js': Language.JS,
        '.ts': Language.TS,
        '.tsx': Language.TS,
        '.java': Language.JAVA,
        '.cpp': Language.CPP,
    }
    lang = ext_map.get(extension)
    if lang:
        # We increase chunk_size to 1500 so the AI sees whole functions
        return RecursiveCharacterTextSplitter.from_language(
            language=lang, chunk_size=1500, chunk_overlap=200
        )
    return RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)

def chunk_docs(docs):
    all_chunks = []
    for doc in docs:
        splitter = get_splitter(doc["metadata"]["file_ext"])
        chunks = splitter.split_text(doc["content"])
        for i, chunk in enumerate(chunks):
            metadata = doc["metadata"].copy()
            metadata["chunk_id"] = i
            all_chunks.append({"content": chunk, "metadata": metadata})
    return all_chunks