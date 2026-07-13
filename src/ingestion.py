import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

def build_knowledge_base(directory_path="../data/policies", chroma_path="../db/chroma_db"):
    """Loads all PDFs in a directory, chunks them, and builds a fresh vector database."""
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # 1. The Safety Wipe: Empty the DB from the INSIDE
    if os.path.exists(chroma_path):
        print("🧹 Connecting to existing database to drop old tables...")
        try:
            old_db = Chroma(persist_directory=chroma_path, embedding_function=embeddings)
            old_db.delete_collection()
            print("🗑️ Old database successfully emptied!")
        except Exception as e:
            print(f"⚠️ Note on cleanup: {e}")

    # 2. Load all PDFs in the directory
    print(f"\n📄 Loading all PDFs from {directory_path}...")
    loader = PyPDFDirectoryLoader(directory_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} total pages across all documents.")
    
    # 3. Chunk the text
    print("\n✂️ Chunking text (Size: 1000, Overlap: 200)...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    valid_chunks = [chunk for chunk in chunks if chunk.page_content.strip()]
    
    # 4. Build the Database
    print(f"\n🏗️ Building Vector Database with {len(valid_chunks)} chunks...")
    db = Chroma.from_documents(
        documents=valid_chunks, 
        embedding=embeddings, 
        persist_directory=chroma_path
    )
    
    print("\n✅ Knowledge Base completely finished!")
    return len(valid_chunks)