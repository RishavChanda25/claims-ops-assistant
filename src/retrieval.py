import os
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers import ContextualCompressionRetriever
from langchain_core.documents import Document

def retrieve_context(query: str, k: int = 12) -> list:
    """
    Connects to the vector database, performs a hybrid search (Dense + Sparse),
    and reranks the results using a Cross-Encoder to return the top-K documents.
    """
    
    # 1. Initialize Embeddings for Dense Search
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # 2. Connect to the Chroma DB
    current_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.abspath(os.path.join(current_dir, "..", "db", "chroma_db"))
    
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Vector database not found at {db_path}.")

    db = Chroma(persist_directory=db_path, embedding_function=embeddings)
    
    # 3. Prepare Documents for Sparse Search (BM25)
    db_data = db.get()
    docs = [
        Document(page_content=txt, metadata=meta)
        for txt, meta in zip(db_data['documents'], db_data['metadatas'])
    ]
    
    # Scale candidate pool to 3x of final k so the reranker has a diverse pool to filter
    candidate_pool_size = k * 3
    
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = candidate_pool_size

    # 4. Initialize Dense Retriever
    dense_retriever = db.as_retriever(search_kwargs={"k": candidate_pool_size})

    # 5. Ensemble Retriever (Hybrid Search)
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, dense_retriever],
        weights=[0.5, 0.5]
    )

    # 6. Initialize Cross-Encoder Reranker
    # The Cross-Encoder evaluates the candidate pool and truncates it back to the exact user-requested 'k'
    model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
    compressor = CrossEncoderReranker(model=model, top_n=k)

    # 7. Final Pipeline: Hybrid Search -> Rerank
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=ensemble_retriever
    )

    # 8. Retrieve and return the top-K chunks
    return compression_retriever.invoke(query)