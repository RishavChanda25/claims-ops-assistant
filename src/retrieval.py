import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

def retrieve_context(query: str, k: int = 5, distance_threshold: float = 1.1) -> list:
    """
    Connects to the vector database, embeds the query, and retrieves 
    relevant document chunks that fall within the similarity threshold.
    """
    
    # 1. Initialize the same local embedding model used in ingestion
    # This must match exactly or the vectors won't align.
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # 2. Connect to the Chroma DB
    # Calculate the absolute path dynamically so it works on any computer
    current_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.abspath(os.path.join(current_dir, "..", "db", "chroma_db"))
    
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Vector database not found at {db_path}. Please run ingestion first.")

    db = Chroma(persist_directory=db_path, embedding_function=embeddings)
    
    # 3. Perform similarity search with distance scores
    # Lower distance = Higher similarity
    results_with_scores = db.similarity_search_with_score(query, k=k)
    
    valid_documents = []
    
    for doc, distance in results_with_scores:
        # 4. Apply the Distance Threshold
        # Matches above this distance are considered irrelevant
        if distance <= distance_threshold:
            valid_documents.append(doc)
        else:
            # Optional: Log rejected chunks for debugging
            # print(f"Rejected chunk from {doc.metadata.get('source')} due to distance {distance:.4f}")
            pass
            
    return valid_documents

if __name__ == "__main__":
    # Self-test block
    test_query = "Does my auto policy cover flood damage to my car and my laptop?"
    print(f"Testing Retrieval for: '{test_query}'...")
    
    try:
        docs = retrieve_context(test_query)
        print(f"Successfully retrieved {len(docs)} relevant chunks.\n")
        
        for i, doc in enumerate(docs):
            source = doc.metadata.get('source', 'Unknown').split('/')[-1].split('\\')[-1]
            print(f"--- Result {i+1} | Source: {source} ---")
            print(f"{doc.page_content[:200]}...\n")
            
    except Exception as e:
        print(f"Error during retrieval: {e}")