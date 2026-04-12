import numpy as np
from sentence_transformers import SentenceTransformer
from document_pyramid import DocumentPyramid

def cosine_similarity(a: list, b: list) -> float:
    a_np = np.array(a)
    b_np = np.array(b)
    return np.dot(a_np, b_np) / (np.linalg.norm(a_np) * np.linalg.norm(b_np) + 1e-10)

class PyramidRetriever:
    def __init__(self, encoder_model='all-MiniLM-L6-v2'):
        self.encoder = SentenceTransformer(encoder_model)
        
    def retrieve(self, query: str, pyramid: DocumentPyramid, top_k: int = 1):
        """
        Takes a user query and matches it against the embedding layer of the
        knowledge pyramid using cosine similarity.
        """
        query_embedding = self.encoder.encode(query).tolist()
        
        results = []
        for chunk in pyramid.chunks:
            if chunk.embedding is not None:
                sim = cosine_similarity(query_embedding, chunk.embedding)
                results.append((sim, chunk))
                
        # Sort by highest similarity
        results.sort(key=lambda x: x[0], reverse=True)
        
        top_results = results[:top_k]
        
        output = []
        for sim, chunk in top_results:
            response = {
                "score": sim,
                "category": chunk.category,
                "summary": chunk.summary,
                "text": chunk.raw_text,
                "keywords": chunk.keywords
            }
            output.append(response)
            
        return output

if __name__ == "__main__":
    from ingestion import DocumentProcessor
    
    print("Initializing dummy data...")
    sample_text = (
        "The quick brown fox jumps over the lazy dog. This is a very standard test sentence. " * 50 +
        "Recently, the supreme court ruled that a contract must be signed by both parties to be legally binding. " * 30 +
        "Let's calculate the sum of the variables using a linear equation." * 30
    )
    processor = DocumentProcessor()
    pyramid = processor.ingest_document("test_doc_01", sample_text)
    
    retriever = PyramidRetriever()
    
    queries = [
        "What did the court say about contracts?",
        "How do I add these variables?",
        "Tell me about the dog."
    ]
    
    for q in queries:
        print(f"\n--- Query: '{q}' ---")
        best_match = retriever.retrieve(q, pyramid, top_k=1)[0]
        print(f"Similarity Score: {best_match['score']:.4f}")
        print(f"Detected Category: {best_match['category']}")
        print(f"Retrieved Summary: {best_match['summary']}")
