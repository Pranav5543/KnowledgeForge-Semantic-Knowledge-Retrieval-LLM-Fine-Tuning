import json
import re
from typing import List
from sentence_transformers import SentenceTransformer
from document_pyramid import DocumentPyramid, KnowledgeChunk

class DocumentProcessor:
    def __init__(self, window_size=2000, overlap=200):
        """
        Initializes the processor with a 2-page sliding window strategy.
        Assuming 1 page ~ 1000 characters for simplicity here,
        a 2-page window is 2000 characters.
        """
        self.window_size = window_size
        self.overlap = overlap
        # Load a small, fast local embedding model for the 'Distilled Knowledge' layer
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
    def _sliding_window_chunk(self, text: str) -> List[str]:
        """
        Chunks the input text using a character-based sliding window with overlap.
        """
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = min(start + self.window_size, text_length)
            
            # Try to snap the end to a natural boundary (like a period or newline)
            # if we aren't at the very end of the text
            if end < text_length:
                last_period = text.rfind('.', start, end)
                if last_period != -1 and (end - last_period) < 300:
                    end = last_period + 1
                    
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(chunk_text)
                
            # Move the window forward, minus the overlap
            new_start = end - self.overlap
            # Prevent infinite loops if the window size or adjusted boundary is smaller than the overlap
            if new_start <= start:
                start = end
            else:
                start = new_start

        return chunks

    def _placeholder_summarize(self, text: str) -> str:
        """
        Layer 2: Chunk Summary
        A placeholder summarization function that extracts the first two sentences.
        In production, this would be an LLM call.
        """
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return " ".join(sentences[:2]) if len(sentences) > 0 else text

    def _assign_category(self, text: str) -> str:
        """
        Layer 3: Category/Theme Label
        A rule-based or simulated labeler based on keyword matching.
        """
        text_lower = text.lower()
        if any(word in text_lower for word in ['court', 'legal', 'law', 'contract', 'plaintiff']):
            return "Legal"
        if any(word in text_lower for word in ['math', 'calculate', 'sum', 'add', 'multiply', 'equation']):
            return "Math"
        if any(word in text_lower for word in ['server', 'code', 'database', 'python', 'algorithm']):
            return "Technology"
        return "General"

    def _distill_knowledge(self, text: str):
        """
        Layer 4: Distilled Knowledge
        Extracts keywords (mock logic) and generates vector embeddings.
        """
        # Mock keyword extraction: take longest words as "keywords" for simplicity
        words = re.findall(r'\b\w+\b', text.lower())
        unique_words = list(set([w for w in words if len(w) > 5]))
        keywords = sorted(unique_words, key=len, reverse=True)[:5]
        
        # Actual embedding generation using SentenceTransformers
        embedding = self.encoder.encode(text).tolist()
        
        return keywords, embedding

    def ingest_document(self, document_id: str, text: str) -> DocumentPyramid:
        """
        End-to-End Pipeline: Load -> Sliding Window -> Build Pyramid -> Store
        """
        print(f"Ingesting document: '{document_id}'...")
        pyramid = DocumentPyramid(document_id=document_id)
        
        raw_chunks = self._sliding_window_chunk(text)
        print(f"Generated {len(raw_chunks)} chunks using sliding window ({self.window_size} chars, {self.overlap} overlap).")
        
        for i, chunk_text in enumerate(raw_chunks):
            # Layer 1: Raw Text
            # Layer 2: Summary
            summary = self._placeholder_summarize(chunk_text)
            
            # Layer 3: Category
            category = self._assign_category(chunk_text)
            
            # Layer 4: Distilling Knowledge
            keywords, embedding = self._distill_knowledge(chunk_text)
            
            chunk = KnowledgeChunk(
                chunk_id=i,
                raw_text=chunk_text,
                summary=summary,
                category=category,
                keywords=keywords,
                embedding=embedding
            )
            pyramid.add_chunk(chunk)
            
        print("Ingestion complete.")
        return pyramid

if __name__ == "__main__":
    # Test script for ingestion
    sample_text = (
        "The quick brown fox jumps over the lazy dog. This is a very standard test sentence. " * 50 +
        "Recently, the supreme court ruled that a contract must be signed by both parties to be legally binding. " * 30 +
        "Let's calculate the sum of the variables using a linear equation." * 30
    )
    processor = DocumentProcessor()
    pyramid = processor.ingest_document("test_doc_01", sample_text)
    
    print("\nSample Chunk Pyramid Structure (Chunk 1):")
    print(json.dumps(pyramid.to_dict()["chunks"][1], indent=2))
