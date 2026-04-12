from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class KnowledgeChunk:
    """
    Represents a single chunk from the sliding window strategy.
    It contains all 4 layers of the Knowledge Pyramid.
    """
    # Layer 1: Raw Text
    chunk_id: int
    raw_text: str
    
    # Layer 2: Chunk Summary
    summary: str = ""
    
    # Layer 3: Category/Theme Label
    category: str = "General"
    
    # Layer 4: Distilled Knowledge (Keyword list and Vector Embedding)
    keywords: List[str] = field(default_factory=list)
    embedding: Optional[List[float]] = None

@dataclass
class DocumentPyramid:
    """
    Represents the full document ingested into a hierarchical pyramid structure.
    """
    document_id: str
    chunks: List[KnowledgeChunk] = field(default_factory=list)
    
    def add_chunk(self, chunk: KnowledgeChunk):
        self.chunks.append(chunk)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize the pyramid for simple saving or viewing."""
        return {
            "document_id": self.document_id,
            "chunks": [
                {
                    "chunk_id": c.chunk_id,
                    "raw_text": c.raw_text,
                    "summary": c.summary,
                    "category": c.category,
                    "keywords": c.keywords,
                    "has_embedding": c.embedding is not None,
                }
                for c in self.chunks
            ]
        }
