# Use LangChain to break documents into semantic nodes.

from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict
import hashlib
import logging
from config import Config

logger = logging.getLogger(__name__)

class DocumentChunker:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        # Use paragraph-aware separators for better semantic boundaries
        self.splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n\n", "\n\n", "\n", ". ", "! ", "? ", " ", ""],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
        self.max_chunks = Config.MAX_CHUNKS_PER_DOC

    def generate_chunk_id(self, content: str, source: str) -> str:
        """Generates a deterministic ID for a chunk based on content hash."""
        text = f"{source}:{content[:100]}" # Use first 100 chars + source for uniqueness
        return hashlib.sha256(text.encode()).hexdigest()

    def process_document(self, content: str, source_url: str, metadata: dict = None) -> List[Dict]:
        """
        Splits a document into chunks and assigns IDs.
        Respects MAX_CHUNKS_PER_DOC to cap graph size for large documents.
        """
        try:
            chunks = self.splitter.split_text(content)
            
            # Apply cap if set (0 = unlimited)
            if self.max_chunks > 0 and len(chunks) > self.max_chunks:
                logger.info(f"Capping {source_url}: {len(chunks)} → {self.max_chunks} chunks")
                # Sample evenly across the document to preserve topical diversity
                step = len(chunks) / self.max_chunks
                indices = [int(i * step) for i in range(self.max_chunks)]
                chunks = [chunks[i] for i in indices]
            
            logger.info(f"Split {source_url} into {len(chunks)} chunks.")
            
            nodes = []
            for i, chunk_text in enumerate(chunks):
                node = {
                    "chunk_id": self.generate_chunk_id(chunk_text, source_url),
                    "content": chunk_text,
                    "source": source_url,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "metadata": metadata or {}
                }
                nodes.append(node)
            return nodes
        except Exception as e:
            logger.error(f"Error chunking {source_url}: {e}")
            return []