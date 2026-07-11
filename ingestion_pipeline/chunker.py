# Use LangChain to break documents into semantic nodes.

from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict
import hashlib
import logging

logger = logging.getLogger(__name__)

class DocumentChunker:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ". ", " ", ""],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )

    def generate_chunk_id(self, content: str, source: str) -> str:
        """Generates a deterministic ID for a chunk based on content hash."""
        text = f"{source}:{content[:100]}" # Use first 100 chars + source for uniqueness
        return hashlib.sha256(text.encode()).hexdigest()

    def process_document(self, content: str, source_url: str, metadata: dict = None) -> List[Dict]:
        """
        Splits a document into chunks and assigns IDs.
        Returns a list of Node objects (dicts).
        """
        try:
            chunks = self.splitter.split_text(content)
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