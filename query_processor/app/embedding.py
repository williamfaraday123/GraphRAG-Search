from typing import List
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)

class EmbeddingService:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initializes the embedding model.
        'all-MiniLM-L6-v2' is a popular, fast, local model (384 dimensions).
        Ensure your Milvus collection uses the same dimension (384).
        If using OpenAI (1536 dims), change model and logic accordingly.
        """
        logger.info(f"Loading embedding model: {model_name}")
        try:
            self.model = SentenceTransformer(model_name)
            self.dimension = self.model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded. Dimension: {self.dimension}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise e

    def generate_embedding(self, text: str) -> List[float]:
        """
        Converts a single text string into a vector list.
        """
        if not text:
            return [0.0] * self.dimension
        
        try:
            # encode returns a numpy array, convert to list for JSON/gRPC compatibility
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            # Return zero vector as fallback to prevent crash, though search will be poor
            return [0.0] * self.dimension

    def generate_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Efficiently embeds multiple texts at once.
        """
        if not texts:
            return []
        
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()