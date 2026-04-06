from pymilvus import connections, Collection
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

class MilvusService:
    def __init__(self, host: str, port: str):
        self.host = host
        self.port = port
        self._connect()

    def _connect(self):
        try:
            connections.connect(host=self.host, port=self.port)
            logger.info(f"Connected to Milvus at {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            raise

    def search_vectors(self, collection_name: str, query_vector: List[float], top_k: int) -> List[Dict]:
        """
        Performs vector similarity search.
        Returns list of documents with metadata.
        """
        try:
            collection = Collection(collection_name)
            collection.load()
            
            search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
            
            results = collection.search(
                data=[query_vector],
                anns_field="vector",
                param=search_params,
                limit=top_k,
                output_fields=["content", "source_url", "doc_id"]
            )
            
            hits = []
            for hit in results[0]:
                hits.append({
                    "id": hit.entity.get("doc_id"),
                    "content": hit.entity.get("content"),
                    "source": hit.entity.get("source_url"),
                    "score": hit.score
                })
            return hits
        except Exception as e:
            logger.error(f"Milvus search error: {e}")
            return []