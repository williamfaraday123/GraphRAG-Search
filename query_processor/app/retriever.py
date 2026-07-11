from typing import List, Dict, Any
from services.milvus_client import MilvusService
from services.neo4j_client import Neo4jService
from services.object_storage_client import ObjectStorageRetriever
from embedding import EmbeddingService
import logging

logger = logging.getLogger(__name__)

class HybridRetriever:
    def __init__(self, milvus_host: str, milvus_port: str, 
                 neo4j_uri: str, neo4j_user: str, neo4j_password: str,
                 embedding_model: str = "all-MiniLM-L6-v2"):
        
        # Initialize Clients
        self.milvus_service = MilvusService(host=milvus_host, port=milvus_port)
        self.neo4j_service = Neo4jService(uri=neo4j_uri, user=neo4j_user, password=neo4j_password)
        self.embedding_service = EmbeddingService(model_name=embedding_model)
        self.object_storage = ObjectStorageRetriever(vector_store=None, bucket_name="rag-datasets", aws_access_key_id="minioadmin", aws_secret_access_key="minioadmin123", endpoint_url="http://minio:9000")
        
        logger.info("HybridRetriever initialized with Milvus, Neo4j, MinIO, and Local Embeddings")

    def search(self, query: str, top_k: int = 5, collection_name: str = "ai_docs") -> Dict[str, Any]:
        """
        Performs the full hybrid retrieval pipeline:
        1. Generate Embedding for the query.
        2. Search Milvus for semantic similarity.
        3. Search Neo4j for keyword/graph relationships.
        4. Enrich with Full Documents from MinIO (Object Storage).
        5. Merge and return results.
        """
        logger.info(f"Starting sub-graph retrieval for query: '{query}' (Top-K: {top_k})")
        
        # --- STEP 1: Generate Embedding ---
        try:
            query_vector = self.embedding_service.generate_embedding(query)
            logger.debug(f"Generated vector of dimension: {len(query_vector)}")
        except Exception as e:
            logger.error(f"Critical failure in embedding generation: {e}")
            # Fallback: Return empty results or raise error depending on strictness
            return {"chunks": [], "relations": [], "error": str(e)}

        # --- STEP 2: Vector Search (Milvus) ---
        milvus_results = []
        try:
            milvus_results = self.milvus_service.search_vectors(
                collection_name=collection_name,
                query_vector=query_vector,
                top_k=top_k
            )
            logger.info(f"Milvus returned {len(milvus_results)} chunks")
        except Exception as e:
            logger.error(f"Milvus search failed: {e}")
            return {"chunks": [], "relations": [], "error": str(e)}

        if not milvus_results:
            return {"chunks": [], "relations": [], "error": "No relevant chunks found in Milvus"}

        # --- STEP 3: Graph Search (Neo4j) ---
        # In production, use an LLM or NLP library to extract entities
        # --- Extract Node IDs for Sub-graph ---
        # Assumption: Your Milvus results have a field 'id' that matches Neo4j 'chunk_id'
        candidate_ids = [item['id'] for item in milvus_results if 'id' in item]
        
        logger.info(f"Extracted {len(candidate_ids)} unique IDs for sub-graph construction")

        # Instead of keyword search, query Neo4j to find connections based on IDs from those chunks from milvus search
        adjacency_matrix = []
        if candidate_ids:
            try:
                adjacency_matrix = self.neo4j_service.get_subgraph_adjacency(candidate_ids)
                logger.info(f"Neo4j returned adjacency matrix with {len(adjacency_matrix)} edges")
            except Exception as e:
                logger.error(f"Neo4j subgraph extraction failed: {e}")


        # --- STEP 4: Enrich with Object Storage (MinIO) ---
        # This is where we use the ObjectStorageRetriever
        enriched_chunks = []
        for chunk in milvus_results:
            # We assume your Milvus metadata contains the S3/MinIO path under 'source'
            source_uri = chunk.get('metadata', {}).get('source')
            
            full_document_content = None
            if source_uri and source_uri.startswith("s3://"):
                try:
                    # Use the retriever to fetch the full file content from MinIO
                    # Note: We adapt the call to just fetch the file based on the URI
                    full_document_content = self.object_storage.fetch_file_content(source_uri)
                except Exception as e:
                    logger.warning(f"Could not fetch full doc from MinIO for {source_uri}: {e}")

            # Append the chunk with the potentially enriched full document
            chunk['full_document_content'] = full_document_content
            enriched_chunks.append(chunk)

        # --- STEP 5: Format Response ---
        return {
            "query_vector_dim": len(query_vector),
            "chunks": enriched_chunks, # the nodes (with content)
            "relations": adjacency_matrix, # the edges
            "node_count": len(candidate_ids),
            "edge_count": len(adjacency_matrix)
        }