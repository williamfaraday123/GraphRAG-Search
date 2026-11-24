from typing import List, Dict, Any
from services.milvus_client import MilvusService
from services.neo4j_client import Neo4jService
from services.object_storage_client import ObjectStorageRetriever
from embedding import EmbeddingService
from corpus import collection_for_corpus, file_filter_expr, normalize_corpus_id, WIKI_CORPUS_ID
from graph_retrieval import GraphRetriever
from chunk_context import embed_text
from file_similarity import FileSimilarity
from local_ppr import LocalPPR, chunk_node, entity_node
from coverage import CoverageCheck
import logging
import os

import numpy as np

logger = logging.getLogger(__name__)

# Local PPR start weights: entities the question names lead; vector hits (scaled by their
# similarity) add a smaller share, and are the only start nodes before any facts exist
NAMED_SEED_WEIGHT = 1.0
CHUNK_SEED_WEIGHT = 0.2
PPR_NEW_CHUNKS = 2       # chunks PPR may add that neither vector search nor fact paths found

class HybridRetriever:
    def __init__(self, milvus_host: str, milvus_port: str, 
                 neo4j_uri: str, neo4j_user: str, neo4j_password: str,
                 embedding_model: str = "all-MiniLM-L6-v2"):
        
        # Initialize Clients
        self.milvus_service = MilvusService(host=milvus_host, port=milvus_port)
        self.neo4j_service = Neo4jService(uri=neo4j_uri, user=neo4j_user, password=neo4j_password)
        self.embedding_service = EmbeddingService(model_name=embedding_model)
        self.object_storage = ObjectStorageRetriever(
            vector_store=None,
            bucket_name=os.getenv("MINIO_BUCKET", "rag-datasets"),
            aws_access_key_id=os.getenv("MINIO_ROOT_USER", "minioadmin"),
            aws_secret_access_key=os.getenv("MINIO_ROOT_PASSWORD", "minioadmin"),
            endpoint_url=os.getenv("MINIO_ENDPOINT", "http://localhost:9000"),
        )
        # Follows extracted facts (knowledge_graph.py) to chunks vector search misses
        self.graph_retriever = GraphRetriever(self.neo4j_service.driver, self.embedding_service)
        # "Which two files are most similar?": compares files' knowledge graphs (file_similarity.py)
        self.file_similarity = FileSimilarity(self.neo4j_service.driver, self.embedding_service)
        # Authority ranking for the wiki and upload workspaces (local push Personalized PageRank)
        self.local_ppr = LocalPPR(self.neo4j_service.driver)
        # "Is the context enough?" without a model call (graph_engine analyst)
        self.coverage = CoverageCheck(self.neo4j_service.driver)

        logger.info("HybridRetriever initialized with Milvus, Neo4j, MinIO, and Local Embeddings")

    def search(
        self,
        query: str,
        top_k: int = 5,
        collection_name: str = None,
        corpus_id: str = WIKI_CORPUS_ID,
        file_ids: List[str] = None,
        carried_seeds: Dict[str, float] = None,
    ) -> Dict[str, Any]:
        """
        Performs the full hybrid retrieval pipeline:
        1. Generate Embedding for the query.
        2. Search Milvus for semantic similarity.
        3. Search Neo4j for keyword/graph relationships.
        4. Enrich with Full Documents from MinIO (Object Storage).
        5. Merge and return results.
        """
        logger.info(f"Starting sub-graph retrieval for query: '{query}' (Top-K: {top_k})")
        corpus_id = normalize_corpus_id(corpus_id)
        collection_name = collection_name or collection_for_corpus(corpus_id)
        # Only upload workspaces have per-file chunk ids; wiki searches ignore file_ids.
        file_expr = file_filter_expr(file_ids) if file_ids and corpus_id != WIKI_CORPUS_ID else None

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
                top_k=top_k,
                expr=file_expr,
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

        # --- STEP 3a: Knowledge graph (wiki and upload workspaces) ---
        # Walks extracted facts from entities in the question; adds the chunks behind the
        # best fact paths, including ones vector search didn't return.
        kg = {"seeds": [], "paths": [], "chunk_scores": {}, "chunks": [], "relations": []}
        kg_search = os.getenv("KG_SEARCH_ENABLED", "true").lower() not in {"0", "false", "no"}
        if kg_search:
            try:
                kg = self.graph_retriever.search(
                    query, query_vector, corpus_id, file_ids=file_ids or None,
                    vector_chunk_ids=candidate_ids, max_new_chunks=top_k, carried=carried_seeds,
                )
            except Exception as e:
                logger.error(f"Knowledge-graph retrieval failed: {e}")
        for item in milvus_results:
            item["kg_score"] = kg["chunk_scores"].get(item.get("id"), 0.0)
        vector_hits = list(milvus_results)
        milvus_results.extend(kg["chunks"])
        candidate_ids += [c["id"] for c in kg["chunks"]]

        # --- STEP 3b: Local Personalized PageRank (wiki and upload workspaces) ---
        # Authority = how strongly each chunk connects to the question's entities and the
        # vector hits across the whole corpus graph; replaces PageRank on the induced subgraph.
        ppr_nodes = None
        try:
            ppr_nodes = self._apply_local_ppr(
                query_vector, corpus_id, file_ids if file_expr else None, vector_hits, kg.get("named_keys", []),
                milvus_results, max_new_chunks=PPR_NEW_CHUNKS, carried=carried_seeds,
            )
            candidate_ids = [c["id"] for c in milvus_results]
        except Exception as e:
            logger.error(f"Local PPR failed, falling back to subgraph PageRank: {e}")

        logger.info(f"Extracted {len(candidate_ids)} unique IDs for sub-graph construction")

        # Instead of keyword search, query Neo4j to find connections based on IDs from those chunks from milvus search
        # (only needed for the subgraph PageRank fallback, when local PPR failed)
        adjacency_matrix = []
        if candidate_ids and ppr_nodes is None:
            try:
                adjacency_matrix = self.neo4j_service.get_subgraph_adjacency(
                    candidate_ids,
                    corpus_id=None if corpus_id == WIKI_CORPUS_ID else corpus_id,
                )
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
            "relations": adjacency_matrix + kg["relations"], # the edges, incl. knowledge-graph paths
            "node_count": len(candidate_ids),
            "edge_count": len(adjacency_matrix),
            "kg_paths": kg["paths"],
            "kg_seeds": kg["seeds"],
            "ppr_nodes": ppr_nodes,
            "named_keys": kg.get("named_keys", []),  # entities the query names (coverage check)
        }

    def connect_question(self, query: str, corpus_id: str, keys: List[str], file_ids: List[str] = None,
                         known_chunk_ids=(), top_k: int = 5) -> Dict[str, Any]:
        """Knowledge-graph paths joining the entities a whole question names (graph_retrieval
        connect()), which separate sub-query searches can't find. New chunks behind them get
        no PPR score (they weren't in any search's PPR run)."""
        empty = {"paths": [], "chunk_scores": {}, "chunks": []}
        if len(keys) < 2 or os.getenv("KG_SEARCH_ENABLED", "true").lower() in {"0", "false", "no"}:
            return empty
        corpus_id = normalize_corpus_id(corpus_id)
        try:
            result = self.graph_retriever.connect(
                self.embedding_service.generate_embedding(query), corpus_id, keys,
                file_ids=file_ids if file_ids and corpus_id != WIKI_CORPUS_ID else None,
                known_chunk_ids=known_chunk_ids, max_new_chunks=top_k,
            )
        except Exception as e:
            logger.error(f"Knowledge-graph connection search failed: {e}")
            return empty
        for c in result["chunks"]:
            c["ppr_score"] = 0.0
        return result

    def named_keys(self, query: str, corpus_id: str, file_ids: List[str] = None) -> List[str]:
        """Entities the query names (no embedding or vector search; one Neo4j lookup)."""
        corpus_id = normalize_corpus_id(corpus_id)
        return self.graph_retriever.named_keys(
            query, corpus_id, file_ids if file_ids and corpus_id != WIKI_CORPUS_ID else None)

    def existing_entities(self, corpus_id: str, keys: List[str], file_ids: List[str] = None) -> List[str]:
        corpus_id = normalize_corpus_id(corpus_id)
        return self.graph_retriever.existing_entities(
            corpus_id, keys, file_ids if file_ids and corpus_id != WIKI_CORPUS_ID else None)

    def _apply_local_ppr(self, query_vector, corpus_id, file_ids, vector_hits, named_keys,
                         chunks, max_new_chunks, carried=None):
        """Sets each chunk's 'ppr_score' and appends up to max_new_chunks chunks the graph
        connects to at least as strongly as to a vector hit. Returns the nodes visited.
        carried: {entity key: weight} from earlier chat turns, below the named entities' weight."""
        seeds = {entity_node(k): NAMED_SEED_WEIGHT for k in named_keys}
        for key, weight in (carried or {}).items():
            seeds.setdefault(entity_node(key), weight)
        for hit in vector_hits:
            if hit.get("id"):
                seeds[chunk_node(hit["id"])] = max(seeds.get(chunk_node(hit["id"]), 0.0),
                                                   CHUNK_SEED_WEIGHT * max(hit.get("score", 0.0), 0.01))
        scores, visited = self.local_ppr.scores(seeds, corpus_id, file_ids or None)

        known = {c.get("id") for c in chunks}
        hit_scores = [scores.get(h["id"], 0.0) for h in vector_hits if h.get("id")]
        floor = min(hit_scores) if hit_scores else 0.0
        new_ids = [cid for cid, s in sorted(scores.items(), key=lambda kv: -kv[1])
                   if cid not in known and s >= floor and s > 0][:max_new_chunks]
        if new_ids:
            rows = [r for r in self.graph_retriever._load_chunks(new_ids) if r.get("content")]
            if rows:
                vectors = self.embedding_service.generate_batch_embeddings(
                    [embed_text(r["content"], r.get("header")) for r in rows])
                q = np.asarray(query_vector, dtype=np.float32)
                q /= np.linalg.norm(q) + 1e-9
                for row, vec in zip(rows, vectors):
                    v = np.asarray(vec, dtype=np.float32)
                    chunks.append({
                        "id": row["id"], "content": row["content"], "source": row["source"],
                        "score": float(v @ q / (np.linalg.norm(v) + 1e-9)),  # cosine, like Milvus
                        "kg_score": 0.0, "found_by_graph": True,
                    })
        for c in chunks:
            c["ppr_score"] = scores.get(c.get("id"), 0.0)
        logger.info("Local PPR: %d seed(s) (%d entities), %d graph node(s) visited, %d chunk(s) added",
                    len(seeds), len(named_keys), visited, len(new_ids))
        return visited