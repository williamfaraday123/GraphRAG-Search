"""
Indexing Script — Populate Milvus with chunk embeddings from Neo4j.

Run this AFTER the ingestion pipeline has populated Neo4j.
Creates the 'ai_docs' collection in Milvus (if not exists) and inserts
embeddings for all chunks stored in the knowledge graph.

Usage:
    python index_to_milvus.py
"""

import os
import sys
import logging
from typing import List
from dotenv import load_dotenv

# Load .env from project root
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASSWORD", "password123")
COLLECTION_NAME = "ai_docs"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # 384 dimensions
BATCH_SIZE = 100


def get_neo4j_chunks() -> List[dict]:
    """
    Fetch chunks from Neo4j.
    If INDEX_ONLY_CONNECTED_CHUNKS is True, only fetches chunks that
    have at least one RELATION edge (selective indexing).
    """
    INDEX_ONLY_CONNECTED = os.getenv("INDEX_ONLY_CONNECTED_CHUNKS", "true").lower() == "true"

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
    chunks = []
    with driver.session() as session:
        if INDEX_ONLY_CONNECTED:
            logger.info("Selective indexing: only chunks with edges")
            result = session.run(
                "MATCH (c:Chunk)-[:RELATION]-() "
                "RETURN DISTINCT c.chunk_id AS id, c.content AS content, "
                "c.source AS source_url, c.chunk_index AS chunk_index "
                "ORDER BY c.source, c.chunk_index"
            )
        else:
            result = session.run(
                "MATCH (c:Chunk) RETURN c.chunk_id AS id, c.content AS content, "
                "c.source AS source_url, c.chunk_index AS chunk_index "
                "ORDER BY c.source, c.chunk_index"
            )
        for record in result:
            chunks.append({
                "id": record["id"],
                "content": record["content"],
                "source_url": record.get("source_url", ""),
                "chunk_index": record.get("chunk_index", 0),
            })
    driver.close()
    logger.info(f"Fetched {len(chunks)} chunks from Neo4j")
    return chunks


def create_milvus_collection(dim: int):
    """Create the ai_docs collection if it doesn't exist."""
    if utility.has_collection(COLLECTION_NAME):
        logger.info(f"Collection '{COLLECTION_NAME}' already exists — dropping to reindex")
        utility.drop_collection(COLLECTION_NAME)

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="source_url", dtype=DataType.VARCHAR, max_length=1024),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
    ]
    schema = CollectionSchema(fields, description=f"Chunk embeddings for {EMBEDDING_MODEL}")
    collection = Collection(name=COLLECTION_NAME, schema=schema)

    # Create IVF_FLAT index for fast search
    index_params = {
        "metric_type": "COSINE",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 128},
    }
    collection.create_index(field_name="vector", index_params=index_params)
    collection.load()
    logger.info(f"Collection '{COLLECTION_NAME}' created and indexed (dim={dim})")
    return collection


def main():
    logger.info("=== Indexing chunks from Neo4j → Milvus ===")

    # 1. Connect to Milvus
    connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
    logger.info(f"Connected to Milvus at {MILVUS_HOST}:{MILVUS_PORT}")

    # 2. Load embedding model
    logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    dim = model.get_sentence_embedding_dimension()
    logger.info(f"Model dimension: {dim}")

    # 3. Fetch chunks from Neo4j
    chunks = get_neo4j_chunks()
    if not chunks:
        logger.error("No chunks found in Neo4j. Run the ingestion pipeline first.")
        sys.exit(1)

    # 4. Create Milvus collection
    collection = create_milvus_collection(dim)

    # 5. Generate embeddings and insert in batches
    texts = [c["content"] for c in chunks]
    logger.info(f"Generating embeddings for {len(texts)} chunks...")
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

    total_inserted = 0
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        batch_vectors = embeddings[i:i + BATCH_SIZE]

        entities = [
            [c["id"] for c in batch],              # doc_id
            [c["content"] for c in batch],          # content
            [c["source_url"] for c in batch],       # source_url
            [v.tolist() for v in batch_vectors],    # vector
        ]
        collection.insert(entities)
        total_inserted += len(batch)
        logger.info(f"Inserted batch {i//BATCH_SIZE + 1}: {total_inserted}/{len(chunks)}")

    collection.flush()
    collection.load()
    logger.info(f"=== Indexing complete: {total_inserted} vectors inserted into '{COLLECTION_NAME}' ===")


if __name__ == "__main__":
    main()
