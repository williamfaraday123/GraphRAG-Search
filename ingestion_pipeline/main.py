# Run the whole pipeline

import os
import logging
from config import Config
from data_loader import MinIODataLoader
from chunker import DocumentChunker
from edge_generator import EdgeGenerator
from graph_builder import GraphBuilder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_ingestion_pipeline():
    logger.info("=== Starting Knowledge Graph Construction ===")
    
    # 1. Initialize Components
    chunker = DocumentChunker(Config.CHUNK_SIZE, Config.CHUNK_OVERLAP)
    edge_gen = EdgeGenerator()
    db = GraphBuilder()
    data_loader = MinIODataLoader()
    
    try:
        # 2. Load Raw Data from MinIO
        raw_docs = data_loader.load_documents()
        logger.info(f"Loaded {len(raw_docs)} raw documents.")
        
        all_nodes = []
        all_edges = []
        
        # 3. Process Each Document
        for doc in raw_docs:
            logger.info(f"Processing: {doc['source']}")
            
            # Step A: Chunking
            nodes = chunker.process_document(doc['content'], doc['source'])
            all_nodes.extend(nodes)
            
            # Step B: Edge Generation (LLM)
            edges = edge_gen.generate_edges_for_document(nodes)
            all_edges.extend(edges)
            
        # 4. Persist to Neo4j
        logger.info(f"Batch writing {len(all_nodes)} nodes and {len(all_edges)} edges to Neo4j...")
        db.ingest_nodes(all_nodes)
        db.ingest_edges(all_edges)
        
        logger.info("=== Knowledge Graph Construction Complete ===")
        logger.info("You can now run the Query Processor (FastAPI) to search this graph.")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise
    finally:
        db.close()

if __name__ == "__main__":
    run_ingestion_pipeline()