# Run the whole pipeline

import os
import glob
from config import Config
from data_loader import load_documents # Assume simple file reader
from chunker import DocumentChunker
from edge_generator import EdgeGenerator
from graph_builder import GraphBuilder
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_documents_from_folder(path: str) -> List[Dict]:
    """Simple helper to read text files."""
    docs = []
    for filepath in glob.glob(f"{path}/*.txt"): # Support .txt, .md, etc.
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            docs.append({"content": content, "source": os.path.basename(filepath)})
    return docs

def run_ingestion_pipeline():
    logger.info("=== Starting Knowledge Graph Construction ===")
    
    # 1. Initialize Components
    chunker = DocumentChunker(Config.CHUNK_SIZE, Config.CHUNK_OVERLAP)
    edge_gen = EdgeGenerator()
    db = GraphBuilder()
    
    try:
        # 2. Load Raw Data
        raw_docs = load_documents_from_folder(Config.DATA_SOURCE_PATH)
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