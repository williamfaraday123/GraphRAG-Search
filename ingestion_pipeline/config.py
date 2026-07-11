import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from current dir first, then from project root
load_dotenv()
load_dotenv(Path(__file__).resolve().parent.parent / '.env')

class Config:
    # MinIO
    MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://localhost:9000")
    MINIO_ROOT_USER = os.getenv("MINIO_ROOT_USER", "minioadmin")
    MINIO_ROOT_PASSWORD = os.getenv("MINIO_ROOT_PASSWORD", "minioadmin")
    MINIO_BUCKET = os.getenv("MINIO_BUCKET", "rag-datasets")

    # Neo4j
    NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password123")
    
    # LLM for Edge Detection
    LLM_API_KEY = os.getenv("LLM_API_KEY", "ollama")
    EDGE_MODEL = os.getenv("EDGE_MODEL", "qwen2.5:1.5b")
    EDGE_MODEL_BASE_URL = os.getenv("EDGE_MODEL_BASE_URL", "http://localhost:11434/v1")
    
    # Chunking
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    
    # Input Data Source (Local folder for demo, S3 for prod)
    DATA_SOURCE_PATH = os.getenv("DATA_SOURCE_PATH", "./raw_data")