import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Neo4j
    NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password123")
    
    # LLM for Edge Detection
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    EDGE_MODEL = "gpt-4-turbo" # Can use cheaper model like gpt-3.5-turbo for bulk processing
    
    # Chunking
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    
    # Input Data Source (Local folder for demo, S3 for prod)
    DATA_SOURCE_PATH = os.getenv("DATA_SOURCE_PATH", "./raw_data")