#!/bin/bash
# ============================================
# Ingestion Pipeline Entrypoint
# Runs the full pipeline in sequence:
# 1. Upload raw data to MinIO
# 2. Build the knowledge graph (Neo4j)
# 3. Index chunks into Milvus
# ============================================
set -e

echo "=== Ingestion Pipeline Started ==="

# Wait for dependencies
echo "Waiting for MinIO..."
until curl -s http://minio:9000/minio/health/live > /dev/null 2>&1; do
  sleep 2
done
echo "MinIO is ready."

echo "Waiting for Neo4j..."
until python -c "from neo4j import GraphDatabase; GraphDatabase.driver('bolt://neo4j:7687', auth=('neo4j', '${NEO4J_PASSWORD:-password123}')).verify_connectivity()" 2>/dev/null; do
  sleep 2
done
echo "Neo4j is ready."

# Step 1: Upload data to MinIO
echo "--- Step 1/3: Uploading raw data to MinIO ---"
python upload_to_minio.py

# Step 2: Build knowledge graph
echo "--- Step 2/3: Building knowledge graph in Neo4j ---"
python main.py

# Step 3: Index into Milvus
echo "--- Step 3/3: Indexing chunks into Milvus ---"
python index_to_milvus.py

echo "=== Ingestion Pipeline Complete ==="

# Keep container alive for debugging (optional)
tail -f /dev/null
