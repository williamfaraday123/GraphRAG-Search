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

# Wait for dependencies.
# NOTE: this image is python:3.12-slim and has no curl, so every probe below
# must use the venv python. A curl-based probe silently loops forever here.
wait_for() {
  name="$1"; probe="$2"; attempts="${3:-60}"
  echo "Waiting for ${name}..."
  i=0
  until python -c "${probe}" 2>/dev/null; do
    i=$((i + 1))
    if [ "$i" -ge "$attempts" ]; then
      echo "ERROR: ${name} not reachable after $((attempts * 2))s — last probe output:" >&2
      python -c "${probe}" >&2 || true
      exit 1
    fi
    sleep 2
  done
  echo "${name} is ready."
}

wait_for "MinIO" "import urllib.request; urllib.request.urlopen('${MINIO_ENDPOINT:-http://minio:9000}/minio/health/live', timeout=3)"
wait_for "Neo4j" "from neo4j import GraphDatabase; GraphDatabase.driver('${NEO4J_URI:-bolt://neo4j:7687}', auth=('${NEO4J_USER:-neo4j}', '${NEO4J_PASSWORD:-password123}')).verify_connectivity()"
wait_for "Milvus" "from pymilvus import connections; connections.connect(alias='probe', host='${MILVUS_HOST:-milvus-standalone}', port='${MILVUS_PORT:-19530}')"

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
