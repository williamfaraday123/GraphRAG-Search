# ============================================
# Query Processor — Production Dockerfile
# ============================================

FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system deps for sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends gcc build-essential && \
    rm -rf /var/lib/apt/lists/*

# Virtual env
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy and install deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all source code (including proto)
COPY . .

# Generate gRPC stubs from proto during build
RUN python -m grpc_tools.protoc -I proto --python_out=app --grpc_python_out=app proto/query.proto

# Expose gRPC port
EXPOSE 50051

# Default env overrides for Docker networking
ENV MILVUS_HOST=milvus-standalone
ENV NEO4J_URI=bolt://neo4j:7687
ENV MINIO_ENDPOINT=http://minio:9000

WORKDIR /app/app
CMD ["python", "main.py"]