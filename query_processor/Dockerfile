# ============================================
# Query Processor — Production Dockerfile
# ============================================

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Virtual env
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy and install deps — fully offline (Docker has no internet here)
COPY requirements.txt .
# Pre-download on host:
#   pip download -r requirements.txt -d wheel_cache --platform manylinux2014_x86_64 --only-binary=:all: --python-version 312
#   pip download torch -d wheel_cache --platform manylinux2014_x86_64 --only-binary=:all: --python-version 312 --index-url https://download.pytorch.org/whl/cpu
COPY wheel_cache/ /wheel_cache/
RUN pip install --no-cache-dir --no-index --find-links=/wheel_cache -r requirements.txt

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