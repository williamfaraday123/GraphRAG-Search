# ============================================
# Ingestion Pipeline — Production Dockerfile
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

# Copy app code
COPY . .

# Entrypoint: runs upload, main, then index in sequence
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

ENTRYPOINT ["/docker-entrypoint.sh"]