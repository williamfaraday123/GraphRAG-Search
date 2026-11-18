# Wiki snapshot, Milvus and MinIO part.
#   target "tool":    wiki_snapshot.py only; exports a snapshot from the build project
#   default (last):   the tool plus the exported vectors and article texts; restores them
#                     on `docker compose up`. Needs ./snapshot from an export.
FROM python:3.12-slim AS tool

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY wiki_snapshot.py .

ENTRYPOINT ["python", "wiki_snapshot.py"]

FROM tool AS restore

COPY snapshot/manifest.json snapshot/milvus.jsonl.gz snapshot/minio.jsonl.gz /wiki-snapshot/

CMD ["restore", "/wiki-snapshot"]
