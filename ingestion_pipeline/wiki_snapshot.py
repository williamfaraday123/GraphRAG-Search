"""
Wiki snapshot: export the built wiki from one install, restore it into another.

The wiki is built once, offline (query_processor/app/build_wiki.py), because extracting its
knowledge graph takes hours. The result ships as a snapshot in two Docker images:

  graphrag-wiki-neo4j     neo4j.dump (neo4j-admin database dump), loaded by neo4j-load.sh
                          into an empty Neo4j volume before Neo4j first starts
  graphrag-wiki-restore   this script plus milvus.jsonl.gz (the ai_docs collection: ids,
                          text and vectors) and minio.jsonl.gz (the article texts)

  python wiki_snapshot.py export /snapshot    run against the build project, before the dump
  python wiki_snapshot.py restore /snapshot   one-shot container on every `docker compose up`

restore is idempotent and never overwrites: MinIO objects that exist are skipped, and an
existing ai_docs collection is left as it is. A collection is written under a temporary
name and renamed to ai_docs only when complete, so an interrupted restore is redone.
"""
import gzip
import hashlib
import json
import logging
import os
import sys
import time

import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError
from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("wiki_snapshot")

COLLECTION = "ai_docs"
STAGING = "ai_docs_restoring"
WIKI_PREFIX = "uploads/wiki/"
BATCH = 500


def _s3():
    return boto3.client(
        "s3",
        endpoint_url=os.getenv("MINIO_ENDPOINT", "http://minio:9000"),
        aws_access_key_id=os.getenv("MINIO_ROOT_USER", "minioadmin"),
        aws_secret_access_key=os.getenv("MINIO_ROOT_PASSWORD", "minioadmin"),
        config=BotoConfig(signature_version="s3v4"),
        region_name="us-east-1",
    )


def _bucket():
    return os.getenv("MINIO_BUCKET", "rag-datasets")


def _connect_milvus(attempts=60):
    host, port = os.getenv("MILVUS_HOST", "milvus-standalone"), os.getenv("MILVUS_PORT", "19530")
    for attempt in range(attempts):
        try:
            connections.connect(alias="default", host=host, port=port)
            utility.list_collections()
            return
        except Exception as exc:
            if attempt == attempts - 1:
                raise
            logger.info("Waiting for Milvus at %s:%s (%s)", host, port, exc)
            time.sleep(5)


def _write_jsonl(path, rows):
    count = 0
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def _read_jsonl(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


# --- Export ---------------------------------------------------------------------

def _build_info(allow_incomplete):
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI", "bolt://neo4j:7687"),
        auth=(os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "password123")),
    )
    with driver.session() as session:
        row = session.run("MATCH (m:WikiBuild {name: 'wiki'}) RETURN properties(m) AS p").single()
        other = session.run(
            "MATCH (c:Chunk) WHERE c.corpus_id IS NULL OR c.corpus_id <> 'wiki' RETURN count(c) AS n"
        ).single()["n"]
    driver.close()
    if row is None:
        sys.exit("No finished wiki build in this database (run build_wiki.py first)")
    info = dict(row["p"])
    info.pop("name", None)
    if other:
        sys.exit(f"This database holds {other} chunk(s) that are not the wiki; export only a wiki-only build")
    if info.get("pending") and not allow_incomplete:
        sys.exit(f"{info['pending']} chunk(s) still await knowledge-graph extraction; finish the "
                 "build (rerun it) or pass --allow-incomplete")
    return info


def export(folder, allow_incomplete=False):
    os.makedirs(folder, exist_ok=True)
    build = _build_info(allow_incomplete)

    _connect_milvus()
    collection = Collection(COLLECTION)
    collection.load()
    iterator = collection.query_iterator(
        batch_size=BATCH, expr="id >= 0", output_fields=["doc_id", "content", "source_url", "vector"]
    )

    def milvus_rows():
        while True:
            batch = iterator.next()
            if not batch:
                iterator.close()
                return
            for row in batch:
                yield {"doc_id": row["doc_id"], "content": row["content"],
                       "source_url": row["source_url"], "vector": [float(x) for x in row["vector"]]}

    rows = _write_jsonl(os.path.join(folder, "milvus.jsonl.gz"), milvus_rows())
    logger.info("Exported %d vectors from %s", rows, COLLECTION)

    s3, bucket = _s3(), _bucket()

    def minio_objects():
        for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=WIKI_PREFIX):
            for item in page.get("Contents", []):
                obj = s3.get_object(Bucket=bucket, Key=item["Key"])
                yield {"key": item["Key"], "content_type": obj.get("ContentType") or "text/plain; charset=utf-8",
                       "text": obj["Body"].read().decode("utf-8")}

    objects = _write_jsonl(os.path.join(folder, "minio.jsonl.gz"), minio_objects())
    logger.info("Exported %d article texts from MinIO", objects)

    if rows != build.get("chunks"):
        sys.exit(f"Milvus has {rows} vectors but Neo4j has {build.get('chunks')} chunks; the build is inconsistent")
    fingerprint = json.dumps({**build, "vectors": rows, "objects": objects}, sort_keys=True, default=str)
    manifest = {
        "snapshot_id": hashlib.sha256(fingerprint.encode()).hexdigest()[:16],
        "build": build,
        "vectors": rows,
        "objects": objects,
        "dimension": len(next(_read_jsonl(os.path.join(folder, "milvus.jsonl.gz")))["vector"]) if rows else 0,
    }
    with open(os.path.join(folder, "manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, default=str)
    logger.info("Snapshot %s written to %s. Next: stop Neo4j and dump it (see the README)",
                manifest["snapshot_id"], folder)


# --- Restore --------------------------------------------------------------------

def _restore_minio(folder):
    s3, bucket = _s3(), _bucket()
    for attempt in range(60):
        try:
            s3.head_bucket(Bucket=bucket)
            break
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") in {"404", "NoSuchBucket", "NotFound"}:
                s3.create_bucket(Bucket=bucket)
                break
            raise
        except Exception as exc:  # MinIO not up yet
            if attempt == 59:
                raise
            logger.info("Waiting for MinIO (%s)", exc)
            time.sleep(5)
    written = skipped = 0
    for obj in _read_jsonl(os.path.join(folder, "minio.jsonl.gz")):
        try:
            s3.head_object(Bucket=bucket, Key=obj["key"])
            skipped += 1
            continue
        except ClientError:
            pass
        s3.put_object(Bucket=bucket, Key=obj["key"], Body=obj["text"].encode("utf-8"),
                      ContentType=obj["content_type"])
        written += 1
    logger.info("MinIO: %d article text(s) restored, %d already present", written, skipped)


def _restore_milvus(folder, manifest):
    _connect_milvus()
    marker = f"wiki snapshot {manifest['snapshot_id']}"
    if utility.has_collection(COLLECTION):
        description = Collection(COLLECTION).description or ""
        if marker in description:
            logger.info("Milvus: %s already restored from this snapshot", COLLECTION)
        else:
            logger.warning("Milvus: %s already exists and is not from this snapshot; leaving it as it is",
                           COLLECTION)
        return
    if utility.has_collection(STAGING):
        utility.drop_collection(STAGING)  # left by an interrupted restore
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="source_url", dtype=DataType.VARCHAR, max_length=1024),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=manifest["dimension"]),
    ]
    collection = Collection(STAGING, CollectionSchema(fields, description=f"Chunk embeddings ({marker})"))
    batch, total = [], 0

    def flush_batch():
        collection.insert([[r["doc_id"] for r in batch], [r["content"] for r in batch],
                           [r["source_url"] for r in batch], [r["vector"] for r in batch]])

    for row in _read_jsonl(os.path.join(folder, "milvus.jsonl.gz")):
        batch.append(row)
        if len(batch) >= BATCH:
            flush_batch()
            total += len(batch)
            batch = []
    if batch:
        flush_batch()
        total += len(batch)
    collection.flush()
    # FLAT: exact search; the wiki is small enough that an approximate index gains nothing
    collection.create_index("vector", {"metric_type": "COSINE", "index_type": "FLAT", "params": {}})
    if total != manifest["vectors"]:
        raise RuntimeError(f"Restored {total} vectors, expected {manifest['vectors']}")
    utility.rename_collection(STAGING, COLLECTION)
    Collection(COLLECTION).load()
    logger.info("Milvus: restored %d vectors into %s", total, COLLECTION)


def restore(folder):
    with open(os.path.join(folder, "manifest.json"), encoding="utf-8") as handle:
        manifest = json.load(handle)
    logger.info("Restoring wiki snapshot %s (%s article(s), %s chunk(s))", manifest["snapshot_id"],
                manifest["build"].get("articles"), manifest["vectors"])
    _restore_minio(folder)
    _restore_milvus(folder, manifest)
    logger.info("Wiki restore finished")


if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] not in {"export", "restore"}:
        sys.exit("usage: wiki_snapshot.py export|restore <folder> [--allow-incomplete]")
    if sys.argv[1] == "export":
        export(sys.argv[2], allow_incomplete="--allow-incomplete" in sys.argv)
    else:
        restore(sys.argv[2])
