import logging
import os
import uuid
from pathlib import PurePath

import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)
from neo4j import GraphDatabase

from corpus import collection_for_corpus, normalize_corpus_id

logger = logging.getLogger(__name__)

MAX_FILE_BYTES = 5 * 1024 * 1024
MAX_CHUNKS_PER_FILE = 500
ALLOWED_EXTENSIONS = {".txt", ".md"}


class CorpusIndexer:
    def __init__(self, embedding_service):
        self.embedding_service = embedding_service
        self.neo4j_driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            auth=(
                os.getenv("NEO4J_USER", "neo4j"),
                os.getenv("NEO4J_PASSWORD", "password123"),
            ),
        )
        self.minio_endpoint = os.getenv("MINIO_ENDPOINT", "http://localhost:9000")
        self.bucket_name = os.getenv("MINIO_BUCKET", "rag-datasets")
        self.s3 = boto3.client(
            "s3",
            endpoint_url=self.minio_endpoint,
            aws_access_key_id=os.getenv("MINIO_ROOT_USER", "minioadmin"),
            aws_secret_access_key=os.getenv("MINIO_ROOT_PASSWORD", "minioadmin"),
            config=BotoConfig(signature_version="s3v4"),
            region_name="us-east-1",
        )
        self.splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n\n", "\n\n", "\n", ". ", "! ", "? ", " ", ""],
            chunk_size=int(os.getenv("CHUNK_SIZE", "1500")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200")),
            length_function=len,
        )

    def close(self):
        self.neo4j_driver.close()

    @staticmethod
    def validate_filename(filename: str) -> str:
        safe_name = PurePath((filename or "").replace("\\", "/").rsplit("/", 1)[-1]).name
        if not safe_name or safe_name in {".", ".."}:
            raise ValueError("A valid filename is required")
        if PurePath(safe_name).suffix.lower() not in ALLOWED_EXTENSIONS:
            raise ValueError("Only .txt and .md files are supported")
        return safe_name

    def index_file(self, corpus_id: str, filename: str, content: bytes):
        self._ensure_bucket()
        corpus_id = normalize_corpus_id(corpus_id)
        if corpus_id == "wiki":
            raise ValueError("Uploads require a user workspace, not the wiki corpus")
        filename = self.validate_filename(filename)
        if not content:
            raise ValueError("The uploaded file is empty")
        if len(content) > MAX_FILE_BYTES:
            raise ValueError("Files must be 5 MB or smaller")

        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ValueError("File must contain valid UTF-8 text") from exc
        if not text.strip():
            raise ValueError("The uploaded file has no searchable text")

        file_id = str(uuid.uuid4())
        object_key = f"uploads/{corpus_id}/{file_id}/{filename}"
        source_uri = f"s3://{self.bucket_name}/{object_key}"
        chunks_text = self.splitter.split_text(text)
        chunks_text = chunks_text[:MAX_CHUNKS_PER_FILE]
        if not chunks_text:
            raise ValueError("No searchable text was found in the uploaded file")

        nodes = [
            {
                "chunk_id": f"{file_id}:{index}",
                "content": chunk,
                "source": source_uri,
                "chunk_index": index,
                "total_chunks": len(chunks_text),
                "corpus_id": corpus_id,
                "file_id": file_id,
            }
            for index, chunk in enumerate(chunks_text)
        ]
        edges = [
            {
                "source": nodes[index]["chunk_id"],
                "target": nodes[index + 1]["chunk_id"],
                "relation_type": "NEXT_CHUNK",
                "weight": 1.0,
                "reason": "Sequential document order",
            }
            for index in range(len(nodes) - 1)
        ]

        try:
            vectors = self.embedding_service.generate_batch_embeddings(chunks_text)
            self._write_graph(nodes, edges)
            self._write_vectors(
                collection_for_corpus(corpus_id), nodes, vectors
            )
            self.s3.put_object(
                Bucket=self.bucket_name,
                Key=object_key,
                Body=content,
                ContentType="text/markdown; charset=utf-8"
                if filename.lower().endswith(".md")
                else "text/plain; charset=utf-8",
            )
        except Exception:
            logger.exception("Indexing upload %s failed", file_id)
            self._rollback_upload(corpus_id, file_id, object_key, nodes)
            raise

        return {"file_id": file_id, "filename": filename, "chunk_count": len(nodes)}

    def list_files(self, corpus_id: str):
        self._ensure_bucket()
        corpus_id = normalize_corpus_id(corpus_id)
        if corpus_id == "wiki":
            raise ValueError("The wiki corpus does not contain user uploads")
        prefix = f"uploads/{corpus_id}/"
        files = []
        for page in self.s3.get_paginator("list_objects_v2").paginate(
            Bucket=self.bucket_name, Prefix=prefix
        ):
            for item in page.get("Contents", []):
                key = item["Key"]
                remainder = key[len(prefix):]
                parts = remainder.split("/", 1)
                if len(parts) == 2:
                    files.append({"file_id": parts[0], "filename": parts[1]})
        return files

    def _ensure_bucket(self):
        try:
            self.s3.head_bucket(Bucket=self.bucket_name)
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code")
            if code not in {"404", "NoSuchBucket", "NotFound"}:
                raise
            self.s3.create_bucket(Bucket=self.bucket_name)

    def _rollback_upload(self, corpus_id, file_id, object_key, nodes):
        try:
            with self.neo4j_driver.session() as session:
                session.run(
                    "MATCH (c:Chunk {corpus_id: $corpus_id, file_id: $file_id}) "
                    "DETACH DELETE c",
                    corpus_id=corpus_id,
                    file_id=file_id,
                ).consume()
        except Exception:
            logger.exception("Could not roll back graph data for upload %s", file_id)

        try:
            collection_name = collection_for_corpus(corpus_id)
            if utility.has_collection(collection_name):
                chunk_ids = [node["chunk_id"] for node in nodes]
                escaped_ids = ", ".join(f'"{chunk_id}"' for chunk_id in chunk_ids)
                Collection(collection_name).delete(f"doc_id in [{escaped_ids}]")
                Collection(collection_name).flush()
        except Exception:
            logger.exception("Could not roll back vector data for upload %s", file_id)

        try:
            self.s3.delete_object(Bucket=self.bucket_name, Key=object_key)
        except Exception:
            logger.exception("Could not roll back source object for upload %s", file_id)

    def _write_graph(self, nodes, edges):
        upsert_nodes = """
        UNWIND $nodes AS node
        MERGE (c:Chunk {chunk_id: node.chunk_id})
        SET c.content = node.content,
            c.source = node.source,
            c.chunk_index = node.chunk_index,
            c.total_chunks = node.total_chunks,
            c.corpus_id = node.corpus_id,
            c.file_id = node.file_id
        """
        with self.neo4j_driver.session() as session:
            session.run(upsert_nodes, nodes=nodes).consume()
            if edges:
                upsert_edges = """
                UNWIND $edges AS edge
                MATCH (a:Chunk {chunk_id: edge.source})
                MATCH (b:Chunk {chunk_id: edge.target})
                MERGE (a)-[r:RELATION {type: edge.relation_type}]->(b)
                SET r.weight = edge.weight, r.reason = edge.reason
                """
                session.run(upsert_edges, edges=edges).consume()

    @staticmethod
    def _write_vectors(collection_name, nodes, vectors):
        connections.connect(
            alias="default",
            host=os.getenv("MILVUS_HOST", "localhost"),
            port=os.getenv("MILVUS_PORT", "19530"),
        )
        if not utility.has_collection(collection_name):
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=256),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="source_url", dtype=DataType.VARCHAR, max_length=1024),
                FieldSchema(
                    name="vector",
                    dtype=DataType.FLOAT_VECTOR,
                    dim=len(vectors[0]),
                ),
            ]
            collection = Collection(
                name=collection_name,
                schema=CollectionSchema(fields, description="Workspace chunk embeddings"),
            )
            collection.create_index(
                field_name="vector",
                index_params={
                    "metric_type": "COSINE",
                    "index_type": "IVF_FLAT",
                    "params": {"nlist": min(128, max(1, len(nodes)))},
                },
            )
        else:
            collection = Collection(collection_name)

        collection.insert(
            [
                [node["chunk_id"] for node in nodes],
                [node["content"] for node in nodes],
                [node["source"] for node in nodes],
                vectors,
            ]
        )
        collection.flush()
        collection.load()
