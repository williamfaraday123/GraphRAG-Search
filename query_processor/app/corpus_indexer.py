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

from chunk_context import chunk_sections, embed_text, make_header
from corpus_map import build_corpus_map
from knowledge_graph import delete_file_entities, load_graph
from corpus import collection_for_corpus, file_filter_expr, normalize_corpus_id, normalize_file_id
from document_text import (
    STORED_CONTENT_TYPE,
    SUPPORTED_EXTENSIONS,
    display_name,
    extract_text,
    is_audio,
    read_text_object,
    stored_text_name,
    transcript_key_for,
)

logger = logging.getLogger(__name__)

MAX_FILE_BYTES = 5 * 1024 * 1024
MAX_CHUNKS_PER_FILE = 500
MAX_MAP_CHUNKS = 600  # keeps the corpus map responsive on CPU


def document_title(filename):
    """'sample.docx' -> 'sample' (stored names add '.txt' to non-text uploads)."""
    name = display_name(filename)
    return name.rsplit(".", 1)[0] if "." in name else name


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
        if PurePath(safe_name).suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError("Only .txt, .md, .pdf, .docx and .mp3 files are supported")
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

        text = extract_text(filename, content)
        if not text.strip():
            raise ValueError("The uploaded file has no searchable text")

        return self.index_text(corpus_id, str(uuid.uuid4()), filename, text)

    def index_text(self, corpus_id: str, file_id: str, filename: str, text: str):
        """Chunks, embeds and stores one document's text: chunks and NEXT_CHUNK edges in
        Neo4j, vectors in the corpus's Milvus collection, the text in MinIO. Used for
        uploads (after text extraction) and by build_wiki.py for each wiki article, so
        both corpora share one layout. Knowledge-graph extraction runs afterwards."""
        self._ensure_bucket()
        corpus_id = normalize_corpus_id(corpus_id)
        file_id = normalize_file_id(file_id)
        # Only the extracted text is stored; the original PDF/DOCX/MP3 bytes are discarded
        object_key = f"uploads/{corpus_id}/{file_id}/{stored_text_name(filename)}"
        source_uri = f"s3://{self.bucket_name}/{object_key}"
        chunks_text = self.splitter.split_text(text)
        chunks_text = chunks_text[:MAX_CHUNKS_PER_FILE]
        if not chunks_text:
            raise ValueError("No searchable text was found in the file")

        title = document_title(filename)
        sections = chunk_sections(text, chunks_text)
        nodes = [
            {
                "chunk_id": f"{file_id}:{index}",
                "content": chunk,
                "source": source_uri,
                "chunk_index": index,
                "total_chunks": len(chunks_text),
                "corpus_id": corpus_id,
                "file_id": file_id,
                # Embedded in front of the chunk (chunk_context.py); entity names are added
                # once the knowledge graph has extracted the file (refresh_vectors)
                "doc_title": title,
                "section": section,
                "context_header": make_header(title, section),
            }
            for index, (chunk, section) in enumerate(zip(chunks_text, sections))
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
            vectors = self.embedding_service.generate_batch_embeddings(
                [embed_text(n["content"], n["context_header"]) for n in nodes])
            self._write_graph(nodes, edges)
            self._write_vectors(
                collection_for_corpus(corpus_id), nodes, vectors
            )
            # Written last: once the object exists the file is listed as ready
            self.s3.put_object(
                Bucket=self.bucket_name,
                Key=object_key,
                Body=text.encode("utf-8"),
                ContentType=STORED_CONTENT_TYPE,
            )
        except Exception:
            logger.exception("Indexing upload %s failed", file_id)
            self._rollback_upload(corpus_id, file_id, object_key)
            raise

        return {"file_id": file_id, "filename": filename, "chunk_count": len(nodes)}

    def refresh_vectors(self, corpus_id: str, file_id: str):
        """Re-embeds a file's chunks with the entities the knowledge graph found in each
        (names first, then the most connected), after extraction. The new vectors are
        inserted before the old ones are deleted, so a search never finds the file missing.
        Returns the number of chunks re-embedded."""
        corpus_id = normalize_corpus_id(corpus_id)
        file_id = normalize_file_id(file_id)
        with self.neo4j_driver.session() as session:
            rows = session.run(
                """
                MATCH (c:Chunk {corpus_id: $corpus_id, file_id: $file_id})
                OPTIONAL MATCH (c)-[:MENTIONS]->(e:Entity)
                OPTIONAL MATCH (e)-[f:FACT {chunk_id: c.chunk_id}]-()
                WITH c, e, count(f) AS facts
                ORDER BY c.chunk_index, CASE e.kind WHEN 'name' THEN 0 ELSE 1 END, facts DESC, e.name
                RETURN c.chunk_id AS chunk_id, c.content AS content, c.source AS source,
                       c.doc_title AS title, c.section AS section,
                       [n IN collect(e.name) WHERE n IS NOT NULL] AS entities
                ORDER BY c.chunk_index
                """,
                corpus_id=corpus_id, file_id=file_id,
            ).data()
        rows = [r for r in rows if r.get("content")]
        if not rows:
            return 0
        for r in rows:
            r["header"] = make_header(r["title"], r["section"] or "", r["entities"])
        vectors = self.embedding_service.generate_batch_embeddings(
            [embed_text(r["content"], r["header"]) for r in rows])

        self._connect_milvus()
        collection_name = collection_for_corpus(corpus_id)
        if not utility.has_collection(collection_name):
            return 0
        collection = Collection(collection_name)
        collection.load()
        old_ids = [r["id"] for r in collection.query(
            expr=file_filter_expr([file_id]), output_fields=["id"])]
        collection.insert([
            [r["chunk_id"] for r in rows],
            [r["content"] for r in rows],
            [r["source"] for r in rows],
            vectors,
        ])
        collection.flush()
        if old_ids:
            collection.delete(f"id in {old_ids}")
            collection.flush()
        with self.neo4j_driver.session() as session:
            session.run(
                """
                UNWIND $rows AS r
                MATCH (c:Chunk {chunk_id: r.chunk_id}) SET c.context_header = r.header
                """,
                rows=[{"chunk_id": r["chunk_id"], "header": r["header"]} for r in rows],
            ).consume()
        return len(rows)

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
                    files.append({"file_id": parts[0], "filename": display_name(parts[1])})
        return files

    def get_file(self, corpus_id: str, file_id: str):
        """Returns {'filename', 'content'} for one uploaded file (used for preview)."""
        corpus_id, file_id, object_key = self._locate_file(corpus_id, file_id)
        return {
            "filename": display_name(object_key.rsplit("/", 1)[-1]),
            "content": read_text_object(self.s3, self.bucket_name, object_key),
        }

    def delete_file(self, corpus_id: str, file_id: str) -> str:
        """Removes a file's chunks from Neo4j and Milvus, then its source object.
        Returns the deleted filename."""
        corpus_id, file_id, object_key = self._locate_file(corpus_id, file_id)
        self._delete_graph_chunks(corpus_id, file_id)
        self._delete_vectors(corpus_id, file_id)
        # Object last: if an index delete fails, the file stays listed and can be retried.
        self.s3.delete_object(Bucket=self.bucket_name, Key=object_key)
        if is_audio(object_key):  # older layout kept the transcript separately
            self.s3.delete_object(Bucket=self.bucket_name, Key=transcript_key_for(object_key))
        return display_name(object_key.rsplit("/", 1)[-1])

    def corpus_map(self, corpus_id: str, file_ids=None, kg_builder=None):
        """Entities, relationships and chunk clusters for a workspace (or the given files).
        Uses the stored LLM knowledge graph when it has facts; otherwise keyword statistics."""
        corpus_id = normalize_corpus_id(corpus_id)
        if corpus_id == "wiki":
            raise ValueError("The corpus map is only available for uploaded files")
        files = self.list_files(corpus_id)
        if file_ids:
            wanted = {normalize_file_id(f) for f in file_ids}
            files = [f for f in files if f["file_id"] in wanted]
        if not files:
            raise ValueError("No uploaded files to map")
        with self.neo4j_driver.session() as session:
            records = session.run(
                """
                MATCH (c:Chunk {corpus_id: $corpus_id})
                WHERE c.file_id IN $file_ids
                RETURN c.chunk_id AS chunk_id, c.file_id AS file_id,
                       c.chunk_index AS chunk_index, c.content AS content
                """,
                corpus_id=corpus_id,
                file_ids=[f["file_id"] for f in files],
            ).data()
        order = {f["file_id"]: i for i, f in enumerate(files)}
        chunks = sorted(
            (r for r in records if r.get("content")),
            key=lambda r: (order[r["file_id"]], r["chunk_index"] or 0),
        )[:MAX_MAP_CHUNKS]
        if not chunks:
            raise ValueError("The selected files have no indexed text")
        mapped_file_ids = [f["file_id"] for f in files]
        kg = load_graph(self.neo4j_driver, corpus_id, mapped_file_ids)
        progress = kg_builder.progress(corpus_id, mapped_file_ids) if kg_builder else None
        return build_corpus_map(
            chunks, self.embedding_service.generate_batch_embeddings, files, kg=kg, progress=progress
        )

    def _locate_file(self, corpus_id: str, file_id: str):
        corpus_id = normalize_corpus_id(corpus_id)
        if corpus_id == "wiki":
            raise ValueError("The wiki corpus does not contain user uploads")
        file_id = normalize_file_id(file_id)
        prefix = f"uploads/{corpus_id}/{file_id}/"
        listing = self.s3.list_objects_v2(Bucket=self.bucket_name, Prefix=prefix, MaxKeys=1)
        contents = listing.get("Contents", [])
        if not contents:
            raise FileNotFoundError("File not found in this workspace")
        return corpus_id, file_id, contents[0]["Key"]

    def _delete_graph_chunks(self, corpus_id, file_id):
        with self.neo4j_driver.session() as session:
            session.run(
                "MATCH (c:Chunk {corpus_id: $corpus_id, file_id: $file_id}) DETACH DELETE c",
                corpus_id=corpus_id,
                file_id=file_id,
            ).consume()
            # Knowledge graph: the file's facts, then entities no remaining chunk mentions
            delete_file_entities(session, corpus_id, file_id)

    def _delete_vectors(self, corpus_id, file_id):
        self._connect_milvus()
        collection_name = collection_for_corpus(corpus_id)
        if utility.has_collection(collection_name):
            collection = Collection(collection_name)
            collection.delete(file_filter_expr([file_id]))
            collection.flush()

    @staticmethod
    def _connect_milvus():
        connections.connect(
            alias="default",
            host=os.getenv("MILVUS_HOST", "localhost"),
            port=os.getenv("MILVUS_PORT", "19530"),
        )

    def _ensure_bucket(self):
        try:
            self.s3.head_bucket(Bucket=self.bucket_name)
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code")
            if code not in {"404", "NoSuchBucket", "NotFound"}:
                raise
            self.s3.create_bucket(Bucket=self.bucket_name)

    def _rollback_upload(self, corpus_id, file_id, object_key):
        try:
            self._delete_graph_chunks(corpus_id, file_id)
        except Exception:
            logger.exception("Could not roll back graph data for upload %s", file_id)

        try:
            self._delete_vectors(corpus_id, file_id)
        except Exception:
            logger.exception("Could not roll back vector data for upload %s", file_id)

        try:  # S3 delete of a missing key is a no-op, so this is safe either way
            self.s3.delete_object(Bucket=self.bucket_name, Key=object_key)
        except Exception:
            logger.exception("Could not roll back object %s for upload %s", object_key, file_id)

    def _write_graph(self, nodes, edges):
        upsert_nodes = """
        UNWIND $nodes AS node
        MERGE (c:Chunk {chunk_id: node.chunk_id})
        SET c.content = node.content,
            c.source = node.source,
            c.chunk_index = node.chunk_index,
            c.total_chunks = node.total_chunks,
            c.corpus_id = node.corpus_id,
            c.file_id = node.file_id,
            c.doc_title = node.doc_title,
            c.section = node.section,
            c.context_header = node.context_header
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

    @classmethod
    def create_collection(cls, collection_name, dim, index_params):
        """Creates an empty chunk collection (doc_id, content, source_url, vector).
        ingestion_pipeline/wiki_snapshot.py restores the wiki into the same schema."""
        cls._connect_milvus()
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="source_url", dtype=DataType.VARCHAR, max_length=1024),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
        ]
        collection = Collection(
            name=collection_name,
            schema=CollectionSchema(fields, description="Chunk embeddings"),
        )
        collection.create_index(field_name="vector", index_params=index_params)
        return collection

    @classmethod
    def _write_vectors(cls, collection_name, nodes, vectors):
        cls._connect_milvus()
        if not utility.has_collection(collection_name):
            collection = cls.create_collection(collection_name, len(vectors[0]), {
                "metric_type": "COSINE",
                "index_type": "IVF_FLAT",
                "params": {"nlist": min(128, max(1, len(nodes)))},
            })
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
