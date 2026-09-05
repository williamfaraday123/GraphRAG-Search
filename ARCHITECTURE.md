# System Architecture

This document describes how the application is assembled and how information moves through it. For installation and local start commands, use [README.md](README.md).

## Overview

The system turns source documents into a searchable knowledge base, then combines semantic search and graph authority to answer a user question.

```text
                         Search request / answer stream
+----------------+ HTTP/SSE +---------------------+ gRPC +-----------------------+
| React client   | <-------> | Orchestration layer | <--> | Query processor       |
| (port 3000)    |           | Spring Boot :8080   |      | Python gRPC :50051    |
+----------------+           +---------------------+      +-----------+-----------+
                                                                  |     |       |
                                                     semantic     |     | graph | source
                                                     search       |     |       | document
                                                                  v     v       v
                                                           +--------+ +-------+ +-------+
                                                           | Milvus | | Neo4j | | MinIO |
                                                           | :19530 | | :7687 | | :9000 |
                                                           +--------+ +-------+ +-------+
                                                                ^        ^        ^
                                                                |        |        |
                                                        +-------+--------+--------+
                                                        | Ingestion pipeline      |
                                                        | Python one-shot job     |
                                                        +-------------------------+
```

## Components

| Component | Responsibility | Main implementation |
|---|---|---|
| React web client | Collects a query and renders progress, streamed answer text, and ranked evidence. | `web_search_client/` |
| Orchestration layer | Public HTTP API. Converts the Python gRPC response stream into browser-friendly Server-Sent Events (SSE). It also proxies source-document requests. | `orchestration_layer/` |
| Query processor | Runs retrieval, graph-aware ranking, and answer generation. It exposes the gRPC contract shared with Spring Boot. | `query_processor/app/` and `query_processor/proto/query.proto` |
| Ingestion pipeline | Imports raw files, creates chunks and their relationships, writes the knowledge graph, and indexes chunks in Milvus. | `ingestion_pipeline/` |
| MinIO | S3-compatible object storage for the original source documents, in the `rag-datasets` bucket. | Docker service `minio` |
| Neo4j | Stores `Chunk` nodes and relationships between chunks. | Docker service `neo4j` |
| Milvus | Stores 384-dimensional embeddings for chunk similarity search in the `ai_docs` collection. | Docker service `milvus-standalone` |
| etcd | Stores Milvus metadata; it is an internal Milvus dependency. | Docker service `etcd` |

## Ingestion and indexing flow

The ingestion pipeline is a batch process. In Docker, its entrypoint runs these steps in order:

1. `upload_to_minio.py` uploads files from `ingestion_pipeline/raw_data/` to MinIO.
2. `main.py` loads the stored documents, splits them into chunks, and generates relations between chunks.
3. `GraphBuilder` writes chunk nodes and relations to Neo4j.
4. `index_to_milvus.py` reads chunks from Neo4j, creates embeddings using `all-MiniLM-L6-v2`, and upserts them into Milvus.

```text
raw_data/ -> MinIO -> DocumentChunker -> EdgeGenerator -> Neo4j
                                                   \-> index_to_milvus.py -> Milvus
```

`DocumentChunker` uses deterministic chunk IDs, so the graph and vector index can refer to the same source chunk. `EdgeGenerator` uses the configured LLM to identify semantic relations; it also adds sequential chunk connections. The LLM and connection details are configured through environment variables.

## Search flow

1. The browser sends `POST /api/v1/search` to the orchestration layer with `prompt` and optional `topK`.
2. `SearchController` calls the query processor's `StartProcessing` gRPC method.
3. The query processor's planner either searches the prompt directly or decomposes a multi-part query into a small number of subqueries.
4. For each search query, `HybridRetriever`:
   - embeds the query with `all-MiniLM-L6-v2`;
   - retrieves similar chunks from Milvus;
   - uses those chunk IDs to fetch their local subgraph from Neo4j; and
   - can retrieve the original document from MinIO for source access.
5. The LangGraph workflow calculates PageRank over the retrieved subgraph and re-ranks chunks with `0.6 × semantic score + 0.4 × PageRank score`.
6. The analyst can request another retrieval round when it finds a gap; the workflow allows at most two refinement iterations.
7. The synthesizer LLM produces the final answer from the highest-ranked evidence chunks.
8. The query processor streams statuses, agent steps, answer words, and final evidence over gRPC. Spring Boot relays them as SSE events to the browser.

## Interfaces

### Browser API

| Endpoint | Purpose |
|---|---|
| `POST /api/v1/search` | Starts a search and returns an SSE stream. Body: `{"prompt":"...","topK":5}`. |
| `GET /api/v1/chunk-source?uri=s3://...` | Retrieves the full source document through the backend. |
| `GET /api/v1/health` | Returns the orchestration service health state. |

The search stream emits `status`, `agent_step`, `chunk`, `done`, and `error` events.

### Internal gRPC API

The source of truth for the internal interface is [query.proto](query_processor/proto/query.proto):

- `StartProcessing(SearchRequest)` streams `SearchResponse` messages.
- `GetChunkSource(ChunkSourceRequest)` returns one source document.

## Runtime topology

`docker-compose.yml` defines seven services:

| Service | Host port | Notes |
|---|---:|---|
| `milvus-standalone` | 19530 | Vector database |
| `neo4j` | 7474, 7687 | Browser and Bolt driver |
| `minio` | 9000, 9001 | S3 API and console |
| `etcd` | — | Internal Milvus dependency |
| `ingestion_pipeline` | — | Batch ingestion/indexing container |
| `query_processor` | 50051 | Internal gRPC service, also exposed for local development |
| `orchestration_layer` | 8080 | Public backend API |

Within Docker Compose, services communicate over the `ai-net` network using service names rather than `localhost`. Persistent Docker volumes hold Milvus, Neo4j, MinIO, etcd, and the Hugging Face model cache.

## Key configuration boundaries

- Database endpoints and credentials are supplied through `.env` and Docker Compose environment variables.
- The embedding model must remain compatible with the Milvus collection dimension (currently 384 dimensions).
- The planner, analyst, edge generator, and synthesizer use the configured LLM endpoint and model. The default query-side configuration points at an OpenAI-compatible Ollama endpoint.
- The browser does not access MinIO directly; source content passes through the orchestration layer and query processor.

## Code map

| Path | What to inspect |
|---|---|
| `ingestion_pipeline/main.py` | Knowledge-graph ingestion orchestration |
| `ingestion_pipeline/index_to_milvus.py` | Neo4j-to-Milvus indexing |
| `query_processor/app/main.py` | gRPC server and streamed responses |
| `query_processor/app/retriever.py` | Milvus, Neo4j, and object-storage retrieval |
| `query_processor/app/graph_engine.py` | LangGraph workflow, PageRank, and LLM stages |
| `orchestration_layer/.../SearchController.java` | HTTP/SSE endpoints |
| `orchestration_layer/.../GrpcQueryClient.java` | gRPC-to-SSE bridge |
| `web_search_client/src/App.js` | Browser UI and SSE consumption |
