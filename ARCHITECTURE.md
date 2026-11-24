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
                                                        | Wiki snapshot restore   |
                                                        | one-shot jobs           |
                                                        +-------------------------+
```

## Components

| Component | Responsibility | Main implementation |
|---|---|---|
| React web client | Collects a query and renders progress, streamed answer text, and ranked evidence. | `web_search_client/` |
| Orchestration layer | Public HTTP API. Converts the Python gRPC response stream into browser-friendly Server-Sent Events (SSE). It also proxies source-document requests. | `orchestration_layer/` |
| Query processor | Runs retrieval, graph-aware ranking, and answer generation. It exposes the gRPC contract shared with Spring Boot. | `query_processor/app/` and `query_processor/proto/query.proto` |
| Wiki build | Offline: indexes each WikiText article like an upload and extracts its knowledge graph. | `query_processor/app/build_wiki.py`, `docker-compose.wiki-build.yml` |
| Wiki snapshot | Exports the built wiki (Neo4j dump, Milvus vectors, article texts) and restores it into a new install. | `ingestion_pipeline/` |
| MinIO | S3-compatible object storage for document text (uploads and wiki articles), in the `rag-datasets` bucket. | Docker service `minio` |
| Neo4j | Stores `Chunk` nodes, `NEXT_CHUNK` relations, and the knowledge graph (`Entity` nodes, `MENTIONS` and `FACT` relationships). | Docker service `neo4j` |
| Milvus | Stores 384-dimensional chunk embeddings: `ai_docs` for the wiki, `corpus_<uuid>` per upload workspace. | Docker service `milvus-standalone` |
| etcd | Stores Milvus metadata; it is an internal Milvus dependency. | Docker service `etcd` |

## Indexing flow

Uploads and wiki articles are indexed by the same code (`CorpusIndexer.index_text`, then `KnowledgeGraphBuilder.process_file`):

1. Split the text into chunks with ids `<file_id>:<index>`, and link consecutive chunks with `NEXT_CHUNK`.
2. Embed the chunks with `all-MiniLM-L6-v2` and write them to Neo4j and the corpus's Milvus collection; store the text in MinIO. Each vector is computed from the chunk with a short header in front (`Document: sample | Section: …`, from the file name and the nearest heading; `chunk_context.py`). The stored text has no header.
3. Extract facts sentence by sentence with `KG_MODEL`, showing the two sentences before each one so the model can resolve "it" and "they". Up to `KG_PARALLEL` sentences of a chunk are sent at once (Ollama overlaps them up to its `OLLAMA_NUM_PARALLEL`). Keep facts whose subject and object both appear in the sentence; when the sentence has a pronoun, one end may instead be a name from those earlier sentences, as long as the relation is in the sentence. Merge entities that name the same concept, and write `Entity` nodes with `MENTIONS` and `FACT` relationships. When a file is done, re-embed its chunks with the names of the entities each one mentions added to the header (`CorpusIndexer.refresh_vectors`).

Uploads run steps 1–2 during the upload request and step 3 in a background thread. The wiki runs all three offline in `build_wiki.py`, because step 3 takes hours, and is shipped as a snapshot:

```text
raw_data/*.txt -> build_wiki.py (separate Compose project, empty databases)
               -> wiki_snapshot.py export (Milvus vectors, MinIO texts) + neo4j-admin database dump
               -> images graphrag-wiki-neo4j and graphrag-wiki-restore
new install:   wiki-neo4j-load (neo4j-admin database load into the empty volume, before Neo4j starts)
               wiki-restore    (vectors into ai_docs, texts into MinIO)
```

Both restore jobs only fill empty stores, so they never overwrite an existing install's data.

## Search flow

1. The browser sends `POST /api/v1/search` to the orchestration layer with `prompt`, optional `topK`, and a corpus ID (`wiki` or an upload workspace UUID).
2. `SearchController` calls the query processor's `StartProcessing` gRPC method.
3. The query processor's planner either searches the prompt directly or decomposes a multi-part query into a small number of subqueries.
4. For each search query, `HybridRetriever`:
   - embeds the query with `all-MiniLM-L6-v2`;
   - retrieves similar chunks from Milvus;
   - follows knowledge-graph fact paths from the entities the question names, loading only the facts within two hops of them (entities that contain the same proper name, such as "Hollywood films" and "learning from Hollywood", count as one stop);
   - runs local push Personalized PageRank from the vector hits and those entities over the corpus graph in Neo4j; and
   - can retrieve the original document from MinIO for source access.
   After the first round, the workflow also searches for fact chains (up to 3 facts) that join entities from different parts of the question, which separate sub-query searches can't find; the answer model is given them as chains.
5. The LangGraph workflow re-ranks chunks by semantic score plus an authority score. Authority is local push Personalized PageRank (`local_ppr.py`) over the whole corpus graph, for both the wiki and upload workspaces. The graph has chunks, entities, and `NEXT_CHUNK`, `MENTIONS` and `FACT` edges, and the walk starts from the entities the question names and the vector hits. Neighbours are fetched from Neo4j as the push reaches them, so the cost does not grow with corpus size. If local PPR fails, the retriever falls back to PageRank over the Neo4j subgraph induced by the retrieved chunks. Weights: `0.6 × semantic + 0.4 × authority`, or `0.5 × semantic + 0.2 × authority + 0.3 × fact-path relevance` when knowledge-graph paths were found.
6. The analyst can request another retrieval round when it finds a gap; the workflow allows at most two refinement iterations. When the question names entities in the knowledge graph, the analyst uses a graph coverage check (`coverage.py`) instead of a model call: the context is enough when a chunk in it mentions every named entity that the searched files mention; otherwise the next round searches each missing entity's strongest fact. Questions that name no known entity still use the model; file-comparison questions skip the check.
7. The synthesizer LLM produces the final answer from the highest-ranked evidence chunks. It streams the model's output as it is written (LangGraph's `custom` stream mode), so the browser shows the answer while the model is still writing it.
8. The query processor streams statuses, agent steps, answer text, and final evidence over gRPC. Spring Boot relays them as SSE events to the browser. The final `done` event carries the complete answer, which replaces the streamed text; corrections applied after the model finishes (such as restored option labels) appear at that point.

The query processor selects the corpus's Milvus collection (`ai_docs` for the wiki, `corpus_<uuid>` for a workspace) and filters Neo4j entities and facts by `corpus_id`. For uploads, the answer synthesizer is instructed to rely only on retrieved upload context.

## User-uploaded corpus flow

1. React creates a workspace with `POST /api/v1/corpora` and stores its UUID in browser local storage.
2. The browser uploads `.txt` or `.md` files to `POST /api/v1/corpora/{corpusId}/files`. Spring Boot validates the workspace and upload size, then forwards file bytes to the query processor over gRPC.
3. The query processor decodes UTF-8 text, splits it into chunks, creates sequential chunk relationships, embeds the chunks, writes them to Neo4j and a dedicated `corpus_{uuid}` Milvus collection, and stores the original in MinIO.
4. `GET /api/v1/corpora/{corpusId}/files` lists successfully indexed files from MinIO.
5. Search requests carry the selected corpus ID, so a user-file query cannot retrieve Wiki vectors or chunks from a different workspace.

Uploads are synchronous and limited to five `.txt`/`.md` files per request, 5 MB each. Workspace UUIDs are not a substitute for authentication; deployments serving multiple users should associate corpus IDs with authenticated owners and authorize every corpus operation.

## Interfaces

### Browser API

| Endpoint | Purpose |
|---|---|
| `POST /api/v1/search` | Starts a search and returns an SSE stream. Body: `{"prompt":"...","topK":5}`. |
| `POST /api/v1/corpora` | Creates an isolated file-upload workspace. |
| `POST /api/v1/corpora/{corpusId}/files` | Uploads and indexes text/Markdown files. |
| `GET /api/v1/corpora/{corpusId}/files` | Lists indexed files for a workspace. |
| `GET /api/v1/chunk-source?uri=s3://...` | Retrieves the full source document through the backend. |
| `GET /api/v1/health` | Returns the orchestration service health state. |

The search stream emits `status`, `agent_step`, `chunk`, `done`, and `error` events.

### Internal gRPC API

The source of truth for the internal interface is [query.proto](query_processor/proto/query.proto):

- `StartProcessing(SearchRequest)` streams `SearchResponse` messages.
- `GetChunkSource(ChunkSourceRequest)` returns one source document.

## Runtime topology

`docker-compose.yml` defines these services:

| Service | Host port | Notes |
|---|---:|---|
| `milvus-standalone` | 19530 | Vector database |
| `neo4j` | 7474, 7687 | Browser and Bolt driver; pinned to `neo4j:5.26` so the wiki dump loads |
| `minio` | 9000, 9001 | S3 API and console |
| `etcd` | — | Internal Milvus dependency |
| `wiki-neo4j-load` | — | One-shot, profile `wiki`: loads the wiki's Neo4j dump into an empty volume before Neo4j starts |
| `wiki-restore` | — | One-shot, profile `wiki`: restores the wiki's vectors and article texts |
| `query_processor` | 50051 | Internal gRPC service, also exposed for local development |
| `orchestration_layer` | 8080 | Public backend API |

`docker-compose.hub.yml` runs the same services from Docker Hub images, with the wiki jobs always on, plus `ollama` and the one-shot `ollama-pull`. `docker-compose.wiki-build.yml` is a separate project (`graphrag-wiki-build`) with its own databases, used only to build the wiki snapshot.

Within Docker Compose, services communicate over the `ai-net` network using service names rather than `localhost`. Persistent Docker volumes hold Milvus, Neo4j, MinIO, etcd, and the Hugging Face model cache.

## Key configuration boundaries

- Database endpoints and credentials are supplied through `.env` and Docker Compose environment variables.
- The embedding model must remain compatible with the Milvus collection dimension (currently 384 dimensions).
- The planner, analyst and synthesizer use the configured LLM endpoint and models; knowledge-graph extraction uses `KG_MODEL` through Ollama's API. The defaults point at an Ollama endpoint.
- The wiki snapshot is tied to the embedding model, `CHUNK_SIZE`/`CHUNK_OVERLAP`, `KG_MODEL` and the Neo4j version it was built with; rebuild it when any of these change.
- The browser does not access MinIO directly; source content passes through the orchestration layer and query processor.

## Code map

| Path | What to inspect |
|---|---|
| `query_processor/app/build_wiki.py` | Offline wiki build: WikiText articles indexed and extracted like uploads |
| `ingestion_pipeline/wiki_snapshot.py`, `neo4j-load.sh` | Wiki snapshot export and restore |
| `query_processor/app/corpus_indexer.py` | Chunking, embedding and storage for uploads and wiki articles |
| `query_processor/app/chunk_context.py` | Context headers (document, section, entities) embedded in front of each chunk |
| `query_processor/app/knowledge_graph.py` | Sentence-by-sentence fact extraction |
| `query_processor/app/graph_retrieval.py` | Knowledge-graph fact paths from the entities a question names |
| `query_processor/app/main.py` | gRPC server and streamed responses |
| `query_processor/app/retriever.py` | Milvus, Neo4j, and object-storage retrieval |
| `query_processor/app/graph_engine.py` | LangGraph workflow, PageRank, and LLM stages |
| `query_processor/app/local_ppr.py` | Local push Personalized PageRank over a corpus graph (wiki or upload workspace) |
| `query_processor/app/coverage.py` | Graph coverage check that decides whether another retrieval round is needed |
| `query_processor/app/entity_resolution.py` | Merges entities that name the same concept (same normalised wording, or close meaning with the same main noun) into one node, keeping other names as aliases |
| `orchestration_layer/.../SearchController.java` | HTTP/SSE endpoints |
| `orchestration_layer/.../GrpcQueryClient.java` | gRPC-to-SSE bridge |
| `web_search_client/src/App.js` | Browser UI and SSE consumption |
