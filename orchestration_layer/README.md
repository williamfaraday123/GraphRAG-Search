# Orchestration Layer — Spring Boot API Gateway

This is the REST → gRPC bridge layer. It receives search queries from the React UI via HTTP POST, forwards them to the Python Query Processor via gRPC, and streams results back to the UI using Server-Sent Events (SSE).

## Architecture Flow

```
React UI (port 3000)
    │  HTTP POST /api/v1/search {prompt}
    ▼
Spring Boot (port 8080) ← this service
    │  gRPC StartProcessing(prompt)
    ▼
Python Query Processor (port 50051)
    │  → Milvus (Vector DB)
    │  → Neo4j (Knowledge Graph)
    │  → LangGraph Pregel Engine
    ▼
Streams back: STATUS → CHUNK → DONE (SSE)
```

## Prerequisites

- **Java 25+** — verify: `java -version`
- **Maven** — verify: `mvn --version`
- **Docker Desktop** — running with containers: `milvus-standalone`, `etcd`, `neo4j`, `minio`
- **Python Query Processor** — running on `localhost:50051`

## Quick Start

### 1. Start required services

```powershell
# From the project root
docker compose up milvus-standalone neo4j minio -d
```

Verify they're up:
```powershell
docker compose ps
```

### 2. Start the Python Query Processor

```powershell
cd query_processor
venv\Scripts\activate
cd app
python main.py
```

Leave this terminal running.

### 3. Generate proto stubs & run Spring Boot

In a **new terminal**:

```powershell
cd orchestration_layer
mvnw compile -DskipTests
mvnw spring-boot:run
```

### 4. (Optional) Start the React UI

```powershell
cd web_search_client
npm start
```

Then open `http://localhost:3000`.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/search` | Submit a search query. Body: `{"prompt": "...", "topK": 5}`. Returns SSE stream. |
| `GET`  | `/api/v1/health` | Health check. Returns `{"status": "UP"}` |

## SSE Event Types

The `/api/v1/search` endpoint returns a `text/event-stream` with these event names:

| Event | Data Fields | Description |
|-------|-------------|-------------|
| `status` | `{stage, message}` | Pipeline stage: `RETRIEVING` → `ITERATING` → `SYNTHESIZING` |
| `chunk` | `{content, sourceId}` | Streaming word-by-word answer tokens |
| `done` | `{answer, sources[], confidence}` | Final answer with citation sources |
| `error` | `{message}` | Error information |

## Configuration

Set these in `.env` or environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `SERVER_PORT` | `8080` | Spring Boot port |
| `GRPC_CLIENT_HOST` | `localhost` | Query Processor host |
| `GRPC_CLIENT_PORT` | `50051` | Query Processor gRPC port |

## Troubleshooting

- **`Connection refused: localhost:50051`** → The Python Query Processor isn't running. Start it first.
- **`Fail connecting to Milvus`** → Run `docker compose up milvus-standalone -d` and wait 10 seconds.
- **Proto compilation errors** → Ensure `src/main/proto/query.proto` exists (it mirrors `query_processor/proto/query.proto`).
