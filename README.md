# Final Year Project: Reimagine-Google-Pagerank-Search-for-Generative-Information-Retrieval

This project aims to develop an AI-powered search engine for Internet by leveraging Large Language Models (LLMs) like Perplexity AI and SearchGPT. The system will search and analyze online resources from databases such as the current Internet, providing intelligent retrieval, evaluation, and recommendation of relevant applications. By integrating AI-driven chatbot capabilities with traditional search engine algorithms, it will lead to next-generation search engine for information retrieval in the Internet.

<img width="4111" height="1270" alt="image" src="https://github.com/user-attachments/assets/791b172b-8a35-4b39-8470-c47d323bee05" />

<img width="3448" height="1817" alt="image" src="https://github.com/user-attachments/assets/10c695cb-3b92-4891-abd1-0cad5df408ba" />

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

## Quick Start (instructions to run locally)

### 1. Start required services

```powershell
# From the project root
docker compose up milvus-standalone neo4j minio -d
```

Verify they're up:
```powershell
docker compose ps
```

[run ingestion_pipeline](ingestion_pipeline/README.md)

### 2. Start the Python Query Processor
[query_processor](query_processor/README.md)

```powershell
cd query_processor
venv\Scripts\activate
cd app
python main.py
```

Leave this terminal running.

### 3. Generate proto stubs & run Spring Boot
[orchestration_layer](orchestration_layer/README.md)

In a **new terminal**:

```powershell
cd orchestration_layer
mvnw compile -DskipTests
mvnw spring-boot:run
```

### 4. (Optional) Start the React UI
[web_search_client](web_search_client/README.md)

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

## How to run in docker

### Build React locally
```powershell
cd web_search_client
npm install
npm run build        # creates web_search_client/build/

cd ..
```
### Build the fat JAR locally (includes proto stubs & React UI)
```powershell
# First, copy the React build into Spring Boot's static resources
Copy-Item -Recurse web_search_client\build\* orchestration_layer\src\main\resources\static\ -Force

# Then package everything into a single JAR
cd orchestration_layer
mvnw clean package -DskipTests
```
If error occurs due to stale .class files in target/
```powershell
# Clean the stale files and rebuild
rm src\main\java\com\com -Recurse -Force
mvnw clean package -DskipTests
```

### 1. Build all images
```powershell
docker compose build
```

### 2. Start everything
```powershell
docker compose up -d
```

## Full Docker deployment (when Docker can pull images)


1. **Build everything locally:**
   ```powershell
   cd web_search_client && npm install && npm run build
   Copy-Item -Recurse web_search_client\build\* orchestration_layer\src\main\resources\static\ -Force
   cd orchestration_layer && mvnw clean package -DskipTests
   cd ..
   ```

2. **Build and run all Docker services:**
   ```powershell
   docker compose build && docker compose up -d
   ```

3. **Monitor and verify:**
   ```powershell
   docker compose logs -f ai-ingestion
   docker compose ps
   ```