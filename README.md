# Final Year Project: Reimagine Google PageRank Search for Generative Information Retrieval

GraphRAG-Search combines graph-aware authority ranking with LLM answer synthesis. Instead of ranking hyperlinked web pages, it ranks chunks of unstructured text over a graph built automatically from the documents, then has an LLM agent write an answer from the best evidence and stream it to the browser.

It has two modes:

- **Wiki dump**: a fixed corpus built offline by the [ingestion pipeline](ingestion_pipeline/README.md). An LLM labels relations between chunks, and authority is local push Personalized PageRank over those relations.
- **My files**: upload workspaces. Users upload `.txt`, `.md`, `.pdf`, `.docx` or `.mp3` files, and the query processor indexes them and builds a knowledge graph of entities and facts. Authority is local push Personalized PageRank over the whole workspace graph.

This project uses Python 3.12.

<img width="4111" height="1270" alt="image" src="https://github.com/user-attachments/assets/791b172b-8a35-4b39-8470-c47d323bee05" />

<img width="3448" height="1817" alt="image" src="https://github.com/user-attachments/assets/10c695cb-3b92-4891-abd1-0cad5df408ba" />

## Architecture Flow

```
React UI (port 3000)
    │  HTTP POST /api/v1/search {prompt, topK, corpusId, fileIds}
    ▼
Spring Boot orchestration layer (port 8080)
    │  gRPC StartProcessing (server streaming)
    ▼
Python Query Processor (port 50051)
    │  → Milvus (vectors), Neo4j (graph), MinIO (source files)
    │  → LangGraph agent: planner → retrieval → analyst → synthesizer
    ▼
Streams back as SSE: status / agent_step / chunk → done (or error)
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full design.

### How a query is answered

1. **Plan**: the planner searches the question as written, or splits it into up to 3 subqueries.
2. **Retrieve**: the question is embedded (all-MiniLM-L6-v2, 384 dimensions) and searched in Milvus, then expanded through the graph with local push Personalized PageRank, which also gives each chunk its authority score. Neo4j neighbours are fetched only as the push reaches them, so the cost doesn't grow with corpus size. Both modes use the same function ([local_ppr.py](query_processor/app/local_ppr.py)):
   - Wiki: seeds are the vector hits. The walk follows the chunk relations from ingestion, weighted by the LLM's confidence.
   - Uploads: seeds are the entities the question names plus the vector hits. The walk follows `NEXT_CHUNK`, `MENTIONS` and `FACT` edges, and knowledge-graph fact paths from the named entities add more chunks.
   - If PPR fails, retrieval falls back to PageRank over the subgraph of the retrieved chunks.
3. **Rank**: each score is divided by the highest score among the candidates, then combined:
   - Default: `0.6 × semantic + 0.4 × authority`
   - Uploads, when fact paths are found: `0.5 × semantic + 0.2 × authority + 0.3 × fact-path relevance`
4. **Refine**: the analyst checks the evidence for gaps and can start up to 2 more retrieval rounds. In upload workspaces whose question names known entities, a graph coverage check replaces the model call. The context is enough once every named entity appears in some retrieved chunk. Otherwise the next round searches the strongest fact of up to 3 missing entities.
5. **Synthesize**: the answer model writes the answer from the top evidence, streamed word by word.

### Upload workspace features

- **Knowledge graph**: after a file is indexed, a local model extracts entities and facts in the background, stored as `(:Chunk)-[:MENTIONS]->(:Entity)` and `(:Entity)-[:FACT]->(:Entity)`.
- **Entity resolution**: entities that name the same concept are merged into one node. Merges cover the same wording once normalised, or a close meaning with the same main noun. The other names are kept as aliases.
- **File similarity**: questions such as "which files are most similar?" are answered by comparing the files' graphs, with file-level embedding similarity as a second check.
- **Corpus map**: a knowledge graph view of the workspace. Click an entity to see its relations. A second tab groups chunks into themed clusters by embedding similarity.
- **Evidence and reasoning paths**: the answer shows the evidence chunks it used and the fact paths the graph followed. Hovering over an evidence chunk highlights it in the corpus map.

## Prerequisites

- **Java 21+**: verify with `java -version`
- **Maven**, or use the bundled `mvnw`
- **Python 3.12**
- **Node.js and npm**, for the React UI
- **Docker Desktop**, running `milvus-standalone`, `etcd`, `neo4j` and `minio`
- **An OpenAI-compatible LLM server**, by default Ollama at `http://localhost:11434/v1`. Pull the models you configure (see [Configuration](#configuration)), for example:
  ```
  ollama pull qwen2.5:1.5b
  ollama pull qwen3:4b-instruct
  ```

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

### 2. Build the wiki corpus

[ingestion_pipeline](ingestion_pipeline/README.md)

```
cd ingestion_pipeline
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
python upload_to_minio.py   # upload all .txt files from raw_data into MinIO's rag-datasets bucket
python main.py              # chunk, extract relations with the LLM, write the graph to Neo4j
python index_to_milvus.py   # embed the Neo4j chunks into Milvus; wiki search finds nothing without this
```

Skip this step if you only want to use upload workspaces.

### 3. Start the Python Query Processor

[query_processor](query_processor/README.md)

```
cd query_processor
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
```

Generate the Python gRPC stubs from the proto file:
```
cd proto
python -m grpc_tools.protoc -I. --python_out=../app --grpc_python_out=../app query.proto
cd ..
```
This creates two files in `app/`:
- `app/query_pb2.py`: the message classes
- `app/query_pb2_grpc.py`: the server/client stubs

Run the server:
```
cd app
python main.py
```

Leave this terminal running.

### 4. Run Spring Boot

[orchestration_layer](orchestration_layer/README.md)

In a **new terminal**:

```powershell
cd orchestration_layer
mvnw compile -DskipTests
mvnw spring-boot:run
```

Compiling also generates the Java gRPC stubs from `src/main/proto/query.proto`.

### 5. (Optional) Start the React UI

[web_search_client](web_search_client/README.md)

```powershell
cd web_search_client
npm install
npm start
```

Then open `http://localhost:3000`. Spring Boot also serves the last React build that was copied into its `static/` folder at `http://localhost:8080`.

## API Endpoints

All paths are served by the orchestration layer on port 8080.

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/search` | Submit a search query. Body: `{"prompt": "...", "topK": 5, "corpusId": "wiki", "fileIds": []}`. `corpusId` defaults to `wiki`. `fileIds` optionally limits an upload-workspace search to those files. Returns an SSE stream. |
| `GET` | `/api/v1/chunk-source?uri=s3://...` | Full text of a source document, fetched through the query processor so the browser never talks to MinIO. |
| `POST` | `/api/v1/corpora` | Create an isolated upload workspace. Returns a UUID `corpusId`. |
| `POST` | `/api/v1/corpora/{corpusId}/files` | Upload and index `.txt`, `.md`, `.pdf`, `.docx` or `.mp3` files using multipart field `files` (up to 5 MB each, 1–5 files per request). Returns one result per file. |
| `GET` | `/api/v1/corpora/{corpusId}/files` | List files indexed in that workspace. |
| `GET` | `/api/v1/corpora/{corpusId}/files/{fileId}` | Full text of one uploaded file, for the preview panel. |
| `DELETE` | `/api/v1/corpora/{corpusId}/files/{fileId}` | Remove one file from object storage, the graph and the vector index. |
| `GET` | `/api/v1/corpora/{corpusId}/map?fileIds=a,b` | Corpus map: entities, relationships and chunk clusters. `fileIds` is optional. |
| `GET` | `/api/v1/health` | Health check. Returns `{"status": "UP", "service": "orchestration-layer"}`. |

The React UI offers **Wiki dump** and **My files** modes. In My files mode, each indexed file has a checkbox, so selected files can be previewed, deleted or chosen as the only files a question searches.

Uploads are chunked, embedded and indexed before the upload request returns, so vector search works right away. Knowledge-graph extraction is queued and runs in the background, and it pauses while a search is running. Graph features (fact paths, local PPR over entities, the coverage check, the corpus map's fact edges) appear as extraction progresses. Until facts exist, the corpus map links entities that appear in the same sentence.

Each workspace gets its own Milvus collection, and its Neo4j nodes are tagged with the workspace ID, so one workspace's search never returns another's chunks. Wiki requests use the `ai_docs` collection. Workspace IDs are kept in browser local storage and act like bearer tokens: add authentication and ownership checks before exposing this API to untrusted users.

## SSE Event Types

The `/api/v1/search` endpoint returns a `text/event-stream` with these event names:

| Event | Data Fields | Description |
|-------|-------------|-------------|
| `status` | `{stage, message}` | Pipeline stage: `PLANNING` → `RETRIEVING` → `ANALYZING` → (`RETRIEVING` again for each extra round) → `SYNTHESIZING` |
| `agent_step` | `{agent, detail}` | What each agent node decided, for the agent trace |
| `chunk` | `{content, sourceId}` | Answer text, streamed as the model writes it |
| `done` | `{answer, sources[], confidence, chunks[], graphPaths[], graphSeeds[]}` | Final answer, its evidence chunks, and the knowledge-graph fact paths behind it with the entities the walk started from |
| `error` | `{message}` | Error information, including a message when the stream times out |

## Configuration

Set these as environment variables. Docker Compose reads them from `.env` in the project root.

**Orchestration layer (Spring Boot)**

| Variable | Default | Description |
|----------|---------|-------------|
| `SERVER_PORT` | `8080` | Spring Boot port |
| `GRPC_CLIENT_HOST` | `localhost` | Query processor host |
| `GRPC_CLIENT_PORT` | `50051` | Query processor gRPC port |
| `SEARCH_SSE_TIMEOUT_MS` | `900000` | How long a search stream stays open before the UI is told it timed out |

**Query processor and ingestion pipeline**

| Variable | Default | Description |
|----------|---------|-------------|
| `EDGE_MODEL_BASE_URL` | `http://localhost:11434/v1` | OpenAI-compatible LLM endpoint |
| `LLM_API_KEY` | `ollama` | API key for that endpoint |
| `EDGE_MODEL` | `qwen3:4b-instruct` (query processor), `qwen2.5:1.5b` (ingestion) | Planner and analyst model; relation labelling in the ingestion pipeline |
| `ANSWER_MODEL` | `qwen3:4b-instruct` | Model that writes the final answer |
| `LLM_TIMEOUT_SECONDS` | `600` | LLM request timeout |
| `KG_ENABLED` | `true` | Extract a knowledge graph from uploads |
| `KG_MODEL` | `qwen2.5:1.5b` | Ollama model for knowledge-graph extraction |
| `KG_OLLAMA_URL` | `EDGE_MODEL_BASE_URL` without `/v1` | Ollama URL for extraction |
| `KG_SEARCH_ENABLED` | `true` | Use knowledge-graph fact paths in upload searches |
| `WHISPER_MODEL` | `base` | Speech-to-text model for `.mp3` uploads (`tiny`, `base`, `small`, `medium`, `large-v3`) |
| `WHISPER_DEVICE` / `WHISPER_COMPUTE_TYPE` | `cpu` / `int8` | Whisper runtime settings |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1500` / `200` | Chunking settings |
| `MILVUS_HOST` / `MILVUS_PORT` | `localhost` / `19530` | Milvus connection |
| `NEO4J_URI` / `NEO4J_USER` / `NEO4J_PASSWORD` | `bolt://localhost:7687` / `neo4j` / `password123` | Neo4j connection |
| `MINIO_ENDPOINT` / `MINIO_BUCKET` | `http://localhost:9000` / `rag-datasets` | MinIO connection |
| `MINIO_ROOT_USER` / `MINIO_ROOT_PASSWORD` | `minioadmin` / `minioadmin` | MinIO credentials |

In Docker, `.env` keeps host-side values (`localhost`). The containers reach a host Ollama through `DOCKER_EDGE_MODEL_BASE_URL`, which defaults to `http://host.docker.internal:11434/v1`.

## Troubleshooting

- **`Connection refused: localhost:50051`**: the Python query processor isn't running. Start it first.
- **`Fail connecting to Milvus`**: run `docker compose up milvus-standalone -d` and wait 10 seconds.
- **Proto compilation errors**: make sure `orchestration_layer/src/main/proto/query.proto` exists and matches `query_processor/proto/query.proto`.
- **Search times out or is very slow**: the LLMs are probably running on CPU. Use smaller models, ask shorter questions, or raise `SEARCH_SSE_TIMEOUT_MS`.
- **No graph features in a new workspace**: knowledge-graph extraction is still running in the background (roughly 4 s per sentence on CPU).

## How to run in Docker

[docker-compose.yml](docker-compose.yml) builds every image from the `Dockerfile.render` variants. These install Python packages straight from PyPI, and the orchestration image compiles the JAR itself.

### 1. Refresh the React build (only when the UI has changed)

The orchestration image serves whatever is in `orchestration_layer/src/main/resources/static/`, so copy a fresh React build there first:

```powershell
cd web_search_client
npm install
npm run build        # creates web_search_client/build/
cd ..
Copy-Item -Recurse web_search_client\build\* orchestration_layer\src\main\resources\static\ -Force
```

### 2. Build and start everything

```powershell
docker compose build
docker compose up -d
```

In China, override the package index with the Tsinghua mirror:
```powershell
docker compose build --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
```

### 3. Monitor and verify

```powershell
docker compose logs -f ingestion_pipeline
docker compose logs -f query_processor
docker compose logs -f orchestration_layer
docker compose ps

# If you see Created instead of Up, start them:
docker compose start
```

The ingestion pipeline is a one-shot job: it builds the wiki corpus and exits.

## Publish to Docker Hub and run on another device

Only the three app images are published. The wiki data is baked into the ingestion image, and the React build into the orchestration image. Milvus, Neo4j, MinIO and etcd come from their public images.

### On this machine: build and push

1. Add your Docker Hub username to `.env` (see [.env.example](.env.example)):
   ```
   DOCKERHUB_USER=your-dockerhub-username
   TAG=latest
   ```
2. If the UI has changed, refresh the React build first (see [step 1 above](#1-refresh-the-react-build-only-when-the-ui-has-changed)).
3. Log in, build and push:
   ```powershell
   docker login
   docker compose build ingestion_pipeline query_processor orchestration_layer
   docker compose push ingestion_pipeline query_processor orchestration_layer
   ```

These images are `linux/amd64`, which runs on Windows, Linux and Intel Macs. For Apple Silicon or other ARM machines, push multi-platform images with buildx instead of step 3 (slower, because ARM is emulated):
```powershell
docker buildx create --use --name multi
$u = "your-dockerhub-username"
docker buildx build --platform linux/amd64,linux/arm64 -f ingestion_pipeline/Dockerfile.render -t "$u/graphrag-ingestion:latest" --push ingestion_pipeline
docker buildx build --platform linux/amd64,linux/arm64 -f query_processor/Dockerfile.render -t "$u/graphrag-query-processor:latest" --push query_processor
docker buildx build --platform linux/amd64,linux/arm64 -f orchestration_layer/Dockerfile.render -t "$u/graphrag-orchestration:latest" --push orchestration_layer
```

Docker Hub repositories are public by default. If you make them private, the other device must `docker login` before pulling.

### On the other device: pull and run

The other device needs only Docker and two files in one folder. Nothing else is installed, not even Ollama:
- [docker-compose.hub.yml](docker-compose.hub.yml), renamed to `docker-compose.yml`
- `.env`, copied from [.env.example](.env.example), with the same `DOCKERHUB_USER` and `TAG`

Then start everything:
```
docker compose up -d
```

Open `http://localhost:8080`; Spring Boot serves the UI.

What happens on first start:
1. The `ollama` container starts, and the one-shot `ollama-pull` container downloads the three models named in `.env` (`EDGE_MODEL`, `ANSWER_MODEL`, `KG_MODEL`). This is a few GB, so it takes a while. Follow it with `docker compose logs -f ollama-pull`. The models stay in the `ollama_data` volume, so later starts skip the download.
2. The query processor and ingestion job wait for the download to finish, then start.
3. The ingestion container builds the wiki corpus, which takes a while because it calls the LLM. Follow it with `docker compose logs -f ingestion_pipeline`. Upload workspaces work as soon as the query processor is up.

The embedding and Whisper models also download on first use, so the device needs internet access the first time.

By default the models run on the CPU inside the container. To speed them up:
- **NVIDIA GPU** (Linux, or Windows with WSL2 and the NVIDIA Container Toolkit): uncomment the `deploy:` block under the `ollama` service.
- **Mac**: Docker on macOS can't use the Apple GPU. For GPU speed, install the Ollama app on the Mac and set `DOCKER_EDGE_MODEL_BASE_URL=http://host.docker.internal:11434/v1` in `.env`. The models are then downloaded into the container's Ollama, but the app uses the host's Ollama, so pull them there too.
- **Another OpenAI-compatible server**: set `DOCKER_EDGE_MODEL_BASE_URL` and `LLM_API_KEY`. If its model names aren't on the Ollama registry, `ollama-pull` fails and the app containers won't start. In that case, remove `ollama-pull` from their `depends_on`.

To update the other device after pushing new images: `docker compose pull && docker compose up -d`.

### Building a fat JAR locally (optional)

```powershell
cd orchestration_layer
mvnw clean package -DskipTests
```

If the build fails because of stale `.class` files in `target/`:
```powershell
rm src\main\java\com\com -Recurse -Force
mvnw clean package -DskipTests
```

### Offline Python builds (optional)

The plain `Dockerfile`s in `ingestion_pipeline/` and `query_processor/` install from a pre-downloaded `wheel_cache/` instead of PyPI, for machines whose Docker builds can't reach the internet. `docker compose` doesn't use them. To build them yourself:

```powershell
cd ingestion_pipeline && venv\Scripts\activate
Remove-Item -Recurse -Force wheel_cache -ErrorAction SilentlyContinue
pip download "torch==2.6.0+cpu" -d wheel_cache --platform linux_x86_64 --only-binary=:all: --python-version 312 --index-url https://download.pytorch.org/whl/cpu
pip download -r requirements.txt -d wheel_cache --platform manylinux2014_x86_64 --only-binary=:all: --python-version 312 --extra-index-url https://download.pytorch.org/whl/cpu
docker build -t graphrag-ingestion .
cd ..\query_processor
# repeat the same two pip download commands, then:
docker build -t graphrag-query-processor .
```


To publish, on this machine:

Add DOCKERHUB_USER=<your username> to .env.
Run:

docker login
docker compose build ingestion_pipeline query_processor orchestration_layer
docker compose push ingestion_pipeline query_processor orchestration_layer
To run, on the other device:

Put docker-compose.hub.yml (renamed to docker-compose.yml) and a .env with the same DOCKERHUB_USER in one folder.
Run docker compose up -d.
Open http://localhost:8080.