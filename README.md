# Final Year Project: Reimagine Google PageRank Search for Generative Information Retrieval

GraphRAG-Search combines graph-aware authority ranking with LLM answer synthesis. Instead of ranking hyperlinked web pages, it ranks chunks of unstructured text over a graph built automatically from the documents, then has an LLM agent write an answer from the best evidence and stream it to the browser.

It has two modes:

- **Wiki dump**: a fixed set of Wikipedia articles (WikiText). Each article is indexed exactly like an uploaded file, knowledge graph included, but offline: the build takes hours, so it's done once and shipped as a [snapshot](#build-the-wiki-snapshot) that a new install restores on first start.
- **My files**: upload workspaces. Users upload `.txt`, `.md`, `.pdf`, `.docx` or `.mp3` files, and the query processor indexes them and builds a knowledge graph of entities and facts in the background.

Both modes search the same kind of graph, and authority is local push Personalized PageRank over it.

The repository also contains a **VS Code extension for codebases** ([vscode-extension/](vscode-extension/README.md)). It applies the same retrieval to code in any language: a code graph from tree-sitter static analysis for Python and TypeScript/JavaScript/React (calls, component renders, imports, subclassing, type use) and from VS Code's language servers for every other language (fetched lazily as PPR reaches each symbol), embeddings in LanceDB, local push PPR seeded from the question and the cursor, a coverage check that puts every caller of a function in context before it changes, and an edit loop with diff previews, approval, git checkpoints and verification. It's a separate package. The text search and the extension share only [core/](core/graphrag_core) (local push PPR, ranking, coverage check, workflow skeleton, progress events, chat client).

```
core/                 shared: graphrag_core (standard library only)
query_processor/      text search (uses core/ for PPR and the chat client)
orchestration_layer/  web_search_client/  ingestion_pipeline/
vscode-extension/     the extension: TypeScript client + Python sidecar (uses core/)
```

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
2. **Retrieve**: the question is embedded (all-MiniLM-L6-v2, 384 dimensions) and searched in Milvus. Then, in both modes:
   - Knowledge-graph fact paths from the entities the question names add chunks vector search missed ([graph_retrieval.py](query_processor/app/graph_retrieval.py)). Only the facts within two hops of those entities are loaded, so this scales with the question, not the corpus.
   - Local push Personalized PageRank from the named entities and the vector hits gives each chunk its authority score, following `NEXT_CHUNK`, `MENTIONS` and `FACT` edges ([local_ppr.py](query_processor/app/local_ppr.py)). Neo4j neighbours are fetched only as the push reaches them.
   - If PPR fails, retrieval falls back to PageRank over the subgraph of the retrieved chunks.
3. **Rank**: each score is divided by the highest score among the candidates, then combined:
   - Default: `0.6 × semantic + 0.4 × authority`
   - When fact paths are found: `0.5 × semantic + 0.2 × authority + 0.3 × fact-path relevance`
4. **Refine**: the analyst checks the evidence for gaps and can start up to 2 more retrieval rounds. When the question names known entities, a graph coverage check replaces the model call. The context is enough once every named entity appears in some retrieved chunk. Otherwise the next round searches the strongest fact of up to 3 missing entities.
5. **Synthesize**: the answer model writes the answer from the top evidence, streamed word by word.

### Knowledge graph (both modes)

- **Extraction**: a local model reads each sentence and returns its facts as (subject, relation, object). A fact is kept only if both ends appear in that sentence, which is stored as its evidence. Facts are stored as `(:Chunk)-[:MENTIONS]->(:Entity)` and `(:Entity)-[:FACT]->(:Entity)`. Uploads are extracted in the background after indexing; the wiki was extracted when its snapshot was built.
- **Entity resolution**: entities that name the same concept are merged into one node. Merges cover the same wording once normalised, or a close meaning with the same main noun. The other names are kept as aliases.

### Upload workspace features

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

### 2. Load the wiki corpus

The wiki comes from a pre-built [snapshot](#build-the-wiki-snapshot). On a fresh install, the snapshot images load it (this also starts the services from step 1):

```powershell
docker compose --profile wiki up -d neo4j wiki-restore
```

The snapshot is only loaded into empty stores. If this machine's Neo4j already has data (for example uploads, or the wiki from an older version), the loaders leave it alone. To replace it with the snapshot, delete the volumes first with `docker compose down -v`, which also deletes every upload workspace.

Skip this step if you only want to use upload workspaces.

### 3. Start the Python Query Processor

[query_processor](query_processor/README.md)

```
cd query_processor
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
pip install -e ../core
```

`core/` is the small shared package (local push PPR, chat client) that the query processor and the [VS Code extension](vscode-extension/README.md) both use.

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

**Query processor and wiki build**

| Variable | Default | Description |
|----------|---------|-------------|
| `EDGE_MODEL_BASE_URL` | `http://localhost:11434/v1` | OpenAI-compatible LLM endpoint |
| `LLM_API_KEY` | `ollama` | API key for that endpoint |
| `EDGE_MODEL` | `qwen3:4b-instruct` | Planner and analyst model |
| `ANSWER_MODEL` | `qwen3:4b-instruct` | Model that writes the final answer |
| `LLM_TIMEOUT_SECONDS` | `600` | LLM request timeout |
| `KG_ENABLED` | `true` | Extract a knowledge graph from uploads (and, in the wiki build, from the wiki) |
| `KG_MODEL` | `qwen2.5:1.5b` | Ollama model for knowledge-graph extraction |
| `KG_OLLAMA_URL` | `EDGE_MODEL_BASE_URL` without `/v1` | Ollama URL for extraction |
| `KG_PARALLEL` | `2` | Sentences sent to Ollama at once during extraction. Ollama must allow as many with `OLLAMA_NUM_PARALLEL` (set for the bundled container in `docker-compose.hub.yml`; for Ollama on the host, set it there and restart Ollama). `1` sends them one by one |
| `KG_SEARCH_ENABLED` | `true` | Use knowledge-graph fact paths in searches |
| `WIKI_FILES` | `wiki.test.txt` | Wiki build only: WikiText files in `ingestion_pipeline/raw_data/` to build, comma-separated |
| `WIKI_MAX_ARTICLES` | `0` (all) | Wiki build only: stop after this many articles |
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
- **Wiki search finds nothing**: the wiki snapshot isn't loaded. Check `docker compose logs wiki-neo4j-load wiki-restore`. Both only load into empty stores, so an install that already had data keeps it.

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
docker compose logs -f query_processor
docker compose logs -f orchestration_layer
docker compose ps

# If you see Created instead of Up, start them:
docker compose start
```

`docker compose up -d` doesn't load the wiki; add `--profile wiki` for that (see [Load the wiki corpus](#2-load-the-wiki-corpus)).

## Build the wiki snapshot

The wiki is built once, on one machine, then shipped as a snapshot. The build runs [build_wiki.py](query_processor/app/build_wiki.py):
1. It splits the WikiText files into articles and undoes WikiText's tokenisation (`guest @-@ starring` becomes `guest-starring`).
2. It indexes each article like an uploaded file.
3. It extracts the knowledge graph sentence by sentence with `KG_MODEL`, merging entities that name the same concept.

Extraction takes about 4 s per sentence on a CPU. The default, `wiki.test.txt` (62 articles, about 9,000 sentences), therefore takes around 10 hours; all three files (751 articles) would take days. Set `WIKI_FILES` and `WIKI_MAX_ARTICLES` in `.env` to choose. A GPU, or the Ollama app on a Mac, is much faster. The build is resumable: if it stops, run the same command again and it continues where it left off.

It runs in a separate Compose project with its own empty databases, so the snapshot contains the wiki and none of your uploads. It uses the Ollama on this machine, so pull `KG_MODEL` first.

```powershell
# 1. Build (hours). Follow it in the logs; it ends with "Wiki build complete"
docker compose -f docker-compose.wiki-build.yml up -d --build
docker compose -f docker-compose.wiki-build.yml logs -f wiki-build

# 2. Export the vectors and article texts to ingestion_pipeline/snapshot/
docker compose -f docker-compose.wiki-build.yml run --rm wiki-export

# 3. Dump the graph with neo4j-admin (Neo4j must be stopped)
docker compose -f docker-compose.wiki-build.yml stop neo4j
docker compose -f docker-compose.wiki-build.yml run --rm neo4j-dump

# 4. Build the two snapshot images (and push them, see below)
docker compose --profile wiki build wiki-neo4j-load wiki-restore
```

`ingestion_pipeline/snapshot/` then holds `neo4j.dump`, `milvus.jsonl.gz`, `minio.jsonl.gz` and `manifest.json` (build settings and counts). It isn't committed; the images carry it. To stop the build project, run `docker compose -f docker-compose.wiki-build.yml down`. Add `-v` to delete its databases as well; without `-v`, a later build continues from them.

What a new install does with the snapshot:
- `wiki-neo4j-load` runs `neo4j-admin database load` into the empty `neo4j_data` volume before Neo4j first starts.
- `wiki-restore` writes the vectors into the `ai_docs` Milvus collection and the article texts into MinIO.

Both exit at once on later starts. Neither overwrites existing data.

The snapshot is tied to the settings it was built with: the embedding model (all-MiniLM-L6-v2), `CHUNK_SIZE`/`CHUNK_OVERLAP`, `KG_MODEL`, and the Neo4j version (pinned to `neo4j:5.26` in every Compose file, because a dump only loads into its own version or a later one). Rebuild it when you change any of these, or the WikiText files.

## Publish to Docker Hub and run on another device

Four images are published: the query processor, the orchestration layer (with the React build inside), and the two wiki snapshot images (`graphrag-wiki-neo4j` and `graphrag-wiki-restore`). Milvus, Neo4j, MinIO, etcd and Ollama come from their public images. [Build the wiki snapshot](#build-the-wiki-snapshot) first.

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
   docker compose --profile wiki build query_processor orchestration_layer wiki-neo4j-load wiki-restore
   docker compose --profile wiki push query_processor orchestration_layer wiki-neo4j-load wiki-restore
   ```

These images are `linux/amd64`, which runs on Windows, Linux and Intel Macs. For Apple Silicon or other ARM machines, push multi-platform images with buildx instead of step 3 (slower, because ARM is emulated):
```powershell
docker buildx create --use --name multi
$u = "your-dockerhub-username"
docker buildx build --platform linux/amd64,linux/arm64 -f query_processor/Dockerfile.render --build-context core=core -t "$u/graphrag-query-processor:latest" --push query_processor
docker buildx build --platform linux/amd64,linux/arm64 -f orchestration_layer/Dockerfile.render -t "$u/graphrag-orchestration:latest" --push orchestration_layer
docker buildx build --platform linux/amd64,linux/arm64 -f ingestion_pipeline/Dockerfile.neo4j -t "$u/graphrag-wiki-neo4j:latest" --push ingestion_pipeline
docker buildx build --platform linux/amd64,linux/arm64 -f ingestion_pipeline/Dockerfile -t "$u/graphrag-wiki-restore:latest" --push ingestion_pipeline
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
1. `wiki-neo4j-load` loads the wiki's graph into the empty Neo4j volume, then Neo4j starts. `wiki-restore` writes the wiki's vectors into Milvus and its article texts into MinIO. Both take seconds and do nothing on later starts.
2. The `ollama` container starts, and the one-shot `ollama-pull` container downloads the three models named in `.env` (`EDGE_MODEL`, `ANSWER_MODEL`, `KG_MODEL`). This is a few GB, so it takes a while. Follow it with `docker compose logs -f ollama-pull`. The models stay in the `ollama_data` volume, so later starts skip the download.
3. The query processor waits for the download to finish, then starts. The wiki, with its knowledge graph, is searchable straight away; nothing is built on this device.

The embedding and Whisper models also download on first use, so the device needs internet access the first time.

By default the models run on the CPU inside the container. To speed them up:
- **NVIDIA GPU** (Linux, or Windows with WSL2 and the NVIDIA Container Toolkit): uncomment the `deploy:` block under the `ollama` service.
- **Mac**: Docker on macOS can't use the Apple GPU. For GPU speed, install the Ollama app on the Mac and set `DOCKER_EDGE_MODEL_BASE_URL=http://host.docker.internal:11434/v1` in `.env`. The models are then downloaded into the container's Ollama, but the app uses the host's Ollama, so pull them there too.
- **Another OpenAI-compatible server**: set `DOCKER_EDGE_MODEL_BASE_URL` and `LLM_API_KEY`. If its model names aren't on the Ollama registry, `ollama-pull` fails and the app containers won't start. In that case, remove `ollama-pull` from their `depends_on`.

To update the other device after pushing new images: `docker compose pull && docker compose up -d`. A new wiki snapshot isn't applied to a device that already has one, because the loaders never overwrite. To switch to it, run `docker compose down -v` first, which also deletes that device's uploads.

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

The plain `Dockerfile` in `query_processor/` installs from a pre-downloaded `wheel_cache/` instead of PyPI, for machines whose Docker builds can't reach the internet. `docker compose` doesn't use it. To build it yourself:

```powershell
cd query_processor && venv\Scripts\activate
Remove-Item -Recurse -Force wheel_cache -ErrorAction SilentlyContinue
pip download "torch==2.6.0+cpu" -d wheel_cache --platform linux_x86_64 --only-binary=:all: --python-version 312 --index-url https://download.pytorch.org/whl/cpu
pip download -r requirements.txt -d wheel_cache --platform manylinux2014_x86_64 --only-binary=:all: --python-version 312 --extra-index-url https://download.pytorch.org/whl/cpu
docker build -t graphrag-query-processor .
```


To publish, on this machine:

Add DOCKERHUB_USER=<your username> to .env.

Run:

```powershell
docker login

docker compose --profile wiki build query_processor orchestration_layer wiki-neo4j-load wiki-restore

docker compose --profile wiki push query_processor orchestration_layer wiki-neo4j-load wiki-restore
```

To run, on the other device:

Create a directory of your choice (e.g. ./GraphRAG-Search) to hold the docker-compose.yml and .env files.

```powershell
wget -O docker-compose.yml https://raw.githubusercontent.com/williamfaraday123/GraphRAG-Search/main/docker-compose.hub.yml

wget -O .env https://raw.githubusercontent.com/williamfaraday123/GraphRAG-Search/main/.env.example

Run docker compose up -d.
```
Open http://localhost:8080.