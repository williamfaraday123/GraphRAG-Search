# 🚀 Project Runbook — AI-Powered Search Engine

> **Reimagining Google PageRank for Generative Information Retrieval**
>
> An AI search engine that combines vector search (Milvus), knowledge graphs (Neo4j), semantic chunking (LangChain), graph-based re-ranking (PageRank), and LLM-powered answer synthesis (LangGraph).

---

## 📋 Table of Contents

1. [System Architecture Overview](#-system-architecture-overview)
2. [Prerequisites](#-prerequisites)
3. [Quick Start — Docker Compose](#-quick-start--docker-compose)
4. [How Each Component Works](#-how-each-component-works)
   - [Infrastructure Services](#1-infrastructure-services)
   - [Ingestion Pipeline](#2-ingestion-pipeline)
   - [Query Processor](#3-query-processor)
   - [Orchestration Layer](#4-orchestration-layer)
5. [Running Components Individually](#-running-components-individually)
6. [Cloud Deployment (Terraform)](#-cloud-deployment-terraform)
7. [CI/CD Pipeline](#-cicd-pipeline)
8. [Port Reference](#-port-reference)
9. [Troubleshooting](#-troubleshooting)

---

## 🧠 System Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Client Layer                              │
│               [React Web Search UI]                          │
└──────────────────────┬──────────────────────────────────────┘
                       │ HTTP / SSE
┌──────────────────────▼──────────────────────────────────────┐
│           Orchestration Layer (Spring Boot)                  │
│         ┌──────────────────────────────────────┐             │
│         │  gRPC Client → Query Processor       │             │
│         │  SSE → React UI (typewriter effect)  │             │
│         └──────────────────────────────────────┘             │
└──────────────────────┬──────────────────────────────────────┘
                       │ gRPC (stream)
┌──────────────────────▼──────────────────────────────────────┐
│              Intelligence Engine (FastAPI + LangGraph)        │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Query Processor Servicer (gRPC Server)                │  │
│  │  1. HybridRetriever → Milvus + Neo4j + MinIO          │  │
│  │  2. LangGraph Pregel Engine (PageRank re-ranking)      │  │
│  │  3. LLM Aggregator → Final Answer Synthesis            │  │
│  └────────────────────────────────────────────────────────┘  │
└────┬──────────┬──────────────┬───────────────────────────────┘
     │          │              │
┌────▼──┐ ┌────▼─────┐ ┌─────▼──────────┐
│ Milvus│ │  Neo4j   │ │    MinIO        │
│Vector │ │  Graph   │ │ Object Storage  │
│  DB   │ │   DB     │ │ (Source Docs)   │
└───────┘ └──────────┘ └─────────────────┘
     ▲          ▲
     │          │
┌────┴──────────┴──────────────────────────────────────────────┐
│              Ingestion Pipeline (Python)                      │
│  ┌──────────┐ ┌───────┐ ┌────────────┐ ┌──────────────┐     │
│  │DataLoader│→│Chunker│→│EdgeGenerator│→│ GraphBuilder │→Neo4j│
│  │ (MinIO)  │ │(LangChain)│  (LLM)   │ │  (Neo4j)    │      │
│  └──────────┘ └───────┘ └────────────┘ └──────────────┘      │
└──────────────────────────────────────────────────────────────┘
```

**Data Flow (3 Phases):**

| Phase | What Happens | Components |
|---|---|---|
| **1. INGEST** | Documents loaded → chunked → edges identified via LLM → stored in Neo4j | `ingestion_pipeline/` |
| **2. INDEX** | Chunks embedded → stored in Milvus (vector search) | `query_processor/app/embedding.py` |
| **3. QUERY** | User question → vector search + graph traversal → PageRank re-rank → LLM answer | `query_processor/app/` |

---

## 📦 Prerequisites

| Tool | Version | Why |
|---|---|---|
| [Docker](https://docs.docker.com/engine/install/) | ≥ 24.x | Runs all services |
| [Docker Compose](https://docs.docker.com/compose/install/) | ≥ 2.24.x | Orchestrates multi-container setup |
| Python | 3.10+ | Ingestion pipeline & query processor |
| Java (JDK) | 25 | Orchestration layer (Spring Boot) |
| Maven | (bundled via `mvnw`) | Java build tool |

---

## 🐳 Quick Start — Docker Compose

### Step 1: Clone & Navigate

```powershell
cd C:\Users\estee\Desktop\reimagine-google-pagerank-search-for-generative-information-retrieval-2
```

### Step 2: Launch Everything

```powershell
docker compose up --build
```

This builds and starts **6 containers**:

| Container | Image | Purpose |
|---|---|---|
| `milvus-standalone` | `milvusdb/milvus:v2.4` | Vector similarity search |
| `neo4j` | `neo4j:5` | Knowledge graph storage |
| `minio` | `minio/minio` | S3-compatible object storage |
| `etcd` | `quay.io/coreos/etcd:v3.5` | Milvus metadata store |
| `ai-ingestion` | (local build) | Document ingestion pipeline |
| `ai-query-api` | (local build) | gRPC query processor API |

### Step 3: Verify Everything is Running

```powershell
docker compose ps
```

| Port | Service | URL |
|---|---|---|
| `19530` | Milvus | (gRPC SDK) |
| `7474` | Neo4j Browser | http://localhost:7474 |
| `7687` | Neo4j Bolt | (driver connection) |
| `9000` | MinIO API | (S3 SDK) |
| `9001` | MinIO Console | http://localhost:9001 |
| `8000` | Query Processor | (gRPC API) |

**Neo4j credentials:** `neo4j` / `password123`  
**MinIO credentials:** `minioadmin` / `minioadmin`

### Step 4: Stop / Clean Up

```powershell
docker compose stop      # Stop containers (preserves volumes/data)
docker compose down      # Delete containers & networks
docker compose down -v   # Full reset — also deletes volumes
```

---

## 🔧 How Each Component Works

---

### 1. Infrastructure Services

#### **Milvus** — Vector Database
- **Role:** Stores numerical embeddings (vector representations) of text chunks.
- **Why:** Enables semantic similarity search — finding content by *meaning*, not just keywords.
- **Port:** `19530` (gRPC SDK)
- **Dimension:** 384 (matching `all-MiniLM-L6-v2` embedding model).
- **Dependencies:** `etcd` (metadata) + `minio` (storage).

#### **Neo4j** — Graph Database
- **Role:** Stores the knowledge graph — `Chunk` nodes connected by `RELATION` edges.
- **Why:** Captures logical relationships between chunks (e.g., `PREMISE_OF`, `ELABORATES_ON`, `CONTRADICTS`).
- **Ports:** `7474` (browser UI), `7687` (Bolt driver).
- **Schema:**
  ```cypher
  (c:Chunk {chunk_id, content, source, chunk_index, total_chunks})
  -[r:RELATION {type, weight, reason}]-> 
  (d:Chunk {...})
  ```

#### **MinIO** — Object Storage
- **Role:** Stores raw source documents (`.txt`, `.pdf`, `.md`).
- **Why:** Provides a central repository for documents before ingestion, and enables fetching full source content during query time.
- **Ports:** `9000` (API), `9001` (Console).
- **Default bucket:** `rag-datasets`

#### **Etcd** — Key-Value Store
- **Role:** Milvus metadata store — tracks collection schemas, segment states, and coordination data.
- **Why:** Required by Milvus to operate.

---

### 2. Ingestion Pipeline

**Location:** `ingestion_pipeline/`  
**Entry Point:** `ingestion_pipeline/main.py`  
**Purpose:** Transforms raw documents into a knowledge graph in Neo4j.

#### Pipeline Steps

```
Raw Documents (MinIO)
     │
     ▼
┌─────────────────┐
│  Data Loader     │  → Reads source files via S3 API from MinIO
│  (data_loader.py)│    (or from local folder `./raw_data`)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Document        │  → LangChain `RecursiveCharacterTextSplitter`
│  Chunker         │    splits text into semantic chunks (default: 1000 chars,
│  (chunker.py)    │    200 overlap). Each chunk gets a deterministic SHA-256 ID.
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Edge Generator  │  → For each adjacent chunk pair, calls an LLM (Qwen via
│  (edge_generator │    DashScope/Alibaba Cloud) to classify the relationship:
│   .py)           │    - PREMISE_OF    (A provides basis for B)
│                  │    - ELABORATES_ON (B adds detail to A)
│                  │    - CONTRASTS_WITH(B contradicts A)
│                  │    - SEQUENCE      (B follows A chronologically)
│                  │    - NONE          (no link — edge is skipped)
│                  │  → Also generates sequential `NEXT_CHUNK` edges without LLM.
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Graph Builder   │  → Writes nodes and edges to Neo4j using Cypher `MERGE`
│  (graph_builder  │    (idempotent — safe to re-run).
│   .py)           │  → `ingest_nodes()`: Creates `:Chunk` nodes.
│                  │  → `ingest_edges()`: Creates `:RELATION` relationships.
└─────────────────┘
         │
         ▼
   Neo4j Knowledge Graph
```

**To run locally (outside Docker):**

```powershell
cd ingestion_pipeline
pip install -r requirements.txt

# Set up environment
set DASHSCOPE_API_KEY=your-api-key-here
set NEO4J_URI=bolt://localhost:7687

# Place documents in ./raw_data (or configure DATA_SOURCE_PATH)
mkdir raw_data
# Copy your .txt files into raw_data/

python main.py
```

**Configuration** (`config.py`):

| Variable | Default | Description |
|---|---|---|
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j connection |
| `NEO4J_USER` | `neo4j` | Neo4j username |
| `NEO4J_PASSWORD` | `password123` | Neo4j password |
| `DASHSCOPE_API_KEY` | *(required)* | LLM API key for edge detection |
| `CHUNK_SIZE` | `1000` | Characters per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between adjacent chunks |
| `DATA_SOURCE_PATH` | `./raw_data` | Input folder for documents |

---

### 3. Query Processor

**Location:** `query_processor/`  
**Entry Point:** `query_processor/app/main.py`  
**Exposes:** gRPC API on port `8000`  
**Purpose:** Processes search queries through hybrid retrieval + graph-powered re-ranking + LLM synthesis.

#### Query Flow

```
User Query (via gRPC)
     │
     ▼
┌─────────────────────────────────────┐
│  Step 1: Hybrid Retrieval            │
│  ┌─────────────────────────────────┐ │
│  │ EmbeddingService                │ │  → Converts query text to vector
│  │ (embedding.py)                  │ │    using SentenceTransformers
│  │                                 │ │    (all-MiniLM-L6-v2, 384-dims)
│  └───────────┬─────────────────────┘ │
│              ▼                       │
│  ┌─────────────────────────────────┐ │
│  │ MilvusService (milvus_client.py)│ │  → Vector similarity search (COSINE)
│  │                                 │ │    Returns top-K candidate chunks
│  └───────────┬─────────────────────┘ │
│              ▼                       │
│  ┌─────────────────────────────────┐ │
│  │ Neo4jService (neo4j_client.py)  │ │  → Fetches the induced subgraph:
│  │                                 │ │    edges connecting candidate chunks
│  └───────────┬─────────────────────┘ │
│              ▼                       │
│  ┌─────────────────────────────────┐ │
│  │ ObjectStorageRetriever          │ │  → Fetches full source documents
│  │ (object_storage_client.py)      │ │    from MinIO for enrichment
│  └─────────────────────────────────┘ │
└──────────────────┬──────────────────┘
                   ▼
┌─────────────────────────────────────┐
│  Step 2: LangGraph Pregel Engine     │
│  (graph_engine.py)                  │
│                                     │
│  ┌─ Calculate PageRank scores ─────┐│
│  │  on the local sub-graph using   ││
│  │  NetworkX. This identifies      ││
│  │  "authority" nodes that are     ││
│  │  structurally important even    ││
│  │  if their vector score is lower.││
│  └─────────────────────────────────┘│
│                                     │
│  ┌─ Hybrid Scoring ────────────────┐│
│  │  FinalScore = (0.6 × Milvus)   ││
│  │              + (0.4 × PageRank) ││
│  │  Re-ranks chunks by this score. ││
│  └─────────────────────────────────┘│
│                                     │
│  ┌─ Iterative Refinement ─────────┐│
│  │  LangGraph loops (supersteps)  ││
│  │  until convergence, simulating ││
│  │  a Pregel-style consensus algo ││
│  └─────────────────────────────────┘│
└──────────────────┬──────────────────┘
                   ▼
┌─────────────────────────────────────┐
│  Step 3: Synthesis & Streaming       │
│                                     │
│  → LLM (via LangChain) generates    │
│    final answer from re-ranked      │
│    context.                         │
│                                     │
│  → Answer is split into words and   │
│    streamed back over gRPC, which   │
│    Spring Boot forwards via SSE to  │
│    the React UI (typewriter effect).│
└─────────────────────────────────────┘
```

#### gRPC Protocol (`proto/query.proto`)

```protobuf
service QueryProcessorService {
  rpc StartProcessing (SearchRequest) returns (stream SearchResponse);
}

message SearchRequest {
  string prompt = 1;
  int32 top_k = 2;
  bool use_graph_refinement = 3;
}

message SearchResponse {
  oneof response_type {
    ProcessingStatus status = 1;  // "RETRIEVING" | "ITERATING" | "SYNTHESIZING"
    StreamChunk chunk = 2;        // Word-by-word answer tokens
    FinalAnswer answer = 3;       // Final metadata with sources & confidence
  }
}
```

**To run locally (outside Docker):**

```powershell
cd query_processor
pip install -r requirements.txt

# Set environment
set MILVUS_HOST=localhost
set MILVUS_PORT=19530
set NEO4J_URI=bolt://localhost:7687

# Generate gRPC stubs (first time only)
python -m grpc_tools.protoc -I proto --python_out=app --grpc_python_out=app proto/query.proto

python app/main.py
```

---

### 4. Orchestration Layer

**Location:** `orchestration_layer/`  
**Framework:** Spring Boot (Java 25) + Spring gRPC  
**Purpose:** Acts as the API gateway — receives HTTP requests from a React UI, forwards them as gRPC streams to the Query Processor, and streams results back via SSE (Server-Sent Events).

```
React UI                Spring Boot                Query Processor
   │                        │                           │
   │  HTTP POST /api/v1     │                           │
   │  search?q="..."        │                           │
   │───────────────────────►│                           │
   │                        │  gRPC StartProcessing()   │
   │                        │──────────────────────────►│
   │                        │                           │
   │                        │◄─── stream: status ──────│
   │                        │     "RETRIEVING"          │
   │                        │                           │
   │                        │◄─── stream: status ──────│
   │                        │     "ITERATING"           │
   │                        │                           │
   │                        │◄─── stream: word tokens ─│
   │  SSE: word by word     │                           │
   │◄───────────────────────│                           │
   │  (typewriter effect)   │                           │
   │                        │◄─── stream: FinalAnswer ─│
   │  SSE: final metadata   │                           │
   │◄───────────────────────│                           │
```

**To run locally:**

```powershell
cd orchestration_layer
.\mvnw spring-boot:run
```

---

## 🧪 Running Components Individually

### Option A: Only Infrastructure (Databases)

Useful if you want to run the Python components locally during development:

```powershell
docker compose up milvus-standalone neo4j minio etcd
```

### Option B: Ingestion Pipeline Only (Local)

```powershell
# 1. Start databases only
docker compose up milvus-standalone neo4j minio etcd -d

# 2. Run ingestion
cd ingestion_pipeline
pip install -r requirements.txt
set DASHSCOPE_API_KEY=your-key
python main.py
```

### Option C: Query Processor Only (Local)

```powershell
# 1. Start databases
docker compose up milvus-standalone neo4j -d

# 2. Run query processor
cd query_processor
pip install -r requirements.txt
python app/main.py
```

### Option D: Full Stack with Local Development

```powershell
# Terminal 1 — Databases
docker compose up milvus-standalone neo4j minio etcd

# Terminal 2 — Ingestion
cd ingestion_pipeline && python main.py

# Terminal 3 — Query Processor
cd query_processor && python app/main.py

# Terminal 4 — Orchestration (optional)
cd orchestration_layer && .\mvnw spring-boot:run
```

---

## ☁️ Cloud Deployment (Terraform)

The `main.tf` provisions the entire stack on **Alibaba Cloud** (Alibaba Cloud).

**Architecture:**
- **1 x `ecs.g7.4xlarge`** (16 vCPU, 64 GB RAM) — runs all Docker containers (Milvus, Neo4j, MinIO, etc.)
- **1 x `ecs.g7.large`** — runs the Spring Boot orchestration layer
- VPC + VSwitch for network isolation
- Security group with HTTP (80), SSH (22), and internal traffic rules

**Deploy:**

```powershell
# Install Terraform first
terraform init
terraform plan
terraform apply
```

The user-data script auto-installs Docker + Docker Compose and starts all services via the same `docker-compose.yml` configuration embedded in the Terraform template.

---

## 🔄 CI/CD Pipeline

The `gitflic-ci.yaml` provides a CI/CD pipeline for **GitFlic** (a Russian Git platform):

| Stage | Action |
|---|---|
| `docker-build` | Builds Docker image, tags with commit SHA, pushes to registry |
| (on default branch) | Also tags and pushes as `:latest` |

---

## 📌 Port Reference

| Port | Service | Protocol | Purpose |
|---|---|---|---|
| `22` | ECS (cloud) | SSH | Admin access |
| `80` | Spring Boot (cloud) | HTTP | Search API endpoint |
| `7474` | Neo4j | HTTP | Browser UI |
| `7687` | Neo4j | Bolt | Driver connections |
| `8000` | Query Processor | gRPC | Search queries |
| `9000` | MinIO | HTTP/S3 | API |
| `9001` | MinIO | HTTP | Console UI |
| `19530` | Milvus | gRPC | SDK connections |

---

## 🐛 Troubleshooting

### "Port already in use"
```powershell
# Find what's using a port
netstat -ano | findstr :7474
# Kill the process
taskkill /PID <PID> /F
```

### "Connection refused to Milvus/Neo4j"
- Ensure all database containers are running: `docker compose ps`
- Check logs: `docker compose logs milvus-standalone`
- When running locally (not Docker), update connection strings in code to use `localhost` instead of container hostnames.

### Ingestion fails with "API key required"
- Set the `DASHSCOPE_API_KEY` environment variable.
- Get an API key from [Alibaba Cloud DashScope](https://dashscope.aliyun.com/).

### Milvus search returns empty results
- Ensure the collection `ai_docs` exists and has data.
- The embedding dimension (384) must match between `embedding.py` and the Milvus collection schema.

### Docker build is slow
- Docker layer caching is configured — only `COPY . .` (the last line) re-runs when source code changes. Requirements are cached.
- First build will be slow as it downloads base images and Python dependencies.

---

> **Project Structure Reference**
>
> ```
> root/
> ├── docker-compose.yml          # Multi-service orchestration
> ├── main.tf                     # Alibaba Cloud Terraform deployment
> ├── gitflic-ci.yaml            # CI/CD pipeline
> ├── ingestion_pipeline/         # Document → Knowledge Graph pipeline
> ├── query_processor/            # gRPC query API (FastAPI + LangGraph)
> ├── orchestration_layer/        # Spring Boot API gateway
> ├── diagrams/                   # PlantUML architecture diagrams
> ├── volumes/                    # Docker persistent data (gitignored)
> └── README.md                   # Project overview
> ```





# Orchestration Layer — Spring Boot API Gateway

This is the REST → gRPC bridge layer. It receives search queries from the React UI via HTTP POST, forwards them to the Python Query Processor via gRPC, and streams results back to the UI using Server-Sent Events (SSE).

