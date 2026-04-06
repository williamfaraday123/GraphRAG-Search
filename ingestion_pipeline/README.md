Instructions to run this:
```
cd query_processor
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
cd app
python main.py
```

File structure:
```
query-processor/
├── app/
│   ├── __init__.py
│   ├── main.py            # FastAPI/gRPC Entry Point
│   ├── retriever.py       # Milvus & Neo4j Logic
│   ├── graph_engine.py    # LangGraph Pregel Engine: This is the brain. It implements the iterative refinement loop shown in diagrams/sequence.plantuml
│   ├── models.py          # Pydantic & Data Models
│   └── services/
│       ├── milvus_client.py # Vector Retrieval
│       └── neo4j_client.py # Graph Retrieval
├── proto/
│   └── query.proto        # gRPC Definition: Defines the contract between Spring Boot and the Query Processor.
├── requirements.txt
└── Dockerfile

```
1. Request Arrival:
    - Spring Boot sends a gRPC request StartProcessing with the user's question.
    - The QueryProcessorServicer in main.py wakes up.
2. Hybrid Retrieval (The "Gather" Phase):
    - It calls MilvusService to find text chunks that semantically match the question (e.g., "How to deploy K8s?").
    - It calls Neo4jService to find structured relationships (e.g., "K8s" -> "uses" -> "Etcd").
    - It combines these into a raw list of facts.
3. LangGraph Iteration (The "Think" Phase):
    - The AgentState is created with these facts.
    - Loop 1: The analyst node reads the facts. It might say, "We know how to install K8s, but we don't know the specific cloud provider constraints." Logic Check: Since it didn't say "CONVERGED", the graph loops back. (In a full implementation, this loop would trigger a new specific search. Here, it simulates the refinement logic).
    - Loop 2: The analyst reviews again, decides the context is now rich enough, and outputs "CONVERGED".
    - The graph exits the loop and moves to synthesizer.
3. Synthesis & Streaming (The "Speak" Phase):
    - The synthesizer node takes the refined context and asks the LLM (OpenAI) to write the final answer.
    Instead of waiting for the whole text, the Python code splits the answer into words and yields them one by one over the gRPC stream.
    - Spring Boot receives these chunks instantly and pushes them to the React UI via SSE, creating the "typewriter" effect.