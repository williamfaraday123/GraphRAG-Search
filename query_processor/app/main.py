import grpc
from concurrent import futures
import logging
import os
import traceback
from typing import Iterator

# Import generated gRPC code (generated via: grpc_tools.protoc)
import query_pb2
import query_pb2_grpc

from retriever import HybridRetriever
from graph_engine import build_query_graph

# Aliases for readability
SearchRequest = query_pb2.SearchRequest
SearchResponse = query_pb2.SearchResponse
ProcessingStatus = query_pb2.ProcessingStatus
StreamChunk = query_pb2.StreamChunk
FinalAnswer = query_pb2.FinalAnswer

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QueryProcessorServicer:
    def __init__(self):
        # config — defaults assume Docker Compose with port mapping; override in .env for custom setups
        MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
        MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
        NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
        NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password123")

        # Initialize Clients
        self.retriever = HybridRetriever(
            milvus_host=MILVUS_HOST,
            milvus_port=MILVUS_PORT,
            neo4j_uri=NEO4J_URI,
            neo4j_user=NEO4J_USER,
            neo4j_password=NEO4J_PASSWORD,
            embedding_model="all-MiniLM-L6-v2" # Must match Milvus collection dimension
        )
        self.app_graph = build_query_graph(self.retriever)

    def StartProcessing(self, request, context) -> Iterator[SearchResponse]:
        """
        gRPC Stream Handler
        1. Hybrid Retrieval
        2. LangGraph Iteration
        3. Streaming Response
        """
        try:
            prompt = request.prompt
            logger.info(f"Processing query: {prompt}")

            yield self._make_status("RETRIEVING", "Fetching vectors and graph relations...")

            initial_state = {
                "query": prompt,
                "top_k": request.top_k or 5,
                "all_chunks": [],
                "all_relations": [],
                "refined_context": [],
                "search_queries": [],
                "iteration_count": 0,
                "messages": [],
                "final_answer": "",
                "graph_scores": {}
            }

            # Stream node-by-node instead of a single blocking invoke() so each
            # agent's step (planner/retriever/analyst/synthesizer) can be surfaced
            # to the UI as it happens, not just the three coarse pipeline stages.
            state_acc = dict(initial_state)
            iterating_announced = False

            for step in self.app_graph.stream(initial_state, stream_mode="updates"):
                for node_name, update in step.items():
                    state_acc.update(update)

                    detail = update["messages"][-1] if update.get("messages") else ""
                    yield self._make_agent_step(node_name, detail)

                    if node_name == "retriever" and not iterating_announced:
                        yield self._make_status("ITERATING", "Running Pregel consensus engine...")
                        iterating_announced = True
                    elif node_name == "analyst" and update.get("search_queries"):
                        yield self._make_status("RETRIEVING", "Gap identified — running another retrieval round...")
                    elif node_name == "synthesizer":
                        yield self._make_status("SYNTHESIZING", "Generating final answer...")

            answer = state_acc.get("final_answer", "")

            # Stream the answer (Simulated streaming of the final text)
            words = answer.split()
            for word in words:
                chunk_resp = SearchResponse()
                chunk_resp.chunk.content = word + " "
                yield chunk_resp

            # Final Metadata
            final_resp = SearchResponse()
            final_resp.answer.text = answer
            final_resp.answer.sources.extend(
                set(c['source'] for c in state_acc.get("all_chunks", []) if c.get('source'))
            )
            final_resp.answer.confidence_score = 0.95
            yield final_resp

        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            logger.error(traceback.format_exc())
            context.set_details(str(e))
            context.set_code(grpc.StatusCode.INTERNAL)

    def _make_status(self, stage: str, msg: str):
        resp = SearchResponse()
        resp.status.stage = stage
        resp.status.message = msg
        return resp

    def _make_agent_step(self, agent: str, detail: str):
        resp = SearchResponse()
        resp.agent_step.agent = agent
        resp.agent_step.detail = detail
        return resp

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    query_pb2_grpc.add_QueryProcessorServiceServicer_to_server(QueryProcessorServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    logger.info("Query Processor gRPC server started on port 50051")
    server.wait_for_termination()

if __name__ == '__main__':
    serve()
