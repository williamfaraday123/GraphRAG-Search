import grpc
from concurrent import futures
import logging
from typing import Iterator

# Import generated gRPC code (assuming you ran grpc_tools.protoc)
# import query_pb2
# import query_pb2_grpc

from app.retriever import HybridRetriever # Wrapper for Milvus/Neo4j
from app.graph_engine import build_query_graph

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mocking the proto classes for this snippet representation
# In real code: from query_pb2 import SearchRequest, SearchResponse, ProcessingStatus, etc.
class SearchRequest: pass
class SearchResponse: pass
class ProcessingStatus: pass
class StreamChunk: pass
class FinalAnswer: pass

class QueryProcessorServicer:
    def __init__(self):
        # config
        MILVUS_HOST = os.getenv("MILVUS_HOST", "milvus-standalone")
        MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
        NEO4J_URI = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
        NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
        NEO4J_PASS = os.getenv("NEO4J_PASS", "password123")

        # Initialize Clients
        self.retriever = HybridRetriever(
            milvus_host=MILVUS_HOST,
            milvus_port=MILVUS_PORT,
            neo4j_uri=NEO4J_URI,
            neo4j_user=NEO4J_USER,
            neo4j_password=NEO4J_PASS,
            embedding_model="all-MiniLM-L6-v2" # Must match Milvus collection dimension
        )
        self.app_graph = build_query_graph()

    def StartProcessing(self, request, context) -> Iterator[SearchResponse]:
        """
        gRPC Stream Handler
        1. Hybrid Retrieval
        2. LangGraph Iteration
        3. Streaming Response
        """
        prompt = request.prompt
        logger.info(f"Processing query: {prompt}")

        # Yield Status: Retrieving
        yield self._make_status("RETRIEVING", "Fetching vectors and graph relations...")

        # 1. Hybrid Retrieval
        
        # This single call handles: Embedding -> Milvus Search -> Neo4j Search
        retrieval_result = self.retriever.search(
            query=prompt,
            top_k=request.top_k,
            collection_name="ai_docs" # Ensure this matches your Milvus collection name
        )

        if "error" in retrieval_result:
            logger.error(f"Retrieval failed: {retrieval_result['error']}")
            context.set_details(retrieval_result['error'])
            context.set_code(grpc.StatusCode.INTERNAL)
            return

        chunks = retrieval_result['chunks']
        relations = retrieval_result['relations']

        logger.info(f"Retrieved {len(chunks)} vectors and {len(relations)} graph nodes")
        # Yield Status: Iterating
        yield self._make_status("ITERATING", "Running Pregel consensus engine...")

        # 2. LangGraph Execution
        initial_state = {
            "query": prompt,
            "initial_chunks": chunks,
            "graph_relations": relations,
            "refined_context": [],
            "iteration_count": 0,
            "messages": [],
            "final_answer": "",
            "graph_scores": {}
        }

        try:
            # Run the graph
            final_state = self.app_graph.invoke(initial_state)
            
            # Yield Status: Synthesizing
            yield self._make_status("SYNTHESIZING", "Generating final answer...")

            # 3. Stream the answer (Simulated streaming of the final text)
            answer = final_state["final_answer"]
            words = answer.split()
            
            for word in words:
                chunk_resp = SearchResponse()
                chunk_resp.chunk.content = word + " "
                # Add source metadata if available
                yield chunk_resp
            
            # Final Metadata
            final_resp = SearchResponse()
            final_resp.answer.text = answer
            final_resp.answer.sources = list(set([c['source'] for c in chunks if 'source' in c and c['source']]))
            final_resp.answer.confidence_score = 0.95
            yield final_resp

        except Exception as e:
            logger.error(f"Graph execution failed: {e}")
            context.set_details(str(e))
            context.set_code(grpc.StatusCode.INTERNAL)

    def _make_status(self, stage: str, msg: str):
        resp = SearchResponse()
        resp.status.stage = stage
        resp.status.message = msg
        return resp

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    # query_pb2_grpc.add_QueryProcessorServiceServicer_to_server(QueryProcessorServicer(), server)
    # For this snippet, we assume the registration happens
    server.add_insecure_port('[::]:50051')
    server.start()
    logger.info("Query Processor gRPC server started on port 50051")
    server.wait_for_termination()

if __name__ == '__main__':
    serve()