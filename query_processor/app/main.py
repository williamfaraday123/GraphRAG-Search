import grpc
from concurrent import futures
import json
import logging
import os
import re
import time
import traceback
from typing import Iterator

# Import generated gRPC code (generated via: grpc_tools.protoc)
import query_pb2
import query_pb2_grpc

from retriever import HybridRetriever
from graph_engine import ANSWER_TOKEN, MAX_ITERATIONS, answer_paths, build_query_graph, similarity_paths
from corpus import normalize_corpus_id, normalize_file_id
from conversation import clean_history
from corpus_indexer import CorpusIndexer
from knowledge_graph import KnowledgeGraphBuilder
from entity_resolution import EntityResolver

# Aliases for readability
SearchRequest = query_pb2.SearchRequest
SearchResponse = query_pb2.SearchResponse
ProcessingStatus = query_pb2.ProcessingStatus
StreamChunk = query_pb2.StreamChunk
FinalAnswer = query_pb2.FinalAnswer

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_EVIDENCE_CHUNKS = 8

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
        self.corpus_indexer = CorpusIndexer(self.retriever.embedding_service)
        # Background LLM extraction of entities/facts from uploads; also resumes any
        # files left unprocessed by a previous run
        # Entities naming the same concept are written onto one node (entity_resolution.py)
        self.entity_resolver = EntityResolver(
            self.corpus_indexer.neo4j_driver, self.retriever.embedding_service.generate_batch_embeddings
        )
        self.kg_builder = KnowledgeGraphBuilder(
            self.corpus_indexer.neo4j_driver, resolver=self.entity_resolver,
            on_file_done=self.corpus_indexer.refresh_vectors,
        )

    def StartProcessing(self, request, context) -> Iterator[SearchResponse]:
        # Knowledge-graph extraction pauses between chunks while a search is running,
        # so both don't compete for the same local model
        self.kg_builder.search_started()
        try:
            yield from self._run_search(request, context)
        finally:
            self.kg_builder.search_finished()

    def _run_search(self, request, context) -> Iterator[SearchResponse]:
        """
        gRPC Stream Handler
        1. Hybrid Retrieval
        2. LangGraph Iteration
        3. Streaming Response
        """
        try:
            prompt = request.prompt
            corpus_id = normalize_corpus_id(request.corpus_id)
            file_ids = [normalize_file_id(f) for f in request.file_ids]
            history = clean_history(request.history)
            logger.info(f"Processing query: {prompt} (files: {file_ids or 'all'}, "
                        f"earlier turns: {len(history)})")

            yield self._make_status("PLANNING", "Reading the follow-up against the earlier questions..."
                                    if history else "Deciding how to search for this question...")

            initial_state = {
                "query": prompt,
                "original_query": prompt,
                "history": history,
                "carried_seeds": {},
                "top_k": request.top_k or 5,
                "corpus_id": corpus_id,
                "file_ids": file_ids,
                "all_chunks": [],
                "all_relations": [],
                "refined_context": [],
                "search_queries": [],
                "iteration_count": 0,
                "messages": [],
                "final_answer": "",
                "graph_scores": {},
                "kg_paths": [],
                "kg_seeds": [],
                "question_entities": [],
                "file_similarity": None,
            }

            # Stream node-by-node instead of a single blocking invoke() so each
            # agent's step (planner/retriever/analyst/synthesizer) can be surfaced
            # to the UI as it happens, not just the three coarse pipeline stages.
            state_acc = dict(initial_state)
            # A node's update only arrives once it has finished, so each status names the step
            # that starts next; each agent step reports how long its step took
            timings = []
            step_started = time.perf_counter()

            # "custom" carries the answer text as the model writes it, so the UI shows the answer
            # while it is being written instead of after the whole answer is done
            streamed = False
            for mode, step in self.app_graph.stream(initial_state, stream_mode=["updates", "custom"]):
                if mode == "custom":
                    if step.get(ANSWER_TOKEN):
                        streamed = True
                        yield self._make_chunk(step[ANSWER_TOKEN])
                    continue
                for node_name, update in step.items():
                    state_acc.update(update)
                    elapsed = time.perf_counter() - step_started
                    timings.append(f"{node_name} {elapsed:.1f}s")

                    detail = update["messages"][-1] if update.get("messages") else ""
                    if node_name == "synthesizer":
                        detail = detail or "Answer written"
                    yield self._make_agent_step(node_name, f"{detail} ({elapsed:.1f} s)")

                    if node_name == "rewriter":
                        yield self._make_status("PLANNING", "Deciding how to search for this question...")
                    elif node_name == "planner":
                        yield self._make_status("RETRIEVING", "Searching vectors and the knowledge graph...")
                    elif node_name == "retriever":
                        yield self._make_status("ANALYZING", "Checking whether the context answers the question...")
                    elif node_name == "analyst":
                        if state_acc["iteration_count"] < MAX_ITERATIONS and state_acc.get("search_queries"):
                            yield self._make_status("RETRIEVING", "Gap identified — running another retrieval round...")
                        else:
                            yield self._make_status(
                                "SYNTHESIZING", f"Writing the answer with {os.getenv('ANSWER_MODEL', 'the answer model')}...")
                    step_started = time.perf_counter()

            logger.info("Search step timings: %s", ", ".join(timings))
            answer = state_acc.get("final_answer", "")

            # Answers not written by the model (e.g. "nothing found") weren't streamed: send them
            # word by word. Each piece keeps the whitespace after it, so line breaks survive.
            if not streamed:
                for piece in re.findall(r"\s*\S+\s*", answer.strip()):
                    yield self._make_chunk(piece)

            # Final Metadata
            final_resp = SearchResponse()
            final_resp.answer.text = answer
            final_resp.answer.sources.extend(
                set(c['source'] for c in state_acc.get("all_chunks", []) if c.get('source'))
            )
            final_resp.answer.confidence_score = 0.95

            # Ranked evidence chunks for the UI (already sorted by hybrid_score, highest first)
            for c in state_acc.get("refined_context", [])[:MAX_EVIDENCE_CHUNKS]:
                chunk_info = final_resp.answer.chunks.add()
                chunk_info.chunk_id = c.get("id") or ""
                chunk_info.content = c.get("content") or ""
                chunk_info.source = c.get("source") or ""
                chunk_info.milvus_score = c.get("milvus_score", 0.0)
                chunk_info.pagerank_score = c.get("pagerank_score", 0.0)
                chunk_info.hybrid_score = c.get("hybrid_score", 0.0)
                chunk_info.graph_score = c.get("graph_score", 0.0)
                chunk_info.found_by_graph = bool(c.get("found_by_graph"))

            # Knowledge-graph paths the answer model was given (their facts' chunks are in its
            # context); for file-similarity questions, the facts the two chosen files share
            paths = similarity_paths(state_acc) if state_acc.get("file_similarity") else answer_paths(state_acc)
            for path in paths:
                path_msg = final_resp.answer.graph_paths.add()
                path_msg.score = path["score"]
                path_msg.kind = path.get("kind", "reasoning")
                for f in path["facts"]:
                    fact = path_msg.facts.add()
                    for field in ("source", "source_label", "relation", "target", "target_label",
                                  "chunk_id", "file_id", "filename", "evidence"):
                        setattr(fact, field, f.get(field) or "")
            final_resp.answer.graph_seeds.extend(state_acc.get("kg_seeds") or [])
            # The client sends these back with the next question (conversation.py)
            final_resp.answer.entity_keys.extend(state_acc.get("question_entities") or [])
            final_resp.answer.standalone_query = state_acc.get("query") or prompt

            yield final_resp

        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            logger.error(traceback.format_exc())
            context.set_details(str(e))
            context.set_code(grpc.StatusCode.INTERNAL)

    def GetChunkSource(self, request, context):
        """
        Unary RPC: fetches the full source document for one chunk from object
        storage, so the UI can offer "view full source" without MinIO ever
        being exposed to the browser directly.
        """
        resp = query_pb2.ChunkSourceResponse()
        try:
            content = self.retriever.object_storage.fetch_file_content(request.source)
            resp.found = True
            resp.content = content
        except Exception as e:
            logger.warning(f"GetChunkSource failed for '{request.source}': {e}")
            resp.found = False
            resp.error = str(e)
        return resp

    def UploadCorpusFile(self, request, context):
        response = query_pb2.UploadCorpusFileResponse()
        try:
            result = self.corpus_indexer.index_file(
                request.corpus_id,
                request.filename,
                request.content,
            )
            response.success = True
            response.file_id = result["file_id"]
            response.filename = result["filename"]
            response.chunk_count = result["chunk_count"]
            response.message = "File indexed and ready to search"
            self.kg_builder.enqueue_file(normalize_corpus_id(request.corpus_id), result["file_id"])
        except ValueError as e:
            response.message = str(e)
        except Exception as e:
            logger.exception("Upload indexing failed")
            response.message = f"Failed to index uploaded file: {e}"
        return response

    def ListCorpusFiles(self, request, context):
        response = query_pb2.ListCorpusFilesResponse()
        try:
            for file in self.corpus_indexer.list_files(request.corpus_id):
                item = response.files.add()
                item.file_id = file["file_id"]
                item.filename = file["filename"]
        except Exception as e:
            logger.exception("Could not list corpus files")
            context.set_details(str(e))
            context.set_code(grpc.StatusCode.INTERNAL)
        return response

    def GetCorpusFile(self, request, context):
        response = query_pb2.GetCorpusFileResponse()
        try:
            file = self.corpus_indexer.get_file(request.corpus_id, request.file_id)
            response.found = True
            response.filename = file["filename"]
            response.content = file["content"]
        except (ValueError, FileNotFoundError) as e:
            response.error = str(e)
        except Exception as e:
            logger.exception("Could not read corpus file")
            response.error = f"Could not read file: {e}"
        return response

    def DeleteCorpusFile(self, request, context):
        response = query_pb2.DeleteCorpusFileResponse()
        try:
            filename = self.corpus_indexer.delete_file(request.corpus_id, request.file_id)
            # Some of its entities may be gone; rebuild the resolver's view of the workspace
            self.entity_resolver.forget(normalize_corpus_id(request.corpus_id))
            response.success = True
            response.message = f"Deleted {filename}"
        except (ValueError, FileNotFoundError) as e:
            response.message = str(e)
        except Exception as e:
            logger.exception("Could not delete corpus file")
            response.message = f"Could not delete file: {e}"
        return response

    def GetCorpusMap(self, request, context):
        response = query_pb2.CorpusMapResponse()
        try:
            corpus_map = self.corpus_indexer.corpus_map(
                request.corpus_id, list(request.file_ids), kg_builder=self.kg_builder
            )
            response.success = True
            response.map_json = json.dumps(corpus_map, ensure_ascii=False)
        except (ValueError, FileNotFoundError) as e:
            response.error = str(e)
        except Exception as e:
            logger.exception("Could not build corpus map")
            response.error = f"Could not build corpus map: {e}"
        return response

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

    def _make_chunk(self, text: str):
        resp = SearchResponse()
        resp.chunk.content = text
        return resp

def serve():
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=[("grpc.max_receive_message_length", 6 * 1024 * 1024)],
    )
    query_pb2_grpc.add_QueryProcessorServiceServicer_to_server(QueryProcessorServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    logger.info("Query Processor gRPC server started on port 50051")
    server.wait_for_termination()

if __name__ == '__main__':
    serve()
