from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from retriever import HybridRetriever
import networkx as nx
import logging
import os

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 2
MAX_SUB_QUERIES = 3

# --- State Definition ---
class AgentState(TypedDict):
    query: str
    top_k: int
    all_chunks: List[dict]        # Chunks accumulated across every retrieval round (deduped by id)
    all_relations: List[dict]     # Graph edges accumulated across every retrieval round (deduped)
    refined_context: List[dict]   # The current re-ranked list of chunks ready for LLM
    search_queries: List[str]     # Queries to run in the next retrieval round (set by planner or critic)
    iteration_count: int
    messages: List[str]
    final_answer: str
    graph_scores: Dict[str, float]   # Stores PageRank scores for debugging/logging

# --- Helper: Graph Analysis ---
def calculate_authority_scores(chunks: List[Dict], edges: List[Dict]) -> Dict[str, float]:
    """
    Builds a local NetworkX graph from the sub-graph data and calculates PageRank.
    This identifies 'hub' nodes that might be semantically less similar but structurally critical.
    """
    G = nx.DiGraph()

    for chunk in chunks:
        chunk_id = chunk.get('id')
        if chunk_id:
            G.add_node(chunk_id, milvus_score=chunk.get('score', 0.0), content=chunk.get('content', ''))

    for edge in edges:
        src = edge.get('source')
        tgt = edge.get('target')
        weight = edge.get('weight', 1.0)

        if G.has_node(src) and G.has_node(tgt):
            G.add_edge(src, tgt, weight=weight)

    if len(G.nodes) == 0:
        return {}

    try:
        pr_scores = nx.pagerank(G, weight='weight', max_iter=100)
        return pr_scores
    except Exception as e:
        logger.error(f"PageRank calculation failed: {e}")
        return {}

def _merge_chunks(existing: List[dict], new: List[dict]) -> List[dict]:
    merged = {c['id']: c for c in existing if c.get('id')}
    for c in new:
        cid = c.get('id')
        if cid:
            merged[cid] = c
    return list(merged.values())

def _merge_relations(existing: List[dict], new: List[dict]) -> List[dict]:
    merged = {(e.get('source'), e.get('target')): e for e in existing}
    for e in new:
        merged[(e.get('source'), e.get('target'))] = e
    return list(merged.values())

def _rank_chunks(chunks: List[dict], pr_scores: Dict[str, float]) -> List[dict]:
    ranked_chunks = []
    max_milvus = max([c.get('score', 0) for c in chunks]) if chunks else 1.0
    max_pr = max(pr_scores.values()) if pr_scores else 1.0

    for chunk in chunks:
        cid = chunk.get('id')
        milvus_s = chunk.get('score', 0.0) / max_milvus if max_milvus else 0.0
        pr_s = pr_scores.get(cid, 0.0) / max_pr if pr_scores and max_pr else 0.0

        # Hybrid Weight: 60% Semantic, 40% Structural Authority
        final_score = (0.6 * milvus_s) + (0.4 * pr_s)

        ranked_chunks.append({
            "id": cid,
            "content": chunk.get('content'),
            "source": chunk.get('source'),
            "milvus_score": milvus_s,
            "pagerank_score": pr_s,
            "hybrid_score": final_score
        })

    ranked_chunks.sort(key=lambda x: x['hybrid_score'], reverse=True)
    return ranked_chunks

# --- Node Functions ---

def plan_query_node(state: AgentState):
    """
    Step 0: Decide how to search. Simple queries are searched as-is; queries with
    multiple distinct parts get decomposed into a few focused sub-queries so the
    first retrieval round covers more ground than a single embedding lookup would.
    """
    llm = ChatOpenAI(
        model=os.getenv("EDGE_MODEL", "qwen2.5:1.5b"),
        base_url=os.getenv("EDGE_MODEL_BASE_URL", "http://localhost:11434/v1"),
        api_key=os.getenv("LLM_API_KEY", "ollama")
    )

    system_msg = SystemMessage(content="You are a Query Planning Agent for a hybrid vector + knowledge-graph search engine.")
    prompt = f"""
    User query: {state['query']}

    Decide how to search for this. If it is a single, focused question, respond with exactly:
    SIMPLE

    If it has multiple distinct parts, comparisons, or would benefit from separate searches
    (e.g. "compare X and Y", "what is A and how does it relate to B"), break it into at most
    {MAX_SUB_QUERIES} focused sub-queries, one per line, each prefixed with "SUBQUERY:".

    Respond with SIMPLE, or only the SUBQUERY lines — no other text.
    """

    response = llm.invoke([system_msg, HumanMessage(content=prompt)])
    content = response.content.strip()

    sub_queries = []
    for line in content.splitlines():
        line = line.strip()
        if line.upper().startswith("SUBQUERY:"):
            sq = line.split(":", 1)[1].strip()
            if sq:
                sub_queries.append(sq)

    sub_queries = sub_queries[:MAX_SUB_QUERIES]
    search_queries = sub_queries if sub_queries else [state["query"]]

    strategy = "decomposed" if len(search_queries) > 1 else "simple"
    return {
        "search_queries": search_queries,
        "messages": [
            f"Planner: {strategy} strategy — queued {len(search_queries)} "
            f"search quer{'y' if len(search_queries) == 1 else 'ies'}"
        ]
    }

def make_retrieve_context_node(retriever: HybridRetriever):
    """
    Builds the retrieval node. Runs the queries queued by the planner (first pass)
    or by the critic (subsequent passes) — accumulating chunks/edges across rounds
    instead of discarding prior context.
    """
    def retrieve_context(state: AgentState):
        queries = state.get("search_queries") or [state["query"]]
        logger.info(f"Retrieval round for {len(queries)} quer{'y' if len(queries) == 1 else 'ies'}: {queries}")

        new_chunks = []
        new_relations = []
        for q in queries:
            result = retriever.search(
                query=q,
                top_k=state.get("top_k") or 5,
                collection_name="ai_docs"
            )
            new_chunks.extend(result.get("chunks", []))
            new_relations.extend(result.get("relations", []))

        all_chunks = _merge_chunks(state.get("all_chunks", []), new_chunks)
        all_relations = _merge_relations(state.get("all_relations", []), new_relations)

        pr_scores = calculate_authority_scores(all_chunks, all_relations)
        ranked_chunks = _rank_chunks(all_chunks, pr_scores)
        top_context = ranked_chunks[:20]

        return {
            "all_chunks": all_chunks,
            "all_relations": all_relations,
            "refined_context": top_context,
            "graph_scores": pr_scores,
            "search_queries": [],  # consumed — cleared until the critic sets new ones
            "messages": [
                f"Retrieved {len(new_chunks)} new chunk(s) across {len(queries)} "
                f"quer{'y' if len(queries) == 1 else 'ies'}. "
                f"Top authoritative node: {top_context[0]['id'] if top_context else 'None'}"
            ]
        }
    return retrieve_context

def critical_analysis_node(state: AgentState):
    """Step 2: LLM analyzes gaps in the current context and, if insufficient,
    proposes a refined follow-up query for another retrieval round."""

    if not state["refined_context"]:
        return {
            "messages": ["Analysis: No context retrieved — nothing to refine further."],
            "search_queries": []
        }

    llm = ChatOpenAI(
        model=os.getenv("EDGE_MODEL", "qwen2.5:1.5b"),
        base_url=os.getenv("EDGE_MODEL_BASE_URL", "http://localhost:11434/v1"),
        api_key=os.getenv("LLM_API_KEY", "ollama")
    )

    context_items = []
    for i, c in enumerate(state["refined_context"]):
        context_items.append(
            f"[Source {i+1} (Score: {c['hybrid_score']:.2f})]: {c['content']}"
        )
    context_text = "\n\n".join(context_items)

    system_msg = SystemMessage(content="You are a Research Analyst analyzing a knowledge sub-graph.")
    prompt = f"""
    Query: {state['query']}

    The following context has been retrieved and re-ranked using both semantic similarity
    and graph connectivity (PageRank).

    Current Context:
    {context_text}

    Task:
    1. Does this context sufficiently answer the query?
    2. Are there contradictions between highly connected nodes?

    If sufficient, respond with exactly: CONVERGED
    Otherwise, respond with a short explanation of the gap, followed by a new line starting with
    "REFINED_QUERY:" containing a more specific search query that would help fill that gap.
    """

    response = llm.invoke([system_msg, HumanMessage(content=prompt)])
    content = response.content.strip()

    if "CONVERGED" in content:
        return {
            "messages": [f"Analysis: Converged at iteration {state['iteration_count']}"],
            "search_queries": []
        }

    refined_query = ""
    for line in content.splitlines():
        if line.strip().upper().startswith("REFINED_QUERY:"):
            refined_query = line.split(":", 1)[1].strip()
            break

    # If the model didn't follow the format, fall back to the original query so we
    # still get a second retrieval pass instead of silently dropping the gap.
    refined_query = refined_query or state["query"]

    return {
        "messages": [f"Analysis: Gap identified - {content}"],
        "iteration_count": state["iteration_count"] + 1,
        "search_queries": [refined_query]
    }

def synthesize_answer(state: AgentState):
    """Step 3: Generate final response"""
    if not state["refined_context"]:
        return {"final_answer": "I couldn't find relevant information in the knowledge base to answer this query."}

    llm = ChatOpenAI(
        model=os.getenv("EDGE_MODEL", "qwen2.5:1.5b"),
        base_url=os.getenv("EDGE_MODEL_BASE_URL", "http://localhost:11434/v1"),
        api_key=os.getenv("LLM_API_KEY", "ollama")
    )

    context_text = "\n\n".join([
        f"Source: {c.get('source', 'Unknown')}\nContent: {c.get('content')}"
        for c in state["refined_context"]
    ])

    prompt = f"""
    Based on the following verified and authority-ranked context, answer the user query.
    Prioritize information from sources with higher structural importance (connectivity).
    Always cite the Source URL/ID.

    Query: {state['query']}

    Context:
    {context_text}
    """

    response = llm.invoke([HumanMessage(content=prompt)])
    return {"final_answer": response.content}

# --- Graph Construction ---

def build_query_graph(retriever: HybridRetriever):
    workflow = StateGraph(AgentState)

    workflow.add_node("planner", plan_query_node)
    workflow.add_node("retriever", make_retrieve_context_node(retriever))
    workflow.add_node("analyst", critical_analysis_node)
    workflow.add_node("synthesizer", synthesize_answer)

    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "retriever")
    workflow.add_edge("retriever", "analyst")

    def should_continue(state: AgentState):
        if state["iteration_count"] < MAX_ITERATIONS and state.get("search_queries"):
            return "retriever"  # Loop back for another retrieval + analysis round
        return "synthesizer"

    workflow.add_conditional_edges(
        "analyst",
        should_continue,
        {
            "retriever": "retriever",
            "synthesizer": "synthesizer"
        }
    )

    workflow.add_edge("synthesizer", END)

    return workflow.compile()
