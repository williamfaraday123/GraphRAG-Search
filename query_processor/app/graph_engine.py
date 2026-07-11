from typing import TypedDict, List, Dict, Annotated, Sequence
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import networkx as nx
import logging
import operator
import os

logger = logging.getLogger(__name__)

# --- State Definition ---
class AgentState(TypedDict):
    query: str
    initial_chunks: List[dict] # From Milvus (contains 'id', 'content', 'source', 'score')
    graph_relations: List[dict] # From Neo4j (contains 'source', 'target', 'relation', 'weight')
    refined_context: List[dict] # The final, re-ranked list of chunks ready for LLM
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
    
    # 1. Add Nodes (from Milvus chunks)
    # We store the original Milvus score as a node attribute for potential hybrid scoring later
    for chunk in chunks:
        chunk_id = chunk.get('id')
        if chunk_id:
            G.add_node(chunk_id, milvus_score=chunk.get('score', 0.0), content=chunk.get('content', ''))
    
    # 2. Add Edges (from Neo4j Adjacency Matrix)
    for edge in edges:
        src = edge.get('source')
        tgt = edge.get('target')
        weight = edge.get('weight', 1.0)
        
        # Only add edge if both nodes exist in our candidate set
        if G.has_node(src) and G.has_node(tgt):
            G.add_edge(src, tgt, weight=weight)
    
    # 3. Calculate PageRank
    if len(G.nodes) == 0:
        return {}
    
    try:
        # damping=0.85 is standard. weight='weight' uses our Neo4j edge weights.
        pr_scores = nx.pagerank(G, weight='weight', max_iter=100)
        return pr_scores
    except Exception as e:
        logger.error(f"PageRank calculation failed: {e}")
        return {}

# --- Node Functions ---

def retrieve_initial_context(state: AgentState):
    """
    Step 1: Graph Construction & Re-Ranking.
    compute authority scores
    """
    chunks = state["initial_chunks"]
    edges = state["graph_relations"]

    logger.info(f"Building sub-graph with {len(chunks)} nodes and {len(edges)} edges.")

    # Calculate Authority (PageRank)
    pr_scores = calculate_authority_scores(chunks, edges)

    # Hybrid Scoring: Combine Milvus Similarity + Graph Authority
    # Formula: FinalScore = (0.6 * Norm_Milvus) + (0.4 * Norm_PageRank)
    # Note: Simple normalization for demonstration
    ranked_chunks = []
    
    # Normalize scores to 0-1 range for fair addition
    max_milvus = max([c.get('score', 0) for c in chunks]) if chunks else 1.0
    max_pr = max(pr_scores.values()) if pr_scores else 1.0
    
    for chunk in chunks:
        cid = chunk.get('id')
        milvus_s = chunk.get('score', 0.0) / max_milvus
        pr_s = pr_scores.get(cid, 0.0) / max_pr if pr_scores else 0.0
        
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
    
    # Sort by Hybrid Score (Highest first)
    ranked_chunks.sort(key=lambda x: x['hybrid_score'], reverse=True)
    
    # Take top N for context window efficiency (e.g., top 20)
    top_context = ranked_chunks[:20]
    
    return {
        "refined_context": top_context,
        "graph_scores": pr_scores,
        "iteration_count": 0,
        "messages": [f"Graph constructed. Top authoritative node: {top_context[0]['id'] if top_context else 'None'}"]
    }

def critical_analysis_node(state: AgentState):
    """Step 2: LLM analyzes gaps in the current context"""
    llm = ChatOpenAI(
        model=os.getenv("EDGE_MODEL", "qwen2.5:1.5b"),
        base_url=os.getenv("EDGE_MODEL_BASE_URL", "http://localhost:11434/v1"),
        api_key=os.getenv("LLM_API_KEY", "ollama")
    )
    
    # Format context to include *why* it was chosen (optional but helpful for debugging)
    context_items = []
    for i, c in enumerate(state["refined_context"]):
        # Include hybrid score in context so LLM knows confidence
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
    
    If sufficient, output "CONVERGED".
    Otherwise, briefly state what is missing.
    """
        
    
    response = llm.invoke([system_msg, HumanMessage(content=prompt)])
    content = response.content.strip()
    
    if "CONVERGED" in content:
        return {"messages": [f"Analysis: Converged at iteration {state['iteration_count']}"]}
    
    # In a fully dynamic system, we would trigger a new Milvus/Neo4j search here.
    # For this iteration, we just increment count to force loop exit eventually.
    return {
        "messages": [f"Analysis: Gap identified - {content}"],
        "iteration_count": state["iteration_count"] + 1
    }

def synthesize_answer(state: AgentState):
    """Step 3: Generate final response"""
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

def build_query_graph():
    workflow = StateGraph(AgentState)
    
    # Add Nodes
    workflow.add_node("retriever", retrieve_initial_context)
    workflow.add_node("analyst", critical_analysis_node)
    workflow.add_node("synthesizer", synthesize_answer)
    
    # Set Entry Point
    workflow.set_entry_point("retriever")
    
    # Define Logic: Retriever -> Analyst
    workflow.add_edge("retriever", "analyst")
    
    # Conditional Edge: Analyst loops back or goes to Synthesizer
    def should_continue(state: AgentState):
        if state["iteration_count"] < 2 and "CONVERGED" not in str(state["messages"][-1]):
            return "analyst" # Loop back
        return "synthesizer"
    
    workflow.add_conditional_edges(
        "analyst",
        should_continue,
        {
            "analyst": "analyst", # Loop
            "synthesizer": "synthesizer" # Exit
        }
    )
    
    workflow.add_edge("synthesizer", END)
    
    return workflow.compile()