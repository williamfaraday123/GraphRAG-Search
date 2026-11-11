from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph, END
from langgraph.config import get_stream_writer
from langchain_core.messages import HumanMessage, SystemMessage
from llm_client import chat_client
from retriever import HybridRetriever
from document_text import display_name
from file_similarity import similarity_table
from graph_retrieval import fact_text
import networkx as nx
import logging
import os
import re

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 2
MAX_SUB_QUERIES = 3

# Key of the answer text the synthesizer streams out while the model writes it
# (graph.stream(..., stream_mode=["updates", "custom"]) yields ("custom", {ANSWER_TOKEN: text}))
ANSWER_TOKEN = "answer_token"


def _answer_writer():
    """LangGraph's writer for the "custom" stream mode, or None outside a graph run
    (e.g. a test calling a node directly)."""
    try:
        return get_stream_writer()
    except RuntimeError:
        return None


def _write_answer(llm, messages, writer) -> str:
    """Runs the answer model, passing its text to `writer` piece by piece as it is written,
    so the UI can show the answer while the model is still writing it."""
    if writer is None:
        return llm.invoke(messages).content
    parts = []
    for piece in llm.stream(messages):
        parts.append(piece)
        writer({ANSWER_TOKEN: piece})
    return "".join(parts)

# --- State Definition ---
class AgentState(TypedDict):
    query: str
    top_k: int
    corpus_id: str
    file_ids: List[str]           # Upload workspaces: restrict retrieval to these files (empty = all)
    all_chunks: List[dict]        # Chunks accumulated across every retrieval round (deduped by id)
    all_relations: List[dict]     # Graph edges accumulated across every retrieval round (deduped)
    refined_context: List[dict]   # The current re-ranked list of chunks ready for LLM
    search_queries: List[str]     # Queries to run in the next retrieval round (set by planner or critic)
    iteration_count: int
    messages: List[str]
    final_answer: str
    graph_scores: Dict[str, float]   # Stores PageRank scores for debugging/logging
    kg_paths: List[dict]          # Knowledge-graph fact paths found across rounds (best first)
    kg_seeds: List[str]           # Entities named in the queries that the walk started from
    question_entities: List[str]  # Entity keys the question (first-round queries) names; coverage check
    file_similarity: dict         # "Which files are most similar?": file_similarity.py result, else None

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
        if not cid:
            continue
        old = merged.get(cid)
        if old:
            # Same chunk from another round/query: keep its best scores; it counts as
            # found by vector search if any round found it that way
            c = {**c,
                 'score': max(old.get('score', 0.0), c.get('score', 0.0)),
                 'kg_score': max(old.get('kg_score', 0.0), c.get('kg_score', 0.0)),
                 'found_by_graph': old.get('found_by_graph', False) and c.get('found_by_graph', False)}
            if 'ppr_score' in old or 'ppr_score' in c:
                c['ppr_score'] = max(old.get('ppr_score', 0.0), c.get('ppr_score', 0.0))
        merged[cid] = c
    return list(merged.values())


def _merge_paths(existing: List[dict], new: List[dict]) -> List[dict]:
    merged = {}
    for path in existing + new:
        key = tuple((f['source'], f['relation'], f['target'], f['chunk_id']) for f in path['facts'])
        if key not in merged or path['score'] > merged[key]['score']:
            merged[key] = path
    return sorted(merged.values(), key=lambda p: -p['score'])

def _merge_relations(existing: List[dict], new: List[dict]) -> List[dict]:
    merged = {(e.get('source'), e.get('target')): e for e in existing}
    for e in new:
        merged[(e.get('source'), e.get('target'))] = e
    return list(merged.values())

def _rank_chunks(chunks: List[dict], pr_scores: Dict[str, float]) -> List[dict]:
    ranked_chunks = []
    max_milvus = max([c.get('score', 0) for c in chunks]) if chunks else 1.0
    max_pr = max(pr_scores.values()) if pr_scores else 1.0
    max_kg = max([c.get('kg_score', 0.0) for c in chunks], default=0.0)

    for chunk in chunks:
        cid = chunk.get('id')
        milvus_s = chunk.get('score', 0.0) / max_milvus if max_milvus else 0.0
        pr_s = pr_scores.get(cid, 0.0) / max_pr if pr_scores and max_pr else 0.0
        kg_s = chunk.get('kg_score', 0.0) / max_kg if max_kg else 0.0

        if max_kg:
            # Knowledge-graph paths found: 50% semantic, 20% authority, 30% fact-path relevance
            final_score = (0.5 * milvus_s) + (0.2 * pr_s) + (0.3 * kg_s)
        else:
            # Hybrid Weight: 60% Semantic, 40% Structural Authority
            final_score = (0.6 * milvus_s) + (0.4 * pr_s)

        ranked_chunks.append({
            "id": cid,
            "content": chunk.get('content'),
            "source": chunk.get('source'),
            "milvus_score": milvus_s,
            "pagerank_score": pr_s,
            "graph_score": kg_s,
            "found_by_graph": bool(chunk.get('found_by_graph')),
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
    llm = chat_client("EDGE_MODEL")

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
        new_paths = []
        ppr_visited = []
        seeds = list(state.get("kg_seeds") or [])
        # The question's own entities come from the first round (the question or the planner's
        # sub-queries); follow-up searches name extra entities that aren't required
        first_round = not state.get("iteration_count")
        question_entities = list(state.get("question_entities") or [])
        for q in queries:
            result = retriever.search(
                query=q,
                top_k=state.get("top_k") or 5,
                corpus_id=state.get("corpus_id") or "wiki",
                file_ids=state.get("file_ids") or None,
            )
            new_chunks.extend(result.get("chunks", []))
            new_relations.extend(result.get("relations", []))
            new_paths.extend(result.get("kg_paths", []))
            seeds += [s for s in result.get("kg_seeds", []) if s not in seeds]
            if result.get("ppr_nodes") is not None:
                ppr_visited.append(result["ppr_nodes"])
            if first_round:
                question_entities += [k for k in result.get("named_keys", []) if k not in question_entities]

        all_chunks = _merge_chunks(state.get("all_chunks", []), new_chunks)
        all_relations = _merge_relations(state.get("all_relations", []), new_relations)
        kg_paths = _merge_paths(state.get("kg_paths") or [], new_paths)
        # After merging: a chunk one sub-query reached through the graph may have been found
        # by vector search in another, and then it doesn't count as graph-only
        round_ids = {c.get("id") for c in new_chunks}
        graph_only = sum(1 for c in all_chunks if c.get("id") in round_ids and c.get("found_by_graph"))
        kg_note = ""
        if new_paths:
            kg_note = (
                f" Knowledge graph: followed {len(new_paths)} fact path(s)"
                + (f" from {', '.join(seeds)}" if seeds else "")
                + (f", adding {graph_only} chunk(s) vector search missed." if graph_only else ".")
            )

        corpus_id = state.get("corpus_id") or "wiki"
        if corpus_id != "wiki" and all_chunks and all("ppr_score" in c for c in all_chunks):
            # Upload workspaces: local Personalized PageRank over the whole workspace graph
            pr_scores = {c["id"]: c["ppr_score"] for c in all_chunks if c.get("id")}
            if ppr_visited:
                kg_note += f" Personalized PageRank reached {max(ppr_visited)} graph node(s)."
        else:
            pr_scores = calculate_authority_scores(all_chunks, all_relations)
        ranked_chunks = _rank_chunks(all_chunks, pr_scores)
        top_context = ranked_chunks[:20]

        # File-similarity questions: compare the files' knowledge graphs once (first round)
        similarity = state.get("file_similarity")
        similarity_note = ""
        if similarity is None and corpus_id != "wiki" and is_file_comparison(state["query"]):
            try:
                similarity = retriever.file_similarity.compare(corpus_id, state.get("file_ids") or None)
            except Exception:
                logger.exception("File similarity failed")
            if similarity:
                similarity_note = (f" Compared {len(similarity['files'])} files' knowledge graphs "
                                   f"({len(similarity['pairs'])} pairs).")

        return {
            "file_similarity": similarity,
            "all_chunks": all_chunks,
            "all_relations": all_relations,
            "refined_context": top_context,
            "graph_scores": pr_scores,
            "kg_paths": kg_paths,
            "kg_seeds": seeds,
            "question_entities": question_entities,
            "search_queries": [],  # consumed — cleared until the critic sets new ones
            "messages": [
                f"Retrieved {len(new_chunks)} new chunk(s) across {len(queries)} "
                f"quer{'y' if len(queries) == 1 else 'ies'}. "
                f"Top authoritative node: {top_context[0]['id'] if top_context else 'None'}"
                + kg_note + similarity_note
            ]
        }
    return retrieve_context

def graph_coverage_analysis(state: AgentState, coverage):
    """Coverage check for upload workspaces whose question names entities in the knowledge
    graph (coverage.py): no model call. Returns None when it can't decide, so the model does."""
    corpus_id = state.get("corpus_id") or "wiki"
    keys = state.get("question_entities") or []
    if corpus_id == "wiki" or not keys or coverage is None:
        return None
    file_ids = state.get("file_ids") or None
    try:
        context_ids = [c["id"] for c in state["refined_context"] if c.get("id")]
        missing = coverage.uncovered(corpus_id, file_ids, keys, context_ids)
        if not missing:
            return {
                "messages": [f"Analysis: Converged at iteration {state['iteration_count']} — the context covers "
                             f"every entity the question names that the searched files mention "
                             f"({len(keys)} named; graph coverage check)"],
                "search_queries": [],
            }
        if state["iteration_count"] >= MAX_ITERATIONS:
            return {"messages": [f"Analysis: {', '.join(e['name'] for e in missing)} still not in the context; "
                                 "search limit reached"],
                    "search_queries": []}
        queries = [coverage.follow_up_query(corpus_id, file_ids, e) for e in missing[:MAX_SUB_QUERIES]]
    except Exception:
        logger.exception("Graph coverage check failed; asking the model instead")
        return None
    return {
        "messages": [f"Analysis: Gap identified - the context has nothing on "
                     f"{', '.join(e['name'] for e in missing)} (graph coverage check). "
                     f"REFINED_QUERY: {' | '.join(queries)}"],
        "iteration_count": state["iteration_count"] + 1,
        "search_queries": queries,
    }


def make_critical_analysis_node(retriever: HybridRetriever = None):
    coverage = getattr(retriever, "coverage", None)

    def analysis(state: AgentState):
        return critical_analysis_node(state, coverage)
    return analysis


def critical_analysis_node(state: AgentState, coverage=None):
    """Step 2: decides whether the context is enough or another retrieval round is needed.
    Upload workspaces whose question names known entities use the graph coverage check;
    otherwise the model analyzes gaps and proposes a refined follow-up query."""

    if not state["refined_context"]:
        return {
            "messages": ["Analysis: No context retrieved — nothing to refine further."],
            "search_queries": []
        }

    if state.get("file_similarity"):
        # Answered from the files' knowledge-graph comparison; more chunks wouldn't change it
        return {"messages": ["Analysis: Converged — file comparison uses the knowledge graph"],
                "search_queries": []}

    decided = graph_coverage_analysis(state, coverage)
    if decided is not None:
        return decided

    llm = chat_client("EDGE_MODEL")

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

def _chunk_position(chunk: dict):
    """Chunk index within its file, parsed from upload ids like '<file_id>:<index>'.
    Wiki chunk ids are hashes, so those return None and keep their ranked order."""
    tail = str(chunk.get("id") or "").rsplit(":", 1)[-1]
    return int(tail) if tail.isdigit() else None


def _format_context_by_file(chunks: List[dict]) -> str:
    """
    Groups retrieved chunks into one section per source file, in document order.
    Listing each chunk as its own 'Source:' block made the model count chunks as
    separate sources (e.g. 4 sources for 2 uploaded files) and mix up which file
    said what. Files are ordered by their best-ranked chunk.
    """
    files: Dict[str, List[dict]] = {}
    for c in chunks:  # already sorted by hybrid score, so dict order = best file first
        files.setdefault(c.get("source") or "Unknown", []).append(c)

    sections = []
    for n, (source, file_chunks) in enumerate(files.items(), start=1):
        if all(_chunk_position(c) is not None for c in file_chunks):
            file_chunks = sorted(file_chunks, key=_chunk_position)
        # s3://bucket/uploads/<corpus>/<file_id>/report.pdf.txt -> report.pdf
        name = display_name(source.rsplit("/", 1)[-1])
        body = "\n[...]\n".join((c.get("content") or "").strip() for c in file_chunks)
        sections.append(f"=== File {n}: {name} ===\n{body}")
    return "\n\n".join(sections)


def _fact_key(f):
    return (f["source"], f["relation"], f["target"], f["chunk_id"])


def answer_paths(state, limit: int = 6) -> List[dict]:
    """Knowledge-graph paths whose every fact comes from a chunk in the answer's context.
    A path whose facts all appear in another path is dropped (sub-queries each walk the
    graph, so 'A overtook B' alone often repeats inside 'A overtook B, then C backed A')."""
    in_context = {c["id"] for c in state.get("refined_context") or []}
    paths = [p for p in state.get("kg_paths") or [] if all(f["chunk_id"] in in_context for f in p["facts"])]
    fact_sets = [frozenset(_fact_key(f) for f in p["facts"]) for p in paths]
    kept = [p for p, facts in zip(paths, fact_sets) if not any(facts < other for other in fact_sets)]
    return kept[:limit]


def format_kg_facts(paths: List[dict]) -> str:
    lines = []
    for p in paths:
        steps = [f"{f['source_label']} {f['relation']} {f['target_label']} [{f['filename']}]" for f in p["facts"]]
        lines.append("- " + " → then: ".join(steps))
    return "\n".join(lines)


# "1. capital  2. narratives" or "A) capital B) narratives": an enumerated marker at the start
# or after whitespace, followed by the option text
_OPTION_MARKER = re.compile(r"(?:^|(?<=\s))\(?([1-9][0-9]?|[A-H])[.)]\s+")
_ALL_FILES = re.compile(r"\b(both|all|each|every)\s+(?:of\s+the\s+)?(files|sources|documents|texts|passages)\b", re.I)


def listed_options(query: str) -> List[tuple]:
    """[(marker, text)] for answer options listed in the query, numbered 1, 2, ... or
    lettered A, B, ...; empty when the query lists fewer than two options."""
    found = list(_OPTION_MARKER.finditer(query))
    for start in range(len(found)):
        first = found[start].group(1)
        if first not in ("1", "A"):
            continue
        run = [found[start]]
        for m in found[start + 1:]:
            prev = run[-1].group(1)
            expected = str(int(prev) + 1) if prev.isdigit() else chr(ord(prev) + 1)
            if m.group(1) == expected:
                run.append(m)
        if len(run) >= 2:
            ends = [m.start() for m in run[1:]] + [len(query)]
            return [(m.group(1), query[m.end():end].strip(" .;,")) for m, end in zip(run, ends)]
    return []


_COUNT_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
_COUNT_EXPLICIT = re.compile(r"\b(?:which|choose|select|pick|name|give)\s+(one|two|three|four|five|[1-5])\b", re.I)
# "Which reason ...", "Which option is ..." (singular noun right after which/what) asks for one
_COUNT_SINGULAR = re.compile(
    r"\b(?:which|what)\s+(?:\w+\s+)?(?:reason|option|statement|factor|cause|answer|one|explanation|example)\b(?!s)", re.I)


def answer_count(query: str):
    """How many options the query asks for (e.g. 'Which TWO ...' -> 2, 'Which reason ...' -> 1),
    or None when it doesn't say."""
    question = query[: _OPTION_MARKER.search(query).start()] if _OPTION_MARKER.search(query) else query
    m = _COUNT_EXPLICIT.search(question)
    if m:
        word = m.group(1).lower()
        return int(word) if word.isdigit() else _COUNT_WORDS[word]
    return 1 if _COUNT_SINGULAR.search(question) else None


_COMMON_RULES = """Answer the query using only the context below. The context is grouped by file under
headings like "=== File 1: report.pdf ==="; cite a file by its name only, e.g. [report.pdf]
(not "File 1"). Treat each file as one source. Reply in Markdown with exactly the two sections
shown, starting each with its ### heading, in order, and nothing before or after them."""


def _options_prompt(query, context_text, options, files) -> str:
    """Multiple choice: check every option against a quoted sentence first (stops the model
    picking distractors the text attributes to a different subject: 0/8 correct without it,
    6/6 with), then answer. The UI shows the answer first and the per-option breakdown below it
    (web_search_client/src/Markdown.js parseOptionChecks)."""
    option_list = "\n".join(f"{marker}. {text}" for marker, text in options)
    # The examples are deliberately about another topic: small models copy examples taken
    # from the task itself instead of reasoning. Evidence comes before the verdict in each
    # bullet so the verdict is decided from the quotes, not guessed first.
    subject_rule = (
        "A sentence counts only if it states the option as an answer to the query, about the subject the "
        "query asks about: not about a different country, person or group mentioned nearby, not merely "
        "sharing some words with the option, and not a related but different idea stretched to fit."
    )
    count = answer_count(query)
    choose = (
        f"Choose exactly {count} option{'s' if count > 1 else ''}, as the query asks; if more are SUPPORTED, "
        "choose the one(s) the context states most directly."
        if count else "Choose only SUPPORTED options, matching the number of answers the query asks for."
    )
    if _ALL_FILES.search(query) and len(files) > 1:
        rule = (
            f"{subject_rule} The query asks what is supported in EVERY file ({', '.join(files)}). In each "
            "bullet, give each file's sentence for the option, or write 'no sentence' for a file that has "
            "none. The option is SUPPORTED only if every file has a sentence that counts; one missing file "
            "makes it NOT SUPPORTED."
        )
        examples = (
            '- **1. lower rents**: [survey.pdf] "Rents fell sharply in the 1990s." [report.docx] no sentence. NOT SUPPORTED: only one file says it.\n'
            '- **3. more tourists**: [survey.pdf] "Tourists visited the old town." [report.docx] "Tourism grew." NOT SUPPORTED: both files mention tourists, but neither gives them as a reason for the growth the query asks about.\n'
            '- **4. new rail links**: [survey.pdf] "Growth followed the new rail links." [report.docx] "The rail links drew new firms." SUPPORTED: both files give it as a reason for the growth.'
        )
    else:
        rule = f"{subject_rule} The option must also meet every other condition in the query."
        examples = (
            '- **2. lower rents**: [survey.pdf] "Rents fell in Leeds in the 1990s." NOT SUPPORTED: this is about Leeds, not the city the query asks about.\n'
            '- **4. new rail links**: [report.docx] "The city grew after new rail links opened." SUPPORTED: growth is tied directly to the rail links.'
        )
    return f"""{_COMMON_RULES}

The query lists {len(options)} options:
{option_list}

### Checked evidence
Exactly {len(options)} bullets, one per option above, in that order (1 first), and no other text.
Each bullet: the option number and text in bold, then the exact sentence(s) from the context with
their [file name], then SUPPORTED or NOT SUPPORTED, then a short reason. {rule}
For example (a different topic):
{examples}

### Answer
One bold line naming the options you choose, as **Answer: <option numbers> (<short option names>)**.
{choose} Then one or two sentences on why, citing [file name]. Do not list the options again.

Query: {query}

Context:
{context_text}"""


def restore_option_labels(answer: str, options) -> str:
    """The model sometimes garbles option text when copying it ("didn-'t"); put the query's own
    wording back into each '**N. ...**' label."""
    text_by_marker = {marker.upper(): text for marker, text in options}

    def fix(m):
        marker = m.group(2).upper()
        return f"{m.group(1)}**{m.group(2)}. {text_by_marker[marker]}**" if marker in text_by_marker else m.group(0)

    return re.sub(r"^(\s*[-*]\s+)\*\*([1-9][0-9]?|[A-Ha-h])[.)]\s+[^*\n]+\*\*", fix, answer, flags=re.M)


def _words(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower().replace("’", "'").replace("'", ""))


def _quote_in(quote: str, text_trigrams: set) -> bool:
    """Whether a quoted sentence appears in a file's text, allowing small copying slips
    (punctuation, a changed word): 80% of its word trigrams must be there."""
    words = _words(quote)
    if len(words) < 3:
        return False
    trigrams = [tuple(words[i:i + 3]) for i in range(len(words) - 2)]
    return sum(t in text_trigrams for t in trigrams) >= 0.8 * len(trigrams)


_OPTION_CHECK = re.compile(r"^(\s*[-*]\s+\*\*([1-9][0-9]?|[A-Ha-h])[.)]\s+[^*\n]+\*\*\s*:?\s*)(.*)$", re.M)
_FILE_QUOTE = re.compile(r"\[([^\]\n]+)\]\s*(?:\"([^\"\n]+)\"|“([^”\n]+)”|no sentence)", re.I)
_VERDICT_WORD = re.compile(r"\b(NOT\s+SUPPORTED|SUPPORTED)\b\s*:?\s*(.*)$", re.I | re.S)
_ANSWER_LINE = re.compile(r"^\*\*Answer:[^\n]*\*\*[ \t]*\n?(.*?)(?=\n\s*\n|\Z)", re.M | re.S)


def enforce_every_file(answer: str, options, chunks: List[dict], files: List[str]) -> str:
    """For 'supported in both/all files' questions: the model sometimes marks or picks an
    option that only one file supports. Checks each option's quotes against the text of the
    file they are cited from; an option counts only if every file has a quote that is really
    in it. Failed options are marked NOT SUPPORTED and taken out of the Answer line."""
    text_by_file: Dict[str, str] = {}
    for c in chunks:
        name = display_name((c.get("source") or "Unknown").rsplit("/", 1)[-1])
        text_by_file[name] = text_by_file.get(name, "") + " " + (c.get("content") or "")
    index = {}
    for name, text in text_by_file.items():
        w = _words(text)
        index[name] = {tuple(w[i:i + 3]) for i in range(len(w) - 2)}

    supported, failed = set(), {}
    out_lines = []
    for line in answer.split("\n"):
        m = _OPTION_CHECK.match(line)
        if not m:
            out_lines.append(line)
            continue
        marker, rest = m.group(2).upper(), m.group(3)
        verdict = _VERDICT_WORD.search(rest)
        missing = []
        for name in files:
            quotes = [q[1] or q[2] for q in _FILE_QUOTE.findall(rest)
                      if q[0].strip().lower() == name.lower() and (q[1] or q[2])]
            if not any(_quote_in(q, index.get(name, set())) for q in quotes):
                missing.append(name)
        failed[marker] = missing
        said_supported = bool(verdict) and not verdict.group(1).upper().startswith("NOT")
        if said_supported and not missing:
            supported.add(marker)
        elif said_supported:
            reason = f"{' and '.join(missing)} {'has' if len(missing) == 1 else 'have'} no sentence that says it."
            line = m.group(1) + rest[: verdict.start()].rstrip() + f" NOT SUPPORTED: {reason}"
        out_lines.append(line)
    answer = "\n".join(out_lines)

    # Rebuild the Answer line when it names an option that didn't pass
    def fix_answer(m):
        # "**Answer: 1 (plenty of capital), 4 (narratives)**" -> ["1", "4"]
        head_text = re.sub(r"\([^)]*\)", "", m.group(0).split("\n")[0][len("**Answer:"):])
        named = re.findall(r"\b([1-9][0-9]?|[A-H])\b", head_text)
        if all(n.upper() in supported for n in named) and named:
            return m.group(0)
        label = {mk.upper(): text for mk, text in options}
        keep = [mk.upper() for mk, _ in options if mk.upper() in supported]
        dropped = [n.upper() for n in named if n.upper() not in supported]
        if keep:
            head = "**Answer: " + ", ".join(f"{k} ({label[k]})" for k in keep) + "**"
            why = f"{'Only this option is' if len(keep) == 1 else 'These options are'} supported by a sentence in every file ({', '.join(files)})."
        else:
            head = "**Answer: none of the options is supported in every file**"
            why = f"No option has a supporting sentence in every file ({', '.join(files)})."
        if dropped:
            for d in dropped:
                why += (f" Option {d} is not: {' and '.join(failed[d])} has no sentence for it." if failed.get(d)
                        else f" Option {d} is not: its check above found it not supported.")
        return f"{head}  \n{why}"

    return _ANSWER_LINE.sub(fix_answer, answer, count=1)


def _open_prompt(query, context_text) -> str:
    # One sentence per point the context makes, and stop. Tried and dropped: a quoted-evidence
    # section first (the answer then retold every quote, relevant or not), and "at most 3
    # sentences" (the model fills the budget with invented cause-and-effect links). Asked "How
    # did Europe contribute to American film dominance?", those drifted into narratives,
    # Germany and Italy; this answers in the one sentence the text supports.
    return f"""Answer the query using only the context below. The context is grouped by file under
headings like "=== File 1: report.pdf ==="; cite a file by its name only, e.g. [report.pdf] (not
"File 1"). Treat each file as one source.

Find the sentences in the context that answer the query itself. Write one sentence for each point
they make, ending with its [file name], with the key terms in **bold**. Stop when there are no more
such points: if the context makes only one point that answers the query, your answer is one
sentence. Leave out points that are only on the same topic. Never join separate points into a claim
about cause or effect that no single sentence of the context makes, and do not end with a summary
sentence. If the query asks for a summary, write one sentence for each main point instead. Reply in
Markdown with no headings and no list of citations.

Query: {query}

Context:
{context_text}"""


# --- File similarity ("Which 2 files are most similar?") ---

_FILE_WORDS = re.compile(r"\b(?:files?|sources?|documents?|docs|texts?|recordings?|uploads?|transcripts?)\b", re.I)
_SIMILARITY = re.compile(
    r"\b(?:most|least|more|less)\s+(?:similar|alike|different|related)\b|\bclosest\b|\bsimilarity\b"
    r"|\bresembl\w*|\bmost\s+in\s+common\b|\boverlap\w*", re.I)
_LEAST = re.compile(r"\b(?:least|less)\s+(?:similar|alike|related)\b|\bmost\s+different\b", re.I)


def is_file_comparison(query: str) -> bool:
    """Asks which files are most (or least) alike, as opposed to a question about their content."""
    return bool(_SIMILARITY.search(query) and _FILE_WORDS.search(query))


def candidate_pairs(state):
    """Pairs the question is about: all of them, or, for "Which file is closest to
    sample.docx?", only those with the file the query names."""
    sim = state.get("file_similarity")
    if not sim or not sim.get("pairs"):
        return []
    query = state["query"].lower()
    named = [f["filename"] for f in sim["files"]
             if f["filename"].lower() in query or f["filename"].rsplit(".", 1)[0].lower() in query]
    if len(named) == 1:
        return [p for p in sim["pairs"] if named[0] in (p["a_name"], p["b_name"])] or sim["pairs"]
    return sim["pairs"]


def chosen_pair(state):
    """The pair the answer names: highest graph overlap (lowest for 'least similar'); text
    similarity decides when no file has any graph data to compare."""
    pairs = candidate_pairs(state)
    if not pairs:
        return None
    least = bool(_LEAST.search(state["query"]))
    key =(lambda p: p["embedding"]) if all(p["graph"] == 0 for p in pairs) else (lambda p: (p["graph"], p["embedding"]))
    return min(pairs, key=key) if least else max(pairs, key=key)


def similarity_paths(state, limit=6):
    """The chosen pair's shared facts, each as a two-fact 'shared' path for the UI and the map."""
    pair = chosen_pair(state)
    if not pair:
        return []
    return [{"score": s["similarity"], "kind": "shared", "facts": [s["a"], s["b"]]}
            for s in pair["shared_facts"][:limit]]


def _quote(fact):
    return f"\"{(fact.get('evidence') or fact_text(fact)).strip()}\""


def similarity_answer(state) -> str:
    """The model explains a ranking computed from the knowledge graph; the pair, the notes and
    the score table come from code, so the model can't pick a different pair or misquote a score."""
    sim = state["file_similarity"]
    pair = chosen_pair(state)
    least = bool(_LEAST.search(state["query"]))
    a, b = pair["a_name"], pair["b_name"]
    others = [f for f in sim["files"] if f["filename"] not in (a, b)]

    # One fact per sentence: the extractor often takes several facts from one sentence, and a
    # sentence listed as both shared and "only in" made the model contradict itself
    shared_lines, used = [], set()
    for s in pair["shared_facts"]:
        qa, qb = _quote(s["a"]), _quote(s["b"])
        if (qa, qb) not in used:
            used |= {(qa, qb), qa, qb}
            shared_lines.append(f"- [{a}] {qa} / [{b}] {qb}")
    shared = "\n".join(shared_lines) or "- none"
    distinct = []
    for f in [x for x in sim["files"] if x["filename"] in (a, b)] + others:
        quotes = list(dict.fromkeys(q for q in map(_quote, f["only_here"]) if q not in used))[:4]
        if quotes:
            distinct.append(f"Only in {f['filename']}:\n" + "\n".join(f"- {q}" for q in quotes))
    other_names = ", ".join(f["filename"] for f in others)
    task = (
        f"First say what {a} and {b} have in common, from the shared facts, citing [file name] after each claim. "
        + (f"Then say how {other_names} differs, from its own facts. " if others else "")
        if not least else
        f"First say what {a} and {b} each cover that the other does not, citing [file name] after each claim. "
        "Then mention anything they do share, if any. "
    )
    prompt = f"""The files below were compared on the knowledge graph extracted from them. {a} and {b} are
the {'least' if least else 'most'} similar pair. Explain this to the user using only the facts below.

Shared between {a} and {b} (the same claim in both files, possibly worded differently):
{shared}

{chr(10).join(distinct)}

Write 2 or 3 short sentences (under 35 words each) in one paragraph, with no headings, lists,
quotations or percentages: sum up each point in a few words of your own. {task}Write file names
exactly as given, and do not add anything the facts above do not say."""
    llm = chat_client("ANSWER_MODEL", temperature=0)
    writer = _answer_writer()
    heading = f"**{'Least' if least else 'Most'} similar: {a} and {b}**  "
    if writer:
        writer({ANSWER_TOKEN: heading + "\n\n"})
    explanation = _write_answer(llm, [HumanMessage(content=prompt)], writer).strip()

    notes = []
    by_text = (min if least else max)(candidate_pairs(state), key=lambda p: p["embedding"])
    if by_text is not pair:
        notes.append(f"By overall text similarity, **{by_text['a_name']}** and **{by_text['b_name']}** are "
                     f"{'least' if least else 'most'} alike instead ({by_text['embedding']:.0%} vs "
                     f"{pair['embedding']:.0%}): the files read alike, but the knowledge graph found "
                     f"{'fewer' if least else 'more'} shared facts between {a} and {b}.")
    for p in sim["pairs"]:
        for name, other, covered in ((p["a_name"], p["b_name"], p["a_covered"]), (p["b_name"], p["a_name"], p["b_covered"])):
            n_facts = next((f["facts"] for f in sim["files"] if f["filename"] == name), 0)
            if covered >= 0.6 and n_facts >= 3:
                notes.append(f"Most of {name}'s facts ({covered:.0%}) also appear in {other}, so it largely "
                             f"repeats part of {other}.")
    thin = [f["filename"] for f in sim["files"] if f["facts"] == 0]
    if thin:
        notes.append(f"No facts have been extracted from {', '.join(thin)} yet (the knowledge graph may still be "
                     "building), so it was compared on its entities only.")
    if all(p["graph"] == 0 for p in candidate_pairs(state)):
        notes.append("The knowledge graph found nothing shared between any files, so this pair was chosen by text similarity.")
    elif pair["graph"] < 0.05 and not least:
        notes.append("Even this pair shares very few facts, so the ranking is weak.")

    rest = []
    if notes:
        rest.append("\n".join(f"- {n}" for n in notes))
    rest.append("#### How the files compare\n\n" + similarity_table(sim))
    rest.append("*Graph overlap*: facts and entities found in both files' knowledge graphs, matched by meaning "
                "and weighted so that things every file mentions count less. *Text similarity*: how alike the "
                "files' text is overall (a second opinion).")
    if writer:
        writer({ANSWER_TOKEN: "\n\n" + "\n\n".join(rest)})
    return "\n\n".join([heading, explanation] + rest)


def synthesize_answer(state: AgentState):
    """Step 3: Generate final response"""
    if state.get("file_similarity"):
        return {"final_answer": similarity_answer(state)}
    if not state["refined_context"]:
        return {"final_answer": "I couldn't find relevant information in the knowledge base to answer this query."}

    # temperature 0: deterministic answers; sampling made option picks vary run to run
    llm = chat_client("ANSWER_MODEL", temperature=0)

    context_text = _format_context_by_file(state["refined_context"])
    facts_text = format_kg_facts(answer_paths(state))
    if facts_text:
        context_text += (
            "\n\nConnections found in the knowledge graph (facts extracted automatically from the "
            "files above; they show how things link across sentences and files, but may contain "
            "errors, so confirm each against the file text before using it):\n" + facts_text
        )

    files = list(dict.fromkeys(
        display_name((c.get("source") or "Unknown").rsplit("/", 1)[-1]) for c in state["refined_context"]
    ))
    options = listed_options(state["query"])
    if options:
        prompt = _options_prompt(state["query"], context_text, options, files)
    else:
        prompt = _open_prompt(state["query"], context_text)

    messages = [HumanMessage(content=prompt)]
    if state.get("corpus_id") and state["corpus_id"] != "wiki":
        messages.insert(0, SystemMessage(
            content=(
                "Answer only from the provided uploaded-file context. Do not use outside "
                "knowledge. If the context does not contain the answer, say that the "
                "uploaded files do not provide enough information."
            )
        ))
    # Streamed to the UI as it is written; the corrections below reach it with the final
    # answer, which replaces the streamed text
    answer = _write_answer(llm, messages, _answer_writer())
    if options:
        answer = restore_option_labels(answer, options)
        if _ALL_FILES.search(state["query"]) and len(files) > 1:
            answer = enforce_every_file(answer, options, state["refined_context"], files)
    return {"final_answer": answer}

# --- Graph Construction ---

def build_query_graph(retriever: HybridRetriever):
    workflow = StateGraph(AgentState)

    workflow.add_node("planner", plan_query_node)
    workflow.add_node("retriever", make_retrieve_context_node(retriever))
    workflow.add_node("analyst", make_critical_analysis_node(retriever))
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
