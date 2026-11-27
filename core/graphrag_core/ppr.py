"""
Local push Personalized PageRank (PPR) over any graph that can list a node's neighbours.

"Push" (Andersen, Chung & Lang, 2006) computes PPR locally: each seed node holds some
probability mass; a node keeps a RESTART share of what it holds and passes the rest to its
neighbours in proportion to edge weight, and only nodes holding at least EPSILON x degree keep
passing. Mass fades with every step, so the work depends on EPSILON and RESTART, not on the
size of the graph: neighbours are fetched (from Neo4j, or an in-memory code graph) only as the
push reaches them.

Used by the text search (query_processor/app/local_ppr.py, over chunks and entities) and by
the VS Code extension (vscode-extension/sidecar, over symbols and code relationships).
"""
import logging
from collections import defaultdict
from typing import Callable, Dict, Hashable, Iterable, Mapping, Tuple

logger = logging.getLogger(__name__)

RESTART = 0.5             # share of a node's mass it keeps; higher = stays closer to the seeds
EPSILON = 1e-4            # stop pushing a node once it holds < EPSILON x its degree
MAX_ROUNDS = 60
MAX_FETCHED_NODES = 5000  # safety cap on the neighbourhood loaded per run

# Given nodes, returns {node: {neighbour: edge weight}}; nodes left out have no neighbours
Neighbours = Callable[[list], Mapping[Hashable, Mapping[Hashable, float]]]


def push_ppr(
    seeds: Mapping[Hashable, float],
    neighbours: Neighbours,
    *,
    restart: float = RESTART,
    epsilon: float = EPSILON,
    max_rounds: int = MAX_ROUNDS,
    max_nodes: int = MAX_FETCHED_NODES,
) -> Tuple[Dict[Hashable, float], int]:
    """PPR from `seeds` ({node: weight}, normalised here). Returns ({node: score}, number of
    nodes whose neighbours were fetched)."""
    total = sum(w for w in seeds.values() if w > 0)
    if not total:
        return {}, 0
    p = defaultdict(float)
    r = defaultdict(float)
    for node, weight in seeds.items():
        if weight > 0:
            r[node] += weight / total

    adjacency, degree = {}, {}
    for _ in range(max_rounds):
        # Unknown degree counts as 1 (the smallest possible) so the node gets fetched
        active = [u for u, ru in r.items() if ru > 0 and ru >= epsilon * degree.get(u, 1.0)]
        unknown = [u for u in active if u not in adjacency]
        if unknown:
            if len(adjacency) + len(unknown) > max_nodes:
                logger.warning("Local PPR stopped at the %d-node fetch limit", max_nodes)
                break
            fetched = neighbours(unknown)
            for node in unknown:
                out = dict(fetched.get(node) or {})
                adjacency[node] = out
                degree[node] = sum(out.values())
            active = [u for u in active if r[u] >= epsilon * max(degree[u], 1.0)]
        if not active:
            break
        pushes = []
        for u in active:
            ru, r[u] = r[u], 0.0
            if not degree[u]:
                p[u] += ru  # no neighbours: it keeps everything it holds
                continue
            p[u] += restart * ru
            pushes.append((u, (1 - restart) * ru / degree[u]))
        for u, share in pushes:
            for v, weight in adjacency[u].items():
                r[v] += share * weight

    return dict(p), len(adjacency)


def top(scores: Mapping[Hashable, float], k: int, exclude: Iterable[Hashable] = ()) -> list:
    """The k best-scoring nodes, best first, leaving out `exclude`."""
    skip = set(exclude)
    return [n for n, _ in sorted(scores.items(), key=lambda kv: -kv[1]) if n not in skip][:k]
