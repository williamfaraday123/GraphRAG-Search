"""
Leiden community detection (Traag, Waltman & van Eck, "From Louvain to Leiden: guaranteeing
well-connected communities", Scientific Reports 9, 2019), maximising modularity with a
resolution parameter, on a weighted undirected graph. Standard library only: no igraph or
leidenalg native modules.

Each level has three phases:
  1. fast local moving: nodes move, one at a time, to the neighbouring community with the best
     modularity gain; only the neighbours of a node that moved are visited again;
  2. refinement: inside each community, nodes start alone and merge (randomly, favouring better
     merges) only with sub-communities that are well connected to the rest of the community,
     so no community ends up internally disconnected (Louvain's known flaw);
  3. aggregation: each refined sub-community becomes one node, starting in its unrefined
     community, and the next level repeats on that smaller graph.
It stops when local moving changes nothing.

Used by the VS Code extension for software architecture recovery: clustering the module graph
into components when a repository has no build files that say where its components are.
"""
import math
import random
from collections import defaultdict
from typing import Dict, Hashable, List, Mapping, Optional

Graph = Mapping[Hashable, Mapping[Hashable, float]]

THETA = 0.01   # refinement randomness: lower is greedier


class _Level:
    """One level's graph: nodes 0..n-1, adjacency without self-loops, each node's strength
    (its weighted degree in the original graph, self-loops included)."""

    def __init__(self, adj: List[Dict[int, float]], strength: List[float]):
        self.adj, self.strength = adj, strength
        self.n = len(adj)


def modularity(graph: Graph, communities: Mapping[Hashable, int], resolution: float = 1.0) -> float:
    """Newman-Girvan modularity of a partition (with the resolution parameter)."""
    two_m = sum(w for u, nbrs in graph.items() for v, w in nbrs.items())
    if two_m <= 0:
        return 0.0
    inside: Dict[int, float] = defaultdict(float)
    total: Dict[int, float] = defaultdict(float)
    for u, nbrs in graph.items():
        for v, w in nbrs.items():
            total[communities[u]] += w
            if communities[u] == communities[v]:
                inside[communities[u]] += w
    return sum(inside[c] / two_m - resolution * (total[c] / two_m) ** 2 for c in total)


def leiden(graph: Graph, resolution: float = 1.0, seed: Optional[int] = 0,
           max_levels: int = 20) -> Dict[Hashable, int]:
    """node -> community number (0, 1, ... largest first). `graph` must be symmetric:
    graph[u][v] == graph[v][u]. Isolated nodes get a community of their own."""
    rng = random.Random(seed)
    nodes = list(graph)
    if not nodes:
        return {}
    index = {u: i for i, u in enumerate(nodes)}
    adj: List[Dict[int, float]] = [defaultdict(float) for _ in nodes]
    strength = [0.0] * len(nodes)
    for u, nbrs in graph.items():
        for v, w in nbrs.items():
            if w <= 0 or v not in index:
                continue
            strength[index[u]] += w
            if u != v:
                adj[index[u]][index[v]] += w
    two_m = sum(strength)
    if two_m <= 0:
        return {u: i for i, u in enumerate(nodes)}

    level = _Level([dict(a) for a in adj], strength)
    # members[i]: original nodes inside node i of the current level
    members: List[List[int]] = [[i] for i in range(len(nodes))]
    partition = list(range(level.n))
    for depth in range(max_levels):
        partition, moved = _move_nodes_fast(level, partition, two_m, resolution, rng)
        # Done: nothing moved (after the first level), or every node is its own community
        if (depth > 0 and not moved) or len(set(partition)) == level.n:
            break
        refined = _refine(level, partition, two_m, resolution, rng)
        level, members, partition = _aggregate(level, members, refined, partition)
    # Flatten, numbering communities largest first
    by_comm: Dict[int, List[int]] = defaultdict(list)
    for node, comm in enumerate(partition):
        by_comm[comm].extend(members[node])
    order = sorted(by_comm.values(), key=lambda m: (-len(m), min(m)))
    return {nodes[i]: c for c, group in enumerate(order) for i in group}


def _move_nodes_fast(level: _Level, partition: List[int], two_m: float, resolution: float,
                     rng: random.Random):
    comm = list(partition)
    total = defaultdict(float)          # community -> summed strength
    for i, c in enumerate(comm):
        total[c] += level.strength[i]
    order = list(range(level.n))
    rng.shuffle(order)
    queue, queued = list(order), set(order)
    head, moved = 0, False
    next_id = max(comm) + 1                  # for a node that is best off alone
    while head < len(queue):
        i = queue[head]
        head += 1
        queued.discard(i)
        k = level.strength[i]
        old = comm[i]
        links: Dict[int, float] = defaultdict(float)
        for j, w in level.adj[i].items():
            links[comm[j]] += w
        total[old] -= k
        # Gain of joining c (relative to being alone): links to c - resolution * k * total(c) / 2m
        best, best_gain = old, links.get(old, 0.0) - resolution * k * total[old] / two_m
        for c, w in links.items():
            gain = w - resolution * k * total[c] / two_m
            if gain > best_gain + 1e-12:
                best, best_gain = c, gain
        if best_gain < -1e-12 and total[old] > 1e-12:  # alone beats everywhere: a new community
            best, next_id = next_id, next_id + 1
        total[best] += k
        if best != old:
            comm[i] = best
            moved = True
            for j in level.adj[i]:
                if comm[j] != best and j not in queued:
                    queue.append(j)
                    queued.add(j)
    return comm, moved


def _refine(level: _Level, partition: List[int], two_m: float, resolution: float,
            rng: random.Random) -> List[int]:
    refined = list(range(level.n))       # every node starts alone
    sub_total = list(level.strength)     # refined community -> summed strength
    singleton = [True] * level.n
    groups: Dict[int, List[int]] = defaultdict(list)
    for i, c in enumerate(partition):
        groups[c].append(i)
    for c, group in groups.items():
        if len(group) == 1:
            continue
        comm_total = sum(level.strength[i] for i in group)
        in_group = set(group)
        # Links from each node to the rest of its community
        to_rest = {i: sum(w for j, w in level.adj[i].items() if j in in_group) for i in group}
        # Refined community -> its links to the rest of the community (starts as the node's)
        sub_rest = {i: to_rest[i] for i in group}
        order = list(group)
        rng.shuffle(order)
        for i in order:
            if not singleton[i]:
                continue
            k = level.strength[i]
            # Only well-connected nodes move
            if to_rest[i] < resolution * k * (comm_total - k) / two_m:
                continue
            links: Dict[int, float] = defaultdict(float)
            for j, w in level.adj[i].items():
                if j in in_group:
                    links[refined[j]] += w
            current = refined[i]
            candidates, gains = [], []
            for r, w in links.items():
                if r == current:
                    continue
                t = sub_total[r]
                # ...into well-connected sub-communities only
                if sub_rest[r] < resolution * t * (comm_total - t) / two_m:
                    continue
                gain = w - resolution * k * t / two_m
                if gain >= 0:
                    candidates.append(r)
                    gains.append(gain)
            if not candidates:
                continue
            top = max(gains)
            weights = [math.exp((g - top) / THETA) for g in gains]
            target = rng.choices(candidates, weights)[0]
            # Merge i into target
            w_i_target = links[target]
            refined[i] = target
            sub_total[target] += k
            sub_total[current] -= k
            sub_rest[target] = sub_rest[target] + to_rest[i] - 2 * w_i_target
            # A non-empty sub-community r always still holds node r (only singletons move)
            singleton[i] = singleton[target] = False
    return refined


def _aggregate(level: _Level, members: List[List[int]], refined: List[int], partition: List[int]):
    ids: Dict[int, int] = {}
    for r in refined:
        ids.setdefault(r, len(ids))
    n = len(ids)
    adj: List[Dict[int, float]] = [defaultdict(float) for _ in range(n)]
    strength = [0.0] * n
    new_members: List[List[int]] = [[] for _ in range(n)]
    new_partition = [0] * n
    for i in range(level.n):
        a = ids[refined[i]]
        strength[a] += level.strength[i]
        new_members[a].extend(members[i])
        new_partition[a] = partition[i]
        for j, w in level.adj[i].items():
            b = ids[refined[j]]
            if a != b:
                adj[a][b] += w
    return _Level([dict(x) for x in adj], strength), new_members, new_partition
