"""
Local push Personalized PageRank (PPR) over a corpus's graph in Neo4j (the wiki corpus or
an upload workspace).

The graph: chunks and entities as nodes; edges are chunk -RELATION- chunk (NEXT_CHUNK in
document order; a relation's `weight` is used when set), chunk -MENTIONS- entity, and
entity -FACT- entity (weighted by how many facts link the pair). The wiki and upload
workspaces have the same graph (build_wiki.py builds the wiki like an upload). PPR scores how
strongly each chunk is connected to the start nodes (the entities the question names and the
chunks vector search found), at any distance.

"Push" (Andersen, Chung & Lang, 2006) computes it locally: each start node holds some
probability mass; a node keeps a RESTART share of what it holds and passes the rest to its
neighbours in proportion to edge weight, and only nodes holding at least EPSILON x degree
keep passing. Mass fades with every step, so the work depends on EPSILON and RESTART, not
on the size of the workspace; neighbours are fetched from Neo4j as the push reaches them.
"""
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

RESTART = 0.5          # share of a node's mass it keeps; higher = stays closer to the start
EPSILON = 1e-4         # stop pushing a node once it holds < EPSILON x its degree
MAX_ROUNDS = 60
MAX_FETCHED_NODES = 5000   # safety cap on the neighbourhood loaded from Neo4j per search


def chunk_node(chunk_id):
    return ("c", chunk_id)


def entity_node(key):
    return ("e", key)


class LocalPPR:
    def __init__(self, neo4j_driver):
        self.driver = neo4j_driver
        self._ensure_index()

    def _ensure_index(self):
        # Chunks are looked up by id on every push round
        try:
            with self.driver.session() as session:
                session.run("CREATE INDEX chunk_id IF NOT EXISTS FOR (c:Chunk) ON (c.chunk_id)").consume()
        except Exception:
            logger.exception("Could not create the Chunk id index")

    def _fetch(self, nodes, corpus_id, file_ids):
        """Weighted neighbour lists for the given nodes, limited to the selected files.
        Chunk edges never cross corpora (chunk ids are '<file_id>:<index>' and file ids are
        unique), so chunk neighbours need no corpus filter."""
        params = {"corpus_id": corpus_id, "file_ids": file_ids or [], "all": not file_ids}
        chunk_ids = [n[1] for n in nodes if n[0] == "c"]
        keys = [n[1] for n in nodes if n[0] == "e"]
        adjacency = {n: defaultdict(float) for n in nodes}
        with self.driver.session() as session:
            if chunk_ids:
                for row in session.run(
                    """
                    UNWIND $ids AS cid
                    MATCH (c:Chunk {chunk_id: cid})
                    RETURN cid AS id,
                           [(c)-[r:RELATION]-(n:Chunk) WHERE $all OR n.file_id IN $file_ids
                               | [n.chunk_id, coalesce(r.weight, 1.0)]] AS chunks,
                           [(c)-[:MENTIONS]->(e:Entity) | e.key] AS entities
                    """,
                    ids=chunk_ids, **params,
                ):
                    out = adjacency[chunk_node(row["id"])]
                    for cid, weight in row["chunks"]:
                        out[chunk_node(cid)] += float(weight)
                    for key in row["entities"]:
                        out[entity_node(key)] += 1.0
            if keys:
                for row in session.run(
                    """
                    UNWIND $keys AS k
                    MATCH (e:Entity {corpus_id: $corpus_id, key: k})
                    RETURN k AS key,
                           [(e)<-[:MENTIONS]-(c:Chunk) WHERE $all OR c.file_id IN $file_ids | c.chunk_id] AS chunks,
                           [(e)-[f:FACT]-(o:Entity) WHERE $all OR f.file_id IN $file_ids | o.key] AS entities
                    """,
                    keys=keys, **params,
                ):
                    out = adjacency[entity_node(row["key"])]
                    for cid in row["chunks"]:
                        out[chunk_node(cid)] += 1.0
                    for key in row["entities"]:
                        if key != row["key"]:
                            out[entity_node(key)] += 1.0  # one per fact: well-linked pairs weigh more
        return adjacency

    def scores(self, seeds, corpus_id, file_ids=None):
        """PPR from `seeds` ({node: weight}). Returns ({chunk_id: score}, nodes fetched)."""
        total = sum(w for w in seeds.values() if w > 0)
        if not total:
            return {}, 0
        p = defaultdict(float)
        r = defaultdict(float)
        for node, weight in seeds.items():
            if weight > 0:
                r[node] += weight / total

        adjacency, degree = {}, {}
        for _ in range(MAX_ROUNDS):
            # Unknown degree counts as 1 (the smallest possible) so the node gets fetched
            active = [u for u, ru in r.items() if ru > 0 and ru >= EPSILON * degree.get(u, 1.0)]
            unknown = [u for u in active if u not in adjacency]
            if unknown:
                if len(adjacency) + len(unknown) > MAX_FETCHED_NODES:
                    logger.warning("Local PPR stopped at the %d-node fetch limit", MAX_FETCHED_NODES)
                    break
                for node, out in self._fetch(unknown, corpus_id, file_ids).items():
                    adjacency[node] = out
                    degree[node] = sum(out.values())
                active = [u for u in active if r[u] >= EPSILON * max(degree[u], 1.0)]
            if not active:
                break
            pushes = []
            for u in active:
                ru, r[u] = r[u], 0.0
                if not degree[u]:
                    p[u] += ru  # no neighbours (e.g. a one-chunk file before extraction): keep it all
                    continue
                p[u] += RESTART * ru
                pushes.append((u, (1 - RESTART) * ru / degree[u]))
            for u, share in pushes:
                for v, weight in adjacency[u].items():
                    r[v] += share * weight

        return {node[1]: score for node, score in p.items() if node[0] == "c"}, len(adjacency)
