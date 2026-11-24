"""
Knowledge-graph retrieval for upload workspaces and the wiki.

Vector search finds chunks that look like the question. This module follows the facts the
local model extracted (see knowledge_graph.py) to find chunks that answer it but don't look
like it, e.g. "What did the country that overtook Europe do about exports?":

  1. Start entities: entities named in the question ("Europe"), plus, with lower weight,
     entities in facts from the chunks vector search already returned.
  2. Walk: every path of 1 or 2 facts from a start entity, in either direction
     (Europe <-overtook- Hollywood -was backed by-> United States).
  3. Score: each fact is compared with the question using the same embedding model as
     vector search; a path scores the mean of its facts, scaled by the start entity's
     weight, slightly lower for 2 hops, with a bonus when it links two entities the
     question names.
  4. Return the best paths, a graph score per chunk, the chunks behind the path facts that
     vector search missed, and chunk-to-chunk links along each path (for PageRank).

Name groups: the extractor often keeps a proper name inside a longer entity ("films from new
Hollywood studios", "learning from Hollywood"), and those entities share no fact with
"Hollywood". The walk treats entities that contain the same proper-name entity as one stop,
so a path can pass from one to the other without a fact between them.

connect() joins entities named in different parts of a question (the planner may search
them as separate sub-queries): paths of up to 3 facts from one named entity to another,
meeting at a shared entity or name group. "Which countries that learned from Hollywood
supplied the country whose films flooded the markets?" joins
  (rest of the world) survived (learning from Hollywood) ~ (films from Hollywood studios) flooded (world's film markets)

Only the facts the walk can use are loaded: those touching a start entity, then those
touching each entity one fact away, until MAX_CANDIDATE_PATHS paths exist. So the cost
depends on the start entities' neighbourhoods, not on the size of the corpus.
"""
import itertools
import logging
import numpy as np

from chunk_context import embed_text
from document_text import display_name
from knowledge_graph import _WORD, _content_words, _word_matches

logger = logging.getLogger(__name__)

MAX_NAME_CANDIDATES = 5000    # entities sharing a word with the question, checked by named_entities
MAX_FACTS_PER_START = 300     # facts loaded per start entity (first hop)
MAX_FACTS_PER_MIDDLE = 100    # facts loaded per entity one hop away (hubs are cut here)
MAX_CANDIDATE_PATHS = 600     # caps embedding work on dense graphs
MAX_PATHS = 6
MIN_PATH_SCORE = 0.3          # cosine-based; unrelated facts score ~0.0-0.2 with MiniLM
NAMED_SEED_WEIGHT = 1.0
CONTEXT_SEED_WEIGHT = 0.7
CONTEXT_SEED_CHUNKS = 3       # top vector chunks whose facts seed the walk
TWO_HOP_FACTOR = 0.9
BRIDGE_BONUS = 0.15
MIN_CORE_LENGTH = 4           # shortest proper name that links entities containing it
MAX_NAME_LINKS = 10           # entities linked through one proper name ("Hollywood")
MAX_CONNECT_FACTS = 3         # longest path connect() returns
MAX_CONNECT_PATHS = 4
CONNECT_BONUS = 0.25          # a path joining two parts of the question ranks above single-part paths


def _ngrams(key):
    """Every run of consecutive words in an entity key, including the whole key."""
    words = key.split()
    return {" ".join(words[i:j]) for i in range(len(words)) for j in range(i + 1, len(words) + 1)}


def _contains_name(key, core):
    return f" {core} " in f" {key} "


def _unit(vectors):
    vectors = np.asarray(vectors, dtype=np.float32)
    return vectors / (np.linalg.norm(vectors, axis=-1, keepdims=True) + 1e-9)


def fact_text(fact):
    return f"{fact['source_label']} {fact['relation']} {fact['target_label']}"


def named_entities(query, labels):
    """Entity keys any of whose names (label, or list of label and aliases) has every
    content word in the query (loose stem match)."""
    query_words = {w.lower().replace("’", "'") for w in _WORD.findall(query)}
    found = []
    for key, names in labels.items():
        for label in ([names] if isinstance(names, str) else names):
            words = _content_words(label)
            if words and all(_word_matches(w, query_words) for w in words):
                found.append(key)
                break
    # Prefer the most specific: drop "hollywood" when "hollywood films" also matched
    return [k for k in found if not any(o != k and f" {k} " in f" {o} " for o in found)]


class GraphRetriever:
    def __init__(self, neo4j_driver, embedding_service):
        self.driver = neo4j_driver
        self.embedding_service = embedding_service

    def _name_candidates(self, corpus_id, file_ids, query):
        """Entities with facts in the selected files whose name or an alias contains one of
        the query's content words (or its first five letters, matching _word_matches).
        A superset of what named_entities() accepts: it makes the final decision.
        Returns {key: [name, *aliases]}."""
        stems = sorted({w[:5] if len(w) >= 5 else w for w in _content_words(query)})
        if not stems:
            return {}
        with self.driver.session() as session:
            rows = session.run(
                """
                MATCH (e:Entity {corpus_id: $corpus_id})
                WHERE any(s IN $stems WHERE e.key CONTAINS s
                          OR any(a IN coalesce(e.aliases, []) WHERE toLower(a) CONTAINS s))
                  AND EXISTS { MATCH (e)-[f:FACT]-() WHERE $all OR f.file_id IN $file_ids }
                RETURN e.key AS key, e.name AS name, coalesce(e.aliases, []) AS aliases
                LIMIT $limit
                """,
                corpus_id=corpus_id, stems=stems, file_ids=file_ids or [], all=not file_ids,
                limit=MAX_NAME_CANDIDATES,
            ).data()
        return {r["key"]: [r["name"] or r["key"], *r["aliases"]] for r in rows}

    def _context_entities(self, corpus_id, file_ids, chunk_ids):
        """Entities in the facts extracted from the given chunks."""
        if not chunk_ids:
            return []
        with self.driver.session() as session:
            rows = session.run(
                """
                UNWIND $ids AS cid
                MATCH (c:Chunk {chunk_id: cid})-[:MENTIONS]->(e:Entity {corpus_id: $corpus_id})-[f:FACT]-()
                WHERE f.chunk_id = cid AND ($all OR f.file_id IN $file_ids)
                RETURN DISTINCT e.key AS key
                """,
                ids=list(chunk_ids), corpus_id=corpus_id, file_ids=file_ids or [], all=not file_ids,
            ).data()
        return [r["key"] for r in rows]

    def _facts_touching(self, corpus_id, file_ids, keys, limit):
        """{key: [fact, ...]}: up to `limit` facts (in the selected files) that start or end
        at each entity. Each fact is a dict with an 'id' that is the same wherever it loads."""
        if not keys:
            return {}
        with self.driver.session() as session:
            rows = session.run(
                """
                UNWIND $keys AS k
                MATCH (e:Entity {corpus_id: $corpus_id, key: k})
                CALL (e) {
                    MATCH (e)-[f:FACT]-(:Entity)
                    WHERE $all OR f.file_id IN $file_ids
                    RETURN f LIMIT $limit
                }
                WITH k, f, startNode(f) AS a, endNode(f) AS b
                OPTIONAL MATCH (c:Chunk {chunk_id: f.chunk_id})
                RETURN k, elementId(f) AS id,
                       a.key AS source, a.name AS source_label, coalesce(a.aliases, []) AS source_aliases,
                       b.key AS target, b.name AS target_label, coalesce(b.aliases, []) AS target_aliases,
                       f.relation AS relation, f.chunk_id AS chunk_id, f.file_id AS file_id,
                       f.evidence AS evidence, c.source AS uri
                """,
                keys=list(keys), corpus_id=corpus_id, file_ids=file_ids or [], all=not file_ids, limit=limit,
            ).data()
        out = {k: [] for k in keys}
        for row in rows:
            key = row.pop("k")
            row["filename"] = display_name((row.pop("uri") or "").rsplit("/", 1)[-1]) or "Unknown file"
            out[key].append(row)
        return out

    def _name_groups(self, corpus_id, keys):
        """{key: set of keys}: each key with the entities that share a proper-name entity
        with it. "learning from hollywood" -> {it, "hollywood", "hollywood films",
        "films from new hollywood studios"}. A key with no proper name in it maps to itself."""
        groups = {k: {k} for k in keys}
        pairs = [{"k": k, "core": c} for k in keys for c in _ngrams(k) if len(c) >= MIN_CORE_LENGTH]
        if not pairs:
            return groups
        with self.driver.session() as session:
            # Proper-name entities whose key is a word run inside each key (index lookups)
            rows = session.run(
                """
                UNWIND $pairs AS p
                MATCH (core:Entity {corpus_id: $corpus_id, key: p.core})
                WHERE core.kind = 'name'
                RETURN p.k AS k, core.key AS core
                """,
                pairs=pairs, corpus_id=corpus_id,
            ).data()
            cores = sorted({r["core"] for r in rows})
            # Entities containing each proper name (text index on Entity.key)
            containers = {}
            if cores:
                for r in session.run(
                    """
                    UNWIND $cores AS ck
                    CALL (ck) {
                        MATCH (m:Entity) WHERE m.key CONTAINS ck AND m.corpus_id = $corpus_id AND m.key <> ck
                        RETURN m.key AS mk LIMIT $limit
                    }
                    RETURN ck, collect(mk) AS keys
                    """,
                    cores=cores, corpus_id=corpus_id, limit=MAX_NAME_LINKS * 3,
                ).data():
                    containers[r["ck"]] = [m for m in r["keys"] if _contains_name(m, r["ck"])][:MAX_NAME_LINKS]
        for r in rows:
            groups[r["k"]] |= {r["core"], *containers.get(r["core"], [])}
        return groups

    def _walk(self, corpus_id, file_ids, seeds, by_entity, groups, remember):
        """Every 1- and 2-fact path from each start entity, as (start, [facts], end).
        Facts are loaded into by_entity as the walk reaches them; entities in one name
        group count as one stop. Groups of starts and middles are added to `groups`."""
        def load_groups(keys):
            missing = [k for k in keys if k not in groups]
            if missing:
                groups.update(self._name_groups(corpus_id, missing))

        def facts_near(key, limit):
            members = groups.get(key, {key})
            remember(self._facts_touching(corpus_id, file_ids, [m for m in members if m not in by_entity], limit))
            seen, out = set(), []
            for m in members:
                for f in by_entity.get(m, []):
                    if f["id"] not in seen:
                        seen.add(f["id"])
                        out.append(f)
            return out

        def other_end(f, group):
            return f["target"] if f["source"] in group else f["source"]

        paths = []
        for seed in seeds:
            load_groups([seed])
            seed_group = groups[seed]
            first_hop = facts_near(seed, MAX_FACTS_PER_START)
            middles = {other_end(f, seed_group) for f in first_hop}
            load_groups(middles)
            # One query for every middle group's facts, then the walk reads them from by_entity
            remember(self._facts_touching(
                corpus_id, file_ids,
                sorted({m for mid in middles for m in groups[mid] if m not in by_entity}), MAX_FACTS_PER_MIDDLE))
            for f in first_hop:
                middle = other_end(f, seed_group)
                if middle in seed_group:
                    continue  # a fact inside the name group ("hollywood" -> "hollywood films")
                paths.append((seed, [f], middle))
                middle_group = groups[middle]
                for g in facts_near(middle, MAX_FACTS_PER_MIDDLE):
                    if g["id"] == f["id"]:
                        continue
                    end = other_end(g, middle_group)
                    if end not in seed_group and end not in middle_group:
                        paths.append((seed, [f, g], end))
            if len(paths) >= MAX_CANDIDATE_PATHS:
                break
        return paths[:MAX_CANDIDATE_PATHS]

    def _relevance(self, facts, query_vector):
        """{fact id: cosine similarity of the fact's text to the query}."""
        unique = {f["id"]: f for f in facts}
        if not unique:
            return {}
        ids = list(unique)
        vectors = _unit(self.embedding_service.generate_batch_embeddings([fact_text(unique[i]) for i in ids]))
        return dict(zip(ids, (vectors @ _unit(query_vector)).tolist()))

    def connect(self, query_vector, corpus_id, keys, file_ids=None, known_chunk_ids=(), max_new_chunks=5):
        """Paths of up to MAX_CONNECT_FACTS facts joining two of the given entity keys (the
        entities a whole question names, across its sub-queries), best first, plus the chunks
        behind them that aren't in known_chunk_ids. Same shape as search()'s result."""
        empty = {"paths": [], "chunk_scores": {}, "chunks": []}
        keys = list(dict.fromkeys(keys))
        if len(keys) < 2:
            return empty
        by_entity, groups, labels = {}, {}, {}

        def remember(facts_by_key):
            for key, facts in facts_by_key.items():
                by_entity[key] = facts

        walked = self._walk(corpus_id, file_ids, keys, by_entity, groups, remember)

        def same(a, b):
            return a == b or b in groups.get(a, ()) or a in groups.get(b, ())

        def reached(path_facts, start):
            """Entities a path passes through or ends at, in order (after the start)."""
            out, at = [], start
            for f in path_facts:
                at = f["target"] if same(f["source"], at) else f["source"]
                out.append(at)
            return out

        candidates = {}
        for (a, pa, end_a), (b, pb, end_b) in itertools.combinations(walked, 2):
            if same(a, b):
                continue
            ids_a = [f["id"] for f in pa]
            if any(f["id"] in ids_a for f in pb):
                continue
            # Meet at the far ends: a -> ... -> X ~ X <- ... <- b
            if len(pa) + len(pb) <= MAX_CONNECT_FACTS and same(end_a, end_b):
                facts = pa + list(reversed(pb))
                candidates.setdefault(tuple(f["id"] for f in facts), (a, b, facts))
        for a, pa, end_a in walked:
            # A path from one named entity that already reaches another
            for b in keys:
                if not same(a, b) and any(same(x, b) for x in reached(pa, a)):
                    candidates.setdefault(tuple(f["id"] for f in pa), (a, b, pa))
        if not candidates:
            return empty

        relevance = self._relevance(
            [f for _, _, facts in candidates.values() for f in facts] + [f for k in keys for f in by_entity.get(k, [])],
            query_vector)

        def own_fact(key, facts):
            """The named entity's own most relevant fact, when the path only reaches it through
            its name group: "(films from Hollywood studios) flooded ..." reaches "learning from
            Hollywood", but only "(rest of the world) survived (learning from Hollywood)" says
            who learned."""
            if any(key in (f["source"], f["target"]) for f in facts):
                return None
            ids = {f["id"] for f in facts}
            own = [f for f in by_entity.get(key, []) if f["id"] not in ids]
            return max(own, key=lambda f: relevance[f["id"]], default=None)

        anchored = {}
        for a, b, facts in candidates.values():
            first, last = own_fact(a, facts), own_fact(b, facts)
            facts = ([first] if first else []) + facts + ([last] if last and last is not first else [])
            anchored.setdefault(tuple(f["id"] for f in facts), (a, b, facts))

        scored = []
        for a, b, facts in anchored.values():
            mean = float(np.mean([relevance[f["id"]] for f in facts]))
            if mean < MIN_PATH_SCORE:
                continue
            score = (mean + CONNECT_BONUS) * TWO_HOP_FACTOR ** (len(facts) - 1)
            scored.append((score, a, b, facts))
        scored.sort(key=lambda s: (-s[0], len(s[3])))

        # A chain that contains a shorter chosen one only adds detours ("A -> B -> C" when
        # "A -> C" was chosen), and a shorter chain inside a chosen one is kept: unlike
        # search(), facts already shown don't make a different chain redundant
        chosen, chosen_keys = [], []
        for score, a, b, facts in sorted(scored, key=lambda s: (len(s[3]), -s[0])):
            key = frozenset(f["id"] for f in facts)
            if any(k <= key for k in chosen_keys):
                continue
            chosen_keys.append(key)
            chosen.append({"score": round(score, 4), "start": a, "end": b, "facts": facts})
        chosen = sorted(chosen, key=lambda p: -p["score"])[:MAX_CONNECT_PATHS]
        for f in (f for p in chosen for f in p["facts"]):
            labels.setdefault(f["source"], f["source_label"])
            labels.setdefault(f["target"], f["target_label"])

        chunk_scores = {}
        for p in chosen:
            for f in p["facts"]:
                chunk_scores[f["chunk_id"]] = max(chunk_scores.get(f["chunk_id"], 0.0), p["score"])
        new_chunks = self._new_chunks(chunk_scores, known_chunk_ids, query_vector, max_new_chunks)
        logger.info("Knowledge graph: %d connecting path(s) between %d named entities, %d chunk(s) added",
                    len(chosen), len(keys), len(new_chunks))
        return {
            "paths": [
                {"score": p["score"], "links": [labels.get(p["start"], p["start"]), labels.get(p["end"], p["end"])],
                 "facts": [_public(f) for f in p["facts"]]}
                for p in chosen
            ],
            "chunk_scores": chunk_scores,
            "chunks": new_chunks,
        }

    def _new_chunks(self, chunk_scores, known_chunk_ids, query_vector, max_new_chunks):
        """The best-scoring chunks not already known, loaded, with their similarity to the query."""
        known = set(known_chunk_ids)
        new_ids = [c for c in sorted(chunk_scores, key=lambda c: -chunk_scores[c]) if c not in known][:max_new_chunks]
        out = []
        if new_ids:
            rows = [r for r in self._load_chunks(new_ids) if r.get("content")]
            if rows:
                # Embedded as Milvus stored them (with the context header), so the scores
                # compare with vector hits
                sims = _unit(self.embedding_service.generate_batch_embeddings(
                    [embed_text(r["content"], r.get("header")) for r in rows])) @ _unit(query_vector)
                for row, sim in zip(rows, sims.tolist()):
                    out.append({
                        "id": row["id"], "content": row["content"], "source": row["source"],
                        "score": float(sim),  # cosine, same scale as Milvus' COSINE score
                        "kg_score": chunk_scores[row["id"]], "found_by_graph": True,
                    })
        return out

    def _load_chunks(self, chunk_ids):
        with self.driver.session() as session:
            return session.run(
                """
                MATCH (c:Chunk) WHERE c.chunk_id IN $ids
                RETURN c.chunk_id AS id, c.content AS content, c.source AS source,
                       c.context_header AS header
                """,
                ids=list(chunk_ids),
            ).data()

    def named_keys(self, query, corpus_id, file_ids=None):
        """Keys of the entities the query names (the same test search() starts from)."""
        return named_entities(query, self._name_candidates(corpus_id, file_ids, query))

    def existing_entities(self, corpus_id, keys, file_ids=None):
        """The given keys that are entities with facts in the selected files (keys sent back by
        a client from an earlier turn may be gone or from another workspace)."""
        if not keys:
            return []
        with self.driver.session() as session:
            rows = session.run(
                """
                UNWIND $keys AS k
                MATCH (e:Entity {corpus_id: $corpus_id, key: k})
                WHERE EXISTS { MATCH (e)-[f:FACT]-() WHERE $all OR f.file_id IN $file_ids }
                RETURN e.key AS key
                """,
                keys=list(keys), corpus_id=corpus_id, file_ids=file_ids or [], all=not file_ids,
            ).data()
        found = {r["key"] for r in rows}
        return [k for k in keys if k in found]

    def search(self, query, query_vector, corpus_id, file_ids=None, vector_chunk_ids=(), max_new_chunks=5,
               carried=None):
        """carried: {entity key: weight} from earlier turns of a chat (conversation.py). They
        start the walk like the question's own entities, at their lower weight, but don't
        count as named (no bridge bonus, not in named_keys)."""
        empty = {"seeds": [], "paths": [], "chunk_scores": {}, "chunks": [], "relations": []}

        # A merged entity is found by any of its names (entity_resolution.py)
        candidates = self._name_candidates(corpus_id, file_ids, query)
        seeds = {key: NAMED_SEED_WEIGHT for key in named_entities(query, candidates)}
        named = set(seeds)
        for key, weight in (carried or {}).items():
            seeds.setdefault(key, weight)
        context_chunks = list(vector_chunk_ids)[:CONTEXT_SEED_CHUNKS]
        for key in self._context_entities(corpus_id, file_ids, context_chunks):
            seeds.setdefault(key, CONTEXT_SEED_WEIGHT)
        if not seeds:
            return empty

        labels = {key: names[0] for key, names in candidates.items()}
        by_entity = {}  # entity key -> facts touching it, loaded as the walk reaches it

        def remember(facts_by_key):
            for key, facts in facts_by_key.items():
                by_entity[key] = facts
                for f in facts:
                    labels.setdefault(f["source"], f["source_label"])
                    labels.setdefault(f["target"], f["target_label"])

        # Every 1- and 2-fact path from a start entity; a path is (start, [facts], end)
        groups = {}
        paths = self._walk(corpus_id, file_ids, sorted(seeds, key=lambda k: -seeds[k]), by_entity, groups, remember)
        relevance = self._relevance([f for _, path_facts, _ in paths for f in path_facts], query_vector)

        def names_other(seed, entity):
            """Whether an entity on the path is (or shares a name with) another named entity."""
            return any(n not in groups.get(seed, {seed}) and (n == entity or n in groups.get(entity, ()))
                       for n in named)

        scored = []
        for seed, path_facts, end in paths:
            score = seeds[seed] * float(np.mean([relevance[f["id"]] for f in path_facts]))
            if len(path_facts) == 2:
                score *= TWO_HOP_FACTOR
            on_path = {f["source"] for f in path_facts} | {f["target"] for f in path_facts} | {end}
            if seed in named and any(names_other(seed, e) for e in on_path):
                score += BRIDGE_BONUS  # connects two things the question names
            scored.append((score, seed, path_facts))
        scored.sort(key=lambda s: -s[0])

        chosen, used = [], set()
        for score, seed, path_facts in scored:
            if score < MIN_PATH_SCORE or len(chosen) >= MAX_PATHS:
                break
            key = frozenset(f["id"] for f in path_facts)
            if key <= used:
                continue  # every fact already shown by a better path
            used |= key
            chosen.append({"score": round(score, 4), "start": seed, "facts": path_facts})

        chunk_scores = {}
        relations = []
        for path in chosen:
            for f in path["facts"]:
                chunk_scores[f["chunk_id"]] = max(chunk_scores.get(f["chunk_id"], 0.0), path["score"])
            first, last = path["facts"][0]["chunk_id"], path["facts"][-1]["chunk_id"]
            if first != last:
                relations.append({"source": first, "target": last, "relation": "KG_PATH", "weight": path["score"]})

        new_chunks = self._new_chunks(chunk_scores, vector_chunk_ids, query_vector, max_new_chunks)

        logger.info(
            "Knowledge graph: %d start entit%s (%d named), %d path(s), %d chunk(s) added",
            len(seeds), "y" if len(seeds) == 1 else "ies", len(named), len(chosen), len(new_chunks),
        )
        return {
            "seeds": [labels[k] for k in seeds if k in named],
            "named_keys": sorted(named),  # start nodes for local PPR (local_ppr.py)
            "paths": [
                {"score": p["score"], "facts": [_public(f) for f in p["facts"]]}
                for p in chosen
            ],
            "chunk_scores": chunk_scores,
            "chunks": new_chunks,
            "relations": relations,
        }


def _public(fact):
    return {k: fact[k] for k in (
        "source", "source_label", "relation", "target", "target_label",
        "chunk_id", "file_id", "filename", "evidence",
    )}
