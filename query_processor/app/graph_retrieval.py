"""
Knowledge-graph retrieval for upload workspaces.

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
"""
import logging
from collections import defaultdict

import numpy as np

from document_text import display_name
from knowledge_graph import _WORD, _content_words, _word_matches

logger = logging.getLogger(__name__)

MAX_FACTS_LOADED = 20000      # per search; workspaces are far smaller in practice
MAX_CANDIDATE_PATHS = 600     # caps embedding work on dense graphs
MAX_PATHS = 6
MIN_PATH_SCORE = 0.3          # cosine-based; unrelated facts score ~0.0-0.2 with MiniLM
NAMED_SEED_WEIGHT = 1.0
CONTEXT_SEED_WEIGHT = 0.7
CONTEXT_SEED_CHUNKS = 3       # top vector chunks whose facts seed the walk
TWO_HOP_FACTOR = 0.9
BRIDGE_BONUS = 0.15


def _unit(vectors):
    vectors = np.asarray(vectors, dtype=np.float32)
    return vectors / (np.linalg.norm(vectors, axis=-1, keepdims=True) + 1e-9)


def fact_text(fact):
    return f"{fact['source_label']} {fact['relation']} {fact['target_label']}"


def named_entities(query, labels):
    """Entity keys whose every content word appears in the query (loose stem match)."""
    query_words = {w.lower().replace("’", "'") for w in _WORD.findall(query)}
    found = []
    for key, label in labels.items():
        words = _content_words(label)
        if words and all(_word_matches(w, query_words) for w in words):
            found.append(key)
    # Prefer the most specific: drop "hollywood" when "hollywood films" also matched
    return [k for k in found if not any(o != k and f" {k} " in f" {o} " for o in found)]


class GraphRetriever:
    def __init__(self, neo4j_driver, embedding_service):
        self.driver = neo4j_driver
        self.embedding_service = embedding_service

    def _load_facts(self, corpus_id, file_ids):
        with self.driver.session() as session:
            rows = session.run(
                """
                MATCH (a:Entity)-[f:FACT {corpus_id: $corpus_id}]->(b:Entity)
                WHERE size($file_ids) = 0 OR f.file_id IN $file_ids
                OPTIONAL MATCH (c:Chunk {chunk_id: f.chunk_id})
                RETURN a.key AS source, a.name AS source_label, b.key AS target, b.name AS target_label,
                       f.relation AS relation, f.chunk_id AS chunk_id, f.file_id AS file_id,
                       f.evidence AS evidence, c.source AS uri
                LIMIT $limit
                """,
                corpus_id=corpus_id, file_ids=file_ids or [], limit=MAX_FACTS_LOADED,
            ).data()
        facts = []
        for i, row in enumerate(rows):
            row["id"] = i
            row["filename"] = display_name((row.pop("uri") or "").rsplit("/", 1)[-1]) or "Unknown file"
            facts.append(row)
        return facts

    def _load_chunks(self, chunk_ids):
        with self.driver.session() as session:
            return session.run(
                """
                MATCH (c:Chunk) WHERE c.chunk_id IN $ids
                RETURN c.chunk_id AS id, c.content AS content, c.source AS source
                """,
                ids=list(chunk_ids),
            ).data()

    def search(self, query, query_vector, corpus_id, file_ids=None, vector_chunk_ids=(), max_new_chunks=5):
        empty = {"seeds": [], "paths": [], "chunk_scores": {}, "chunks": [], "relations": []}
        facts = self._load_facts(corpus_id, file_ids)
        if not facts:
            return empty

        labels, by_entity = {}, defaultdict(list)
        for f in facts:
            labels.setdefault(f["source"], f["source_label"])
            labels.setdefault(f["target"], f["target_label"])
            by_entity[f["source"]].append(f)
            by_entity[f["target"]].append(f)

        seeds = {key: NAMED_SEED_WEIGHT for key in named_entities(query, labels)}
        named = set(seeds)
        context_chunks = set(list(vector_chunk_ids)[:CONTEXT_SEED_CHUNKS])
        for f in facts:
            if f["chunk_id"] in context_chunks:
                for key in (f["source"], f["target"]):
                    seeds.setdefault(key, CONTEXT_SEED_WEIGHT)
        if not seeds:
            return empty

        # Every 1- and 2-fact path from a start entity; a path is (start, [facts], end)
        paths = []
        for seed in sorted(seeds, key=lambda k: -seeds[k]):
            for f in by_entity[seed]:
                middle = f["target"] if f["source"] == seed else f["source"]
                paths.append((seed, [f], middle))
                for g in by_entity[middle]:
                    if g["id"] == f["id"]:
                        continue
                    end = g["target"] if g["source"] == middle else g["source"]
                    if end != seed:
                        paths.append((seed, [f, g], end))
            if len(paths) >= MAX_CANDIDATE_PATHS:
                break
        paths = paths[:MAX_CANDIDATE_PATHS]

        unique = {f["id"]: f for _, path_facts, _ in paths for f in path_facts}
        ids = list(unique)
        vectors = _unit(self.embedding_service.generate_batch_embeddings([fact_text(unique[i]) for i in ids]))
        q = _unit(query_vector)
        relevance = dict(zip(ids, (vectors @ q).tolist()))

        scored = []
        for seed, path_facts, end in paths:
            score = seeds[seed] * float(np.mean([relevance[f["id"]] for f in path_facts]))
            if len(path_facts) == 2:
                score *= TWO_HOP_FACTOR
            if seed in named and end in named:
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

        known = set(vector_chunk_ids)
        new_ids = [c for c in sorted(chunk_scores, key=lambda c: -chunk_scores[c]) if c not in known][:max_new_chunks]
        new_chunks = []
        if new_ids:
            rows = [r for r in self._load_chunks(new_ids) if r.get("content")]
            if rows:
                sims = _unit(self.embedding_service.generate_batch_embeddings([r["content"] for r in rows])) @ q
                for row, sim in zip(rows, sims.tolist()):
                    new_chunks.append({
                        "id": row["id"], "content": row["content"], "source": row["source"],
                        "score": float(sim),  # cosine, same scale as Milvus' COSINE score
                        "kg_score": chunk_scores[row["id"]], "found_by_graph": True,
                    })

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
