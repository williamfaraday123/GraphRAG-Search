"""
Which files in an upload workspace are most alike, measured on the knowledge graph.

Primary measure: graph overlap. Each file's entities (Chunk -MENTIONS-> Entity) and facts
(Entity -FACT {file_id}-> Entity) are compared with every other file's:

  1. Fuzzy matching: an entity or fact in one file matches one in another when their MiniLM
     embeddings are close (cosine >= ENTITY_MATCH / FACT_MATCH; a fact is compared both as
     its triple and as the sentence it was extracted from), so a paraphrase such as "rest
     of the world survived" / "remainder of the globe persevered" still counts as shared.
  2. IDF weighting: an item found in every file (e.g. "Hollywood" in a workspace about film)
     says little about which two files are alike, so each item weighs ln(1 + N / df), where
     df is the number of files it matches in.
  3. Pair score: weighted Dice, i.e. the IDF weight of both files' matched items over the
     IDF weight of all their items (0 = nothing shared, 1 = everything shared).
     Graph score = FACT_SHARE x fact score + (1 - FACT_SHARE) x entity score.

Second opinion: cosine similarity of the files' mean chunk embeddings, i.e. how alike the
texts read overall, independent of what the extractor picked up.
"""
import logging
from collections import defaultdict
from itertools import combinations

import numpy as np

from document_text import display_name
from graph_retrieval import fact_text

logger = logging.getLogger(__name__)

ENTITY_MATCH = 0.80      # MiniLM cosine for two entity names to count as the same thing
# ... and for two facts to count as the same claim (on the triple or its source sentence).
# Calibrated on paraphrased film-history files: real paraphrases scored 0.74-0.76, unrelated
# facts about the same entity 0.60-0.70
FACT_MATCH = 0.72
FACT_SHARE = 0.6         # weight of shared facts vs shared entities in the graph score
MAX_SHARED_SHOWN = 8     # shared facts returned per pair
MAX_CHUNKS_PER_FILE = 200
MAX_ITEMS = 5000         # caps embedding work per comparison


def _unit(vectors):
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim == 1:
        vectors = vectors[None, :]
    return vectors / (np.linalg.norm(vectors, axis=-1, keepdims=True) + 1e-9)


class FileSimilarity:
    def __init__(self, neo4j_driver, embedding_service):
        self.driver = neo4j_driver
        self.embedding_service = embedding_service

    # --- loading ---------------------------------------------------------------------------

    def _load(self, corpus_id, file_ids):
        with self.driver.session() as session:
            chunks = session.run(
                """
                MATCH (c:Chunk {corpus_id: $corpus_id})
                WHERE size($file_ids) = 0 OR c.file_id IN $file_ids
                RETURN c.file_id AS file_id, c.chunk_id AS chunk_id, c.source AS source, c.content AS content
                """,
                corpus_id=corpus_id, file_ids=file_ids or [],
            ).data()
            entities = session.run(
                """
                MATCH (c:Chunk {corpus_id: $corpus_id})-[:MENTIONS]->(e:Entity)
                WHERE size($file_ids) = 0 OR c.file_id IN $file_ids
                RETURN DISTINCT c.file_id AS file_id, e.key AS key, e.name AS name
                """,
                corpus_id=corpus_id, file_ids=file_ids or [],
            ).data()
            facts = session.run(
                """
                MATCH (a:Entity)-[f:FACT {corpus_id: $corpus_id}]->(b:Entity)
                WHERE size($file_ids) = 0 OR f.file_id IN $file_ids
                RETURN a.key AS source, a.name AS source_label, b.key AS target, b.name AS target_label,
                       f.relation AS relation, f.chunk_id AS chunk_id, f.file_id AS file_id,
                       f.evidence AS evidence
                """,
                corpus_id=corpus_id, file_ids=file_ids or [],
            ).data()
        return chunks, entities, facts

    # --- scoring ---------------------------------------------------------------------------

    @staticmethod
    def _cross_matches(order, counts, sim, threshold):
        """best[f][i, k]: item i of file f's closest similarity to any item of file order[k]
        (1 for its own file), and each item's IDF over the files it matches in."""
        best = {}
        for f in order:
            cols = []
            for g in order:
                if g == f:
                    cols.append(np.ones(counts[f], dtype=np.float32))
                elif counts[f] and counts[g]:
                    cols.append(sim(f, g).max(axis=1))
                else:
                    cols.append(np.zeros(counts[f], dtype=np.float32))
            best[f] = np.stack(cols, axis=1) if counts[f] else np.zeros((0, len(order)), dtype=np.float32)
        idf = {}
        n = len(order)
        for f in order:
            df = (best[f] >= threshold).sum(axis=1) if len(best[f]) else np.zeros(0)
            idf[f] = np.log1p(n / np.maximum(df, 1))
        return order, best, idf

    @staticmethod
    def _pair_score(a, b, order, best, idf, threshold):
        ia, ib = order.index(a), order.index(b)
        total = idf[a].sum() + idf[b].sum()
        if total == 0:
            return 0.0, 0.0, 0.0
        shared_a = idf[a][best[a][:, ib] >= threshold].sum() if len(best[a]) else 0.0
        shared_b = idf[b][best[b][:, ia] >= threshold].sum() if len(best[b]) else 0.0
        cover_a = shared_a / idf[a].sum() if idf[a].sum() else 0.0   # share of a's content found in b
        cover_b = shared_b / idf[b].sum() if idf[b].sum() else 0.0
        return float((shared_a + shared_b) / total), float(cover_a), float(cover_b)

    def compare(self, corpus_id, file_ids=None):
        """{'files': [...], 'pairs': [...] ranked by graph score, best first}, or None when the
        selection has fewer than two indexed files."""
        chunks, entity_rows, fact_rows = self._load(corpus_id, file_ids)

        names, texts = {}, defaultdict(list)
        for c in chunks:
            names.setdefault(c["file_id"], display_name((c.get("source") or "").rsplit("/", 1)[-1]) or "Unknown file")
            if c.get("content") and len(texts[c["file_id"]]) < MAX_CHUNKS_PER_FILE:
                texts[c["file_id"]].append(c["content"])
        file_order = sorted(names, key=lambda f: names[f].lower())
        if len(file_order) < 2:
            return None

        # Entities per file (distinct keys); facts per file (distinct claims)
        entities = defaultdict(dict)
        for row in entity_rows:
            if row["file_id"] in names:
                entities[row["file_id"]][row["key"]] = row["name"] or row["key"]
        facts = defaultdict(dict)
        for row in fact_rows:
            if row["file_id"] in names:
                row["filename"] = names[row["file_id"]]
                facts[row["file_id"]].setdefault((row["source"], row["relation"], row["target"]), row)

        def embed_groups(groups, to_text):
            flat = [(f, item) for f in file_order for item in groups[f]][:MAX_ITEMS]
            vectors = _unit(self.embedding_service.generate_batch_embeddings([to_text(i) for _, i in flat])) if flat else None
            out, items = {}, {}
            for f in file_order:
                idx = [k for k, (g, _) in enumerate(flat) if g == f]
                items[f] = [flat[k][1] for k in idx]
                out[f] = vectors[idx] if idx else np.zeros((0, 1), dtype=np.float32)
            return out, items

        ent_vecs, ent_items = embed_groups({f: list(entities[f].values()) for f in file_order}, lambda name: name)
        fact_groups = {f: list(facts[f].values()) for f in file_order}
        fact_vecs, fact_items = embed_groups(fact_groups, fact_text)
        sent_vecs, _ = embed_groups(fact_groups, lambda row: row.get("evidence") or fact_text(row))

        def entity_sim(f, g):
            return ent_vecs[f] @ ent_vecs[g].T

        def fact_sim(f, g):
            # A paraphrase can change the extracted triple a lot ("rest of the world survived" /
            # "remainder of the globe persevered") while the sentence it came from stays close,
            # and vice versa, so two facts match on whichever is closer
            return np.maximum(fact_vecs[f] @ fact_vecs[g].T, sent_vecs[f] @ sent_vecs[g].T)

        e_counts = {f: len(ent_items[f]) for f in file_order}
        f_counts = {f: len(fact_items[f]) for f in file_order}
        e_order, e_best, e_idf = self._cross_matches(file_order, e_counts, entity_sim, ENTITY_MATCH)
        f_order, f_best, f_idf = self._cross_matches(file_order, f_counts, fact_sim, FACT_MATCH)

        # Second opinion: mean chunk embedding per file
        file_vecs = {f: _unit(np.mean(self.embedding_service.generate_batch_embeddings(texts[f]), axis=0))[0]
                     for f in file_order if texts[f]}

        pairs = []
        for a, b in combinations(file_order, 2):
            ent_score, _, _ = self._pair_score(a, b, e_order, e_best, e_idf, ENTITY_MATCH)
            fact_score, cover_a, cover_b = self._pair_score(a, b, f_order, f_best, f_idf, FACT_MATCH)
            has_facts = len(fact_items[a]) and len(fact_items[b])
            graph = FACT_SHARE * fact_score + (1 - FACT_SHARE) * ent_score if has_facts else ent_score
            embedding = float(file_vecs[a] @ file_vecs[b]) if a in file_vecs and b in file_vecs else 0.0
            pairs.append({
                "a": a, "b": b, "a_name": names[a], "b_name": names[b],
                "graph": round(graph, 4), "facts": round(fact_score, 4), "entities": round(ent_score, 4),
                "embedding": round(embedding, 4),
                "a_covered": round(cover_a, 4), "b_covered": round(cover_b, 4),
                "shared_facts": self._shared_facts(a, b, fact_sim, fact_items, f_idf),
                "shared_entities": self._shared_entities(a, b, entity_sim, ent_items, e_idf),
            })
        pairs.sort(key=lambda p: (-p["graph"], -p["embedding"]))
        for rank, p in enumerate(sorted(pairs, key=lambda p: -p["embedding"]), start=1):
            p["embedding_rank"] = rank

        def only_here(f):
            """Facts of f that match nothing in any other file."""
            if not f_counts[f]:
                return []
            others = np.delete(f_best[f], f_order.index(f), axis=1)
            return [fact_items[f][i] for i in range(f_counts[f]) if others.max(axis=1)[i] < FACT_MATCH][:MAX_SHARED_SHOWN]

        files = [{"file_id": f, "filename": names[f], "facts": len(fact_items[f]), "entities": len(ent_items[f]),
                  "only_here": only_here(f)}
                 for f in file_order]
        logger.info("File similarity: %d files, best pair %s / %s (graph %.2f, embedding %.2f)",
                    len(files), pairs[0]["a_name"], pairs[0]["b_name"], pairs[0]["graph"], pairs[0]["embedding"])
        return {"files": files, "pairs": pairs}

    @staticmethod
    def _shared_facts(a, b, sim, items, idf):
        """Matched fact pairs (fact in a, closest fact in b), most distinctive first."""
        if not len(items[a]) or not len(items[b]):
            return []
        sims = sim(a, b)
        found, used = [], set()
        for i in np.argsort(-(idf[a] * sims.max(axis=1))):
            j = int(sims[i].argmax())
            if sims[i, j] < FACT_MATCH or j in used:
                continue
            used.add(j)
            found.append({"similarity": round(float(sims[i, j]), 4), "weight": round(float(idf[a][i]), 4),
                          "a": items[a][i], "b": items[b][j]})
            if len(found) >= MAX_SHARED_SHOWN:
                break
        return found

    @staticmethod
    def _shared_entities(a, b, sim, items, idf):
        if not len(items[a]) or not len(items[b]):
            return []
        sims = sim(a, b)
        best = sims.max(axis=1)
        order = [i for i in np.argsort(-(idf[a] * best)) if best[i] >= ENTITY_MATCH]
        return [items[a][i] for i in order[:12]]


def similarity_table(result) -> str:
    """Markdown table of every pair, best first: appended to the answer by the server, so the
    numbers are the computed ones, not the model's retelling of them."""
    lines = ["| Files | Graph overlap | Shared facts | Shared entities | Text similarity |",
             "|---|---|---|---|---|"]
    for p in result["pairs"]:
        lines.append(f"| {p['a_name']} + {p['b_name']} | {p['graph']:.0%} | {p['facts']:.0%} | "
                     f"{p['entities']:.0%} | {p['embedding']:.0%} |")
    return "\n".join(lines)
