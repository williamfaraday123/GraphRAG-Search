"""
Entity resolution: one node per concept in a workspace's knowledge graph.

The extractor names the same thing in different words across sentences and files
("better-constructed narratives" / "better constructed narratives", "growing audience" /
"increasing audiences"). Separate nodes break fact walks, PPR and coverage at the gap and
make the corpus map and file similarity count one concept several times.

An entity joins an existing one (its "canonical" entity) when either:
  1. Same wording once normalised: case, hyphens, plurals, articles, prepositions and
     quantifiers ("few", "both") ignored. "few film industries" = "film industries".
  2. Close meaning AND the same main noun: MiniLM cosine >= EMBED_MATCH and equal head
     words. "growing audience" ~ "increasing audiences" (0.85, both "audience") joins;
     "Hollywood" ~ "Hollywood films" (0.81, "hollywood" vs "film") doesn't.
Each candidate is compared with canonical entities only, never with other members, so
groups can't drift by chaining (A~B, B~C putting A with C).

Calibrated on this project's workspaces: correct and wrong pairs overlap between 0.78
and 0.85 cosine on names alone (Hollywood ~ Hollywood films 0.81; Europe ~ Germany 0.72),
hence the head-word condition. Aliases of different wording (United States ~ Americans,
0.62, below Germany ~ Japan at 0.68) are out of reach of name similarity and stay separate.

Merging happens when facts are written (knowledge_graph.py), so duplicates aren't
created; `merge_existing` folds duplicates already in Neo4j. A merged entity's name is
kept in the canonical node's `aliases` list.
"""
import logging
import re
import threading

import numpy as np

logger = logging.getLogger(__name__)

EMBED_MATCH = 0.84
_WORD = re.compile(r"[a-z0-9']+")
_IGNORED = {
    "the", "a", "an", "their", "its", "his", "her", "our", "this", "these", "those", "that",
    "in", "on", "over", "by", "for", "with", "from", "of", "to", "at", "as", "into", "and", "or",
    "few", "many", "several", "some", "both", "all", "each", "various", "certain",
}
_HEAD_BREAK = {"of", "for", "from", "in", "on", "with", "to", "at", "by", "into", "about", "between"}


def _singular(word):
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def _tokens(name):
    return _WORD.findall(name.lower().replace("’", "'").replace("-", " ").replace("'s", ""))


def normal_form(name):
    """'The better-constructed narratives' -> 'better constructed narrative'."""
    return " ".join(_singular(w) for w in _tokens(name) if w not in _IGNORED)


def head_word(name):
    """Main noun: the last word before the first preposition, else the last word.
    'largest single market for films' -> 'market'; 'Hollywood films' -> 'film'."""
    all_words = _tokens(name)
    words = all_words
    for i, w in enumerate(all_words):
        if w in _HEAD_BREAK and i > 0:
            words = all_words[:i]
            break
    words = [w for w in words if w not in _IGNORED] or [w for w in all_words if w not in _IGNORED]
    return _singular(words[-1]) if words else ""


def _unit(vectors):
    v = np.asarray(vectors, dtype=np.float32)
    if v.ndim == 1:
        v = v[None, :]
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)


class _Workspace:
    """Canonical entities of one workspace, in memory. The vector matrix grows by doubling,
    so adding stays cheap at the wiki's tens of thousands of entities."""

    def __init__(self):
        self.forms = {}       # normal form -> canonical key
        self.keys = []        # canonical keys, aligned with heads / vector rows
        self.key_set = set()
        self.heads = []
        self._matrix = None   # unit vectors of canonical names; rows [0, len(keys)) are used

    @property
    def vectors(self):
        return None if self._matrix is None else self._matrix[:len(self.keys)]

    def add(self, key, name, vector):
        self.forms.setdefault(normal_form(name), key)
        n = len(self.keys)
        if self._matrix is None:
            self._matrix = np.empty((64, vector.shape[1]), dtype=np.float32)
        elif n == len(self._matrix):
            grown = np.empty((2 * n, self._matrix.shape[1]), dtype=np.float32)
            grown[:n] = self._matrix
            self._matrix = grown
        self._matrix[n] = vector[0]
        self.keys.append(key)
        self.key_set.add(key)
        self.heads.append(head_word(name))

    def match(self, name, vector):
        key = self.forms.get(normal_form(name))
        if key or self.vectors is None:
            return key
        head = head_word(name)
        sims = self.vectors @ vector[0]
        for i in np.argsort(-sims):
            if sims[i] < EMBED_MATCH:
                break
            if head and self.heads[i] == head:
                return self.keys[i]
        return None


class EntityResolver:
    def __init__(self, driver, embed):
        """embed(list of texts) -> list of vectors (EmbeddingService.generate_batch_embeddings)."""
        self.driver = driver
        self.embed = embed
        self._workspaces = {}
        self._lock = threading.Lock()

    def _load(self, corpus_id):
        ws = self._workspaces.get(corpus_id)
        if ws is not None:
            return ws
        with self.driver.session() as session:
            rows = session.run(
                "MATCH (e:Entity {corpus_id: $corpus_id}) RETURN e.key AS key, e.name AS name ORDER BY e.key",
                corpus_id=corpus_id,
            ).data()
        ws = _Workspace()
        if rows:
            vectors = _unit(self.embed([r["name"] or r["key"] for r in rows]))
            for row, vec in zip(rows, vectors):
                ws.add(row["key"], row["name"] or row["key"], vec[None, :])
        self._workspaces[corpus_id] = ws
        return ws

    def forget(self, corpus_id):
        """Call after entities are deleted (file removed) so the cache is rebuilt."""
        with self._lock:
            self._workspaces.pop(corpus_id, None)

    def resolve(self, corpus_id, entities):
        """{key: canonical key} for one chunk's entities ([{'key', 'name', ...}]); new
        concepts become canonical themselves."""
        with self._lock:
            ws = self._load(corpus_id)
            mapping = {}
            fresh = [e for e in entities if e["key"] not in ws.key_set]
            for e in entities:
                if e["key"] in ws.key_set:
                    mapping[e["key"]] = e["key"]
            if fresh:
                vectors = _unit(self.embed([e["name"] for e in fresh]))
                for e, vec in zip(fresh, vectors):
                    vec = vec[None, :]
                    canonical = ws.match(e["name"], vec)
                    if canonical is None:
                        ws.add(e["key"], e["name"], vec)
                        canonical = e["key"]
                    mapping[e["key"]] = canonical
            return mapping

    def merge_existing(self, corpus_id, dry_run=False):
        """Folds duplicate entities already stored for a workspace into one node each.
        The most-mentioned entity of a group stays canonical. Returns [(duplicate name,
        canonical name)]."""
        with self.driver.session() as session:
            rows = session.run(
                """
                MATCH (e:Entity {corpus_id: $corpus_id})
                RETURN e.key AS key, e.name AS name,
                       COUNT { (e)<-[:MENTIONS]-() } AS mentions, COUNT { (e)-[:FACT]-() } AS facts
                ORDER BY mentions DESC, facts DESC, size(e.name), e.key
                """,
                corpus_id=corpus_id,
            ).data()
        if not rows:
            return []
        vectors = _unit(self.embed([r["name"] or r["key"] for r in rows]))
        ws, merges = _Workspace(), []
        names = {r["key"]: r["name"] or r["key"] for r in rows}
        for row, vec in zip(rows, vectors):
            vec = vec[None, :]
            canonical = ws.match(names[row["key"]], vec)
            if canonical is None:
                ws.add(row["key"], names[row["key"]], vec)
            else:
                merges.append((row["key"], canonical))
        if not dry_run:
            with self.driver.session() as session:
                for duplicate, canonical in merges:
                    session.execute_write(_merge_into, corpus_id, duplicate, canonical, names[duplicate])
            self.forget(corpus_id)
        if merges:
            logger.info("Entity resolution: %s %d duplicate entit%s in workspace %s",
                        "would merge" if dry_run else "merged", len(merges),
                        "y" if len(merges) == 1 else "ies", corpus_id)
        return [(names[d], names[c]) for d, c in merges]


def _merge_into(tx, corpus_id, duplicate, canonical, alias):
    """Moves a duplicate entity's mentions and facts onto the canonical node, then deletes it.
    Facts between the two would become self-loops and are dropped."""
    params = {"corpus_id": corpus_id, "dup": duplicate, "canon": canonical}
    tx.run(
        """
        MATCH (d:Entity {corpus_id: $corpus_id, key: $dup}), (k:Entity {corpus_id: $corpus_id, key: $canon})
        MATCH (c:Chunk)-[:MENTIONS]->(d)
        MERGE (c)-[:MENTIONS]->(k)
        """, **params).consume()
    tx.run(
        """
        MATCH (d:Entity {corpus_id: $corpus_id, key: $dup}), (k:Entity {corpus_id: $corpus_id, key: $canon})
        MATCH (d)-[f:FACT]->(o:Entity) WHERE o <> k AND o <> d
        CREATE (k)-[g:FACT]->(o) SET g = properties(f)
        """, **params).consume()
    tx.run(
        """
        MATCH (d:Entity {corpus_id: $corpus_id, key: $dup}), (k:Entity {corpus_id: $corpus_id, key: $canon})
        MATCH (o:Entity)-[f:FACT]->(d) WHERE o <> k AND o <> d
        CREATE (o)-[g:FACT]->(k) SET g = properties(f)
        """, **params).consume()
    tx.run(
        """
        MATCH (d:Entity {corpus_id: $corpus_id, key: $dup}), (k:Entity {corpus_id: $corpus_id, key: $canon})
        SET k.aliases = CASE WHEN $alias IN coalesce(k.aliases, []) THEN k.aliases
                             ELSE coalesce(k.aliases, []) + $alias END
        DETACH DELETE d
        """, alias=alias, **params).consume()
