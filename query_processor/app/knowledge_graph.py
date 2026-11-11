"""
LLM-built knowledge graph for uploaded files.

After a file is indexed, a background worker sends each sentence of each chunk to a
small local model (Ollama, KG_MODEL, default qwen2.5:1.5b) and asks for the facts it
states as (subject, relation, object) triples. One sentence per call: on whole chunks a
1.5B model often loops, leaves objects empty or mixes facts from different sentences.

Each triple is checked against its sentence before it is kept: the words of subject and
object must appear in that sentence, which is stored as the fact's evidence. That removes
most invented facts and facts carried over from the previous sentence.

Stored in Neo4j next to the existing chunk graph:
  (:Entity {corpus_id, key, name, kind})          kind: "name" (capitalised) or "concept"
  (:Chunk)-[:MENTIONS]->(:Entity)
  (:Entity)-[:FACT {corpus_id, file_id, chunk_id, relation, evidence}]->(:Entity)
  Chunk.kg_status: null (pending) | "done" | "failed"

Retrieval matches nodes by chunk_id, which entities don't have, so they never enter
search results. A search stops the sentence being extracted and extraction waits until
no search is running, so the two don't compete for the same CPU-bound model; the stopped
sentence is redone from the start. Expect roughly 4 s per sentence on CPU.

Configuration:
  KG_ENABLED        true | false                         (default: true)
  KG_MODEL          Ollama model for extraction          (default: qwen2.5:1.5b)
  KG_OLLAMA_URL     Ollama base URL; defaults to EDGE_MODEL_BASE_URL without "/v1"
"""
import http.client
import json
import logging
import os
import queue
import re
import socket
import threading
import time
import urllib.parse

from corpus_map import split_sentences

logger = logging.getLogger(__name__)

MAX_FACTS_PER_SENTENCE = 3
MIN_SENTENCE_WORDS = 4
REQUEST_TIMEOUT_SECONDS = 120
RETRY_DELAY_SECONDS = 60  # after Ollama is unreachable

PROMPT = """Extract the facts stated in this sentence.

Sentence: "{sentence}"

For each fact give:
- subject: who or what the fact is about (a short noun phrase from the sentence)
- relation: what the subject does, is, or has (a short verb phrase, e.g. "dominated", "was weakened by", "had")
- object: the thing the relation points to (a short noun phrase from the sentence, different from the subject)

Give at most 3 facts. Resolve "it", "they", "their" to the noun they refer to, using this earlier text if needed: "{previous}"
Return {{"facts": []}} if the sentence states no fact."""

SCHEMA = {
    "type": "object",
    "properties": {
        "facts": {
            "type": "array",
            "maxItems": MAX_FACTS_PER_SENTENCE,
            "items": {
                "type": "object",
                # maxLength stops the model looping inside one string ("X and X and X ...")
                "properties": {
                    "subject": {"type": "string", "maxLength": 60},
                    "relation": {"type": "string", "maxLength": 40},
                    "object": {"type": "string", "maxLength": 60},
                },
                "required": ["subject", "relation", "object"],
            },
        }
    },
    "required": ["facts"],
}

DETERMINERS = {"the", "a", "an", "their", "its", "his", "her", "our", "this", "these", "those", "that"}
LEADING_PREPOSITIONS = {
    "in", "on", "over", "by", "for", "with", "from", "of", "to", "at", "as", "into", "since",
    "during", "after", "before", "about", "through", "under", "between", "within", "without",
    "against", "among", "around", "across", "toward", "towards", "upon", "via",
}
# An entity starting with these is a description ("more impressive"), not a thing
NOT_A_THING = {"more", "most", "less", "least", "very", "much", "since", "because"}
# Too generic to be useful nodes on their own (pronouns the model failed to resolve, too)
GENERIC = {
    "film", "films", "cinema", "movie", "movies", "people", "world", "time", "thing", "things",
    "it", "its", "they", "them", "their", "he", "she", "his", "her", "we", "us", "you", "one",
    "others", "something", "position", "event", "situation",
}
MINOR_WORDS = DETERMINERS | LEADING_PREPOSITIONS | {
    "and", "or", "is", "are", "was", "were", "be", "new", "own", "which", "who", "whose",
}
_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9'’\-]*")
_LIST_SPLIT = re.compile(r",\s*(?:and\s+|or\s+)?|\s+and\s+|\s+or\s+")


# --- Validating model output -------------------------------------------------

def _clean_phrase(text):
    text = str(text or "").replace("’", "'")
    text = re.sub(r"\(\s*\d{3,4}\s*\)", " ", text)               # "(1913)"
    text = re.sub(r"(?<![A-Za-z])'|'(?![A-Za-z])", " ", text)   # quotes, keeping "world's"
    text = " ".join(text.split()).strip(" .,;:!?\"()[]")
    words = text.split()
    # "in conditions of commercial isolation" -> "conditions of commercial isolation"
    while words and words[0].lower() in DETERMINERS | LEADING_PREPOSITIONS:
        words = words[1:]
    return " ".join(words)


def entity_key(name):
    return name.lower()


def _content_words(phrase):
    return [w.lower() for w in _WORD.findall(phrase) if w.lower() not in MINOR_WORDS and len(w) >= 3]


def _word_matches(word, sentence_words):
    # Loose stem match: "impressive" ~ "impression", "markets" ~ "market"
    if word in sentence_words:
        return True
    return len(word) >= 5 and any(len(s) >= 5 and s[:5] == word[:5] for s in sentence_words)


def _in_sentence(phrase, sentence_words):
    words = _content_words(phrase)
    return bool(words) and all(_word_matches(w, sentence_words) for w in words)


def _valid_entity(name):
    words = name.split()
    content = _content_words(name)
    return (
        2 <= len(name) <= 60
        and 1 <= len(words) <= 6
        and name.lower() not in GENERIC
        and bool(content)
        and words[0].lower() not in NOT_A_THING
        and words[-1].lower() not in NOT_A_THING  # cut-off phrases: "special effects were more"
        and not all(w.endswith("ly") for w in content)  # adverbs: "artistically"
    )


def _split_list(phrase):
    """'Soviet Union and Japan' -> ['Soviet Union', 'Japan'] when every part is short."""
    parts = [_clean_phrase(p) for p in _LIST_SPLIT.split(phrase)]
    parts = [p for p in parts if p]
    if len(parts) > 1 and all(len(p.split()) <= 4 and _valid_entity(p) for p in parts):
        return parts
    return [phrase]


def entity_kind(name):
    """'name' when any word is capitalised (Hollywood, First World War), else 'concept'."""
    return "name" if any(w[:1].isupper() for w in name.split()) else "concept"


def validate_facts(sentence, raw_facts):
    """Turns raw model triples for one sentence into checked facts and their entities."""
    sentence_words = {w.lower().replace("’", "'") for w in _WORD.findall(sentence)}
    evidence = " ".join(sentence.split())
    entities, facts, seen = {}, [], set()
    for raw in raw_facts[:MAX_FACTS_PER_SENTENCE]:
        if not isinstance(raw, dict):
            continue
        relation = " ".join(str(raw.get("relation") or "").split()).strip(" .,;:")
        if not relation or len(relation.split()) > 8:
            continue
        if all(w.lower() in DETERMINERS | LEADING_PREPOSITIONS for w in relation.split()):
            continue  # no real word ("since", "by")
        for subject in _split_list(_clean_phrase(raw.get("subject"))):
            for obj in _split_list(_clean_phrase(raw.get("object"))):
                if not (_valid_entity(subject) and _valid_entity(obj)):
                    continue
                if set(_content_words(obj)) <= set(_content_words(relation)):
                    continue  # object repeated inside the relation: "were more impressive | impressive"
                a, b = entity_key(subject), entity_key(obj)
                # Self-loops and near-self-loops ("audiences" -> "popular audiences")
                if a == b or re.search(rf"\b{re.escape(a)}\b", b) or re.search(rf"\b{re.escape(b)}\b", a):
                    continue
                # Both ends must come from this sentence: drops invented objects and facts
                # the model carried over from the previous sentence
                if not (_in_sentence(subject, sentence_words) and _in_sentence(obj, sentence_words)):
                    continue
                rel = relation[0].lower() + relation[1:]
                if (a, b) in seen:
                    continue  # one fact per pair per sentence ("appealed because" / "appealed because of")
                seen.add((a, b))
                entities.setdefault(a, {"key": a, "name": subject, "kind": entity_kind(subject)})
                entities.setdefault(b, {"key": b, "name": obj, "kind": entity_kind(obj)})
                facts.append({"source": a, "target": b, "relation": rel, "evidence": evidence})
    return list(entities.values()), facts


def _squash(text):
    return " ".join(text.split()).lower()


def chunk_sentences(content, previous_content=""):
    """Sentences worth sending to the model: skips fragments and sentences repeated from
    the previous chunk's overlap (they were already processed there)."""
    prior = _squash(previous_content)
    out = []
    for sentence in split_sentences(content):
        if len(_WORD.findall(sentence)) < MIN_SENTENCE_WORDS:
            continue
        if prior and _squash(sentence).rstrip(".!?\"' ") in prior:
            continue
        out.append(sentence)
    return out


def extract_chunk(content, previous_content="", call_model=None):
    """All checked entities and facts for one chunk. Raises ConnectionError if Ollama
    is unreachable. call_model(sentence, previous) replaces extract_facts (the worker
    passes one that yields to searches)."""
    call = call_model or extract_facts
    entities, facts = {}, []
    previous = previous_content.strip()[-300:]
    for sentence in chunk_sentences(content, previous_content):
        found_entities, found_facts = validate_facts(sentence, call(sentence, previous))
        for entity in found_entities:
            entities.setdefault(entity["key"], entity)
        facts.extend(found_facts)
        previous = sentence
    return list(entities.values()), facts


# --- Calling the model --------------------------------------------------------

def _ollama_url():
    explicit = os.getenv("KG_OLLAMA_URL")
    if explicit:
        return explicit.rstrip("/")
    base = os.getenv("EDGE_MODEL_BASE_URL", "http://localhost:11434/v1").rstrip("/")
    return base[:-3] if base.endswith("/v1") else base


def extract_facts(sentence, previous="", on_connect=None):
    """Calls the model for one sentence; returns raw triples.
    Raises ConnectionError if Ollama is unreachable. on_connect(connection) runs once the
    connection is open, so another thread can close it to stop the call midway."""
    body = json.dumps({
        "model": os.getenv("KG_MODEL", "qwen2.5:1.5b"),
        "stream": False,
        "format": SCHEMA,  # Ollama constrains the output to this JSON schema
        # num_predict caps runaway output; repeat_penalty discourages the loops that cause it
        "options": {"temperature": 0, "num_predict": 200, "repeat_penalty": 1.1},
        "messages": [{"role": "user", "content": PROMPT.format(sentence=sentence, previous=previous)}],
    }).encode("utf-8")
    url = urllib.parse.urlsplit(_ollama_url())
    connection_class = http.client.HTTPSConnection if url.scheme == "https" else http.client.HTTPConnection
    connection = connection_class(url.hostname, url.port, timeout=REQUEST_TIMEOUT_SECONDS)
    try:
        connection.connect()
        if on_connect:
            on_connect(connection)
        connection.request("POST", f"{url.path.rstrip('/')}/api/chat", body, {"Content-Type": "application/json"})
        response = connection.getresponse()
        raw = response.read()
        if response.status != 200:
            raise ValueError(f"Ollama returned HTTP {response.status}: {raw[:200]!r}")
        content = json.loads(raw)["message"]["content"]
    except (OSError, http.client.HTTPException) as exc:
        raise ConnectionError(f"Ollama is unreachable at {_ollama_url()}: {exc}") from exc
    finally:
        connection.close()
    try:
        return json.loads(content).get("facts", [])
    except (json.JSONDecodeError, AttributeError):
        # Output cut off at num_predict: keep every triple that was completed before the cut
        return [
            {"subject": s, "relation": r, "object": o}
            for s, r, o in _COMPLETE_TRIPLE.findall(content or "")
        ]


_COMPLETE_TRIPLE = re.compile(
    r'\{\s*"subject"\s*:\s*"((?:[^"\\]|\\.)*)"\s*,\s*"relation"\s*:\s*"((?:[^"\\]|\\.)*)"'
    r'\s*,\s*"object"\s*:\s*"((?:[^"\\]|\\.)*)"\s*\}'
)


# --- Background worker --------------------------------------------------------

class _StoppedForSearch(Exception):
    pass


def _close_now(connection):
    """Unblocks a thread waiting on this connection's response (close() alone may not)."""
    sock = connection.sock
    if sock is not None:
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass

class KnowledgeGraphBuilder:
    def __init__(self, neo4j_driver, resolver=None):
        """resolver: entity_resolution.EntityResolver; entities that name the same concept as
        an existing one are written onto that entity instead of a new node."""
        self.driver = neo4j_driver
        self.resolver = resolver
        self.enabled = os.getenv("KG_ENABLED", "true").lower() not in {"0", "false", "no"}
        self._queue = queue.Queue()
        self._queued = set()
        self._queued_lock = threading.Lock()
        self._searches = 0
        self._search_cv = threading.Condition()
        self._active_call = None   # open connection of the model call in progress
        self._call_stopped = False
        if self.enabled:
            self._ensure_indexes()
            threading.Thread(target=self._run, name="kg-builder", daemon=True).start()
            self.enqueue_backlog()

    # Searches and extraction share one CPU-bound Ollama; searches win. A search stops the
    # sentence being extracted (closing the connection makes Ollama stop generating) and
    # that sentence is extracted again from the start once no search is running.
    def search_started(self):
        with self._search_cv:
            self._searches += 1
            if self._active_call is not None:
                self._call_stopped = True
                _close_now(self._active_call)

    def search_finished(self):
        with self._search_cv:
            self._searches = max(0, self._searches - 1)
            self._search_cv.notify_all()

    def _wait_for_searches(self):
        with self._search_cv:
            while self._searches:
                self._search_cv.wait(timeout=5)

    def _register_call(self, connection):
        with self._search_cv:
            if self._searches:  # a search started between the wait and the connect
                self._call_stopped = True
                raise _StoppedForSearch()
            self._active_call = connection

    def _call_model(self, sentence, previous):
        while True:
            self._wait_for_searches()
            try:
                return extract_facts(sentence, previous, on_connect=self._register_call)
            except Exception:
                with self._search_cv:
                    stopped = self._call_stopped
                if not stopped:
                    raise
                logger.debug("Knowledge graph: sentence stopped for a search; will redo it")
            finally:
                with self._search_cv:
                    self._active_call = None
                    self._call_stopped = False

    def enqueue_file(self, corpus_id, file_id):
        if not self.enabled:
            return
        with self._queued_lock:
            if (corpus_id, file_id) in self._queued:
                return
            self._queued.add((corpus_id, file_id))
        self._queue.put((corpus_id, file_id))

    def enqueue_backlog(self):
        """Queues every uploaded file that still has unprocessed chunks (e.g. after a restart)."""
        if not self.enabled:
            return
        try:
            with self.driver.session() as session:
                rows = session.run(
                    """
                    MATCH (c:Chunk)
                    WHERE c.corpus_id IS NOT NULL AND c.corpus_id <> 'wiki' AND c.kg_status IS NULL
                    RETURN DISTINCT c.corpus_id AS corpus_id, c.file_id AS file_id
                    """
                ).data()
        except Exception:
            logger.exception("Could not read the knowledge-graph backlog")
            return
        for row in rows:
            self.enqueue_file(row["corpus_id"], row["file_id"])
        if rows:
            logger.info("Knowledge graph: %d file(s) queued for extraction", len(rows))

    def progress(self, corpus_id, file_ids):
        with self.driver.session() as session:
            rows = session.run(
                """
                MATCH (c:Chunk {corpus_id: $corpus_id}) WHERE c.file_id IN $file_ids
                RETURN coalesce(c.kg_status, 'pending') AS status, count(*) AS n
                """,
                corpus_id=corpus_id, file_ids=file_ids,
            ).data()
        counts = {row["status"]: row["n"] for row in rows}
        done, failed, pending = counts.get("done", 0), counts.get("failed", 0), counts.get("pending", 0)
        return {
            "enabled": self.enabled,
            "done": done,
            "failed": failed,
            "pending": pending,
            "total": done + failed + pending,
        }

    def _ensure_indexes(self):
        try:
            with self.driver.session() as session:
                session.run(
                    "CREATE INDEX entity_corpus_key IF NOT EXISTS FOR (e:Entity) ON (e.corpus_id, e.key)"
                ).consume()
        except Exception:
            logger.exception("Could not create the Entity index")

    def _merge_existing(self):
        """Folds duplicates stored before entity resolution existed (or since a file delete
        left a stale cache); a no-op once a workspace is clean."""
        try:
            with self.driver.session() as session:
                corpora = [r["c"] for r in session.run(
                    "MATCH (e:Entity) RETURN DISTINCT e.corpus_id AS c").data()]
            for corpus_id in corpora:
                self.resolver.merge_existing(corpus_id)
        except Exception:
            logger.exception("Merging duplicate entities failed")

    def _run(self):
        if self.resolver is not None:
            self._merge_existing()
        while True:
            corpus_id, file_id = self._queue.get()
            with self._queued_lock:
                self._queued.discard((corpus_id, file_id))
            try:
                self._process_file(corpus_id, file_id)
            except ConnectionError as exc:
                logger.warning("Knowledge graph paused: %s. Retrying in %ds", exc, RETRY_DELAY_SECONDS)
                time.sleep(RETRY_DELAY_SECONDS)
                self.enqueue_file(corpus_id, file_id)
            except Exception:
                logger.exception("Knowledge-graph extraction failed for file %s", file_id)

    def _process_file(self, corpus_id, file_id):
        with self.driver.session() as session:
            # Every chunk, not only pending ones: each needs the previous chunk's text
            # to skip the overlapping sentences it repeats
            chunks = session.run(
                """
                MATCH (c:Chunk {corpus_id: $corpus_id, file_id: $file_id})
                RETURN c.chunk_id AS chunk_id, c.content AS content, c.kg_status AS status
                ORDER BY c.chunk_index
                """,
                corpus_id=corpus_id, file_id=file_id,
            ).data()
        pending = [i for i, c in enumerate(chunks) if c["status"] is None]
        if not pending:
            return
        logger.info("Knowledge graph: extracting %d chunk(s) of file %s", len(pending), file_id)
        started = time.time()
        total_facts = 0
        for i in pending:
            chunk = chunks[i]
            previous = (chunks[i - 1]["content"] or "") if i > 0 else ""
            try:
                entities, facts = extract_chunk(
                    chunk["content"] or "", previous, call_model=self._call_model
                )
            except ConnectionError:
                raise  # leave the rest pending; the worker retries later
            except Exception as exc:
                logger.warning("Extraction failed for chunk %s: %s", chunk["chunk_id"], exc)
                self._mark(chunk["chunk_id"], "failed")
                continue
            entities, facts = self._resolve(corpus_id, entities, facts)
            self._write(chunk["chunk_id"], entities, facts)
            total_facts += len(facts)
        logger.info(
            "Knowledge graph: file %s done, %d fact(s) in %.0fs", file_id, total_facts, time.time() - started
        )

    def _mark(self, chunk_id, status):
        with self.driver.session() as session:
            session.run(
                "MATCH (c:Chunk {chunk_id: $chunk_id}) SET c.kg_status = $status",
                chunk_id=chunk_id, status=status,
            ).consume()

    def _resolve(self, corpus_id, entities, facts):
        """Rewrites entity keys to their canonical entity (entity_resolution.py). An entity
        written onto another carries its own name as an alias; facts that become self-loops
        (both ends one concept) are dropped."""
        if self.resolver is None or not entities:
            return entities, facts
        try:
            mapping = self.resolver.resolve(corpus_id, entities)
        except Exception:
            logger.exception("Entity resolution failed; writing entities unmerged")
            return entities, facts
        merged = {}
        for e in entities:
            key = mapping.get(e["key"], e["key"])
            entry = merged.setdefault(key, {"key": key, "name": e["name"], "kind": e["kind"], "aliases": []})
            if key != e["key"]:
                entry["aliases"].append(e["name"])
        out_facts = []
        for f in facts:
            a, b = mapping.get(f["source"], f["source"]), mapping.get(f["target"], f["target"])
            if a != b:
                out_facts.append({**f, "source": a, "target": b})
        return list(merged.values()), out_facts

    def _write(self, chunk_id, entities, facts):
        # Starts from MATCH on the chunk, so nothing is written if the file was deleted meanwhile
        for ent in entities:
            ent.setdefault("aliases", [])
        with self.driver.session() as session:
            session.execute_write(lambda tx: tx.run(
                """
                MATCH (c:Chunk {chunk_id: $chunk_id})
                SET c.kg_status = 'done', c.kg_facts = size($facts)
                WITH c
                UNWIND $entities AS ent
                MERGE (e:Entity {corpus_id: c.corpus_id, key: ent.key})
                  ON CREATE SET e.name = ent.name, e.kind = ent.kind
                WITH c, e, [a IN ent.aliases WHERE a <> e.name AND NOT a IN coalesce(e.aliases, [])] AS extra
                SET e.aliases = CASE WHEN size(extra) = 0 THEN e.aliases ELSE coalesce(e.aliases, []) + extra END
                MERGE (c)-[:MENTIONS]->(e)
                WITH DISTINCT c
                UNWIND $facts AS f
                MATCH (a:Entity {corpus_id: c.corpus_id, key: f.source})
                MATCH (b:Entity {corpus_id: c.corpus_id, key: f.target})
                CREATE (a)-[:FACT {corpus_id: c.corpus_id, file_id: c.file_id, chunk_id: c.chunk_id,
                                   relation: f.relation, evidence: f.evidence}]->(b)
                """,
                chunk_id=chunk_id, entities=entities, facts=facts,
            ).consume())


def delete_file_entities(session, corpus_id, file_id):
    """Removes a file's facts and any entities no remaining chunk mentions.
    Call after the file's chunks are deleted (DETACH DELETE removes their MENTIONS)."""
    session.run(
        "MATCH ()-[f:FACT {corpus_id: $corpus_id, file_id: $file_id}]->() DELETE f",
        corpus_id=corpus_id, file_id=file_id,
    ).consume()
    session.run(
        """
        MATCH (e:Entity {corpus_id: $corpus_id})
        WHERE NOT (e)<-[:MENTIONS]-()
        DETACH DELETE e
        """,
        corpus_id=corpus_id,
    ).consume()


def load_graph(driver, corpus_id, file_ids):
    """Stored entities, mentions and facts for the given files."""
    with driver.session() as session:
        mentions = session.run(
            """
            MATCH (c:Chunk {corpus_id: $corpus_id})-[:MENTIONS]->(e:Entity)
            WHERE c.file_id IN $file_ids
            RETURN c.chunk_id AS chunk_id, e.key AS key, e.name AS name, e.kind AS kind
            """,
            corpus_id=corpus_id, file_ids=file_ids,
        ).data()
        facts = session.run(
            """
            MATCH (a:Entity)-[f:FACT {corpus_id: $corpus_id}]->(b:Entity)
            WHERE f.file_id IN $file_ids
            RETURN a.key AS source, b.key AS target, f.relation AS relation,
                   f.evidence AS evidence, f.chunk_id AS chunk_id, f.file_id AS file_id
            """,
            corpus_id=corpus_id, file_ids=file_ids,
        ).data()
    return {"mentions": mentions, "facts": facts}
