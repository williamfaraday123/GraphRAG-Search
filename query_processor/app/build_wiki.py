"""
Builds the wiki corpus the same way as an upload workspace, offline.

  1. Splits the WikiText files into articles (one per " = Title = " heading) and undoes
     WikiText's tokenisation ("guest @-@ starring" -> "guest-starring", "actor ." -> "actor.").
  2. Indexes each article like an uploaded file (corpus_indexer.index_text): chunks and
     NEXT_CHUNK edges in Neo4j, vectors in Milvus (ai_docs), the text in MinIO.
  3. Extracts the knowledge graph one sentence at a time with KG_MODEL, merging entities
     that name the same concept (knowledge_graph.py, entity_resolution.py).

Resumable: articles already indexed are skipped and only chunks without a kg_status are
extracted, so a stopped build continues where it left off. Run it in an empty project (see
docker-compose.wiki-build.yml) so the snapshot exported afterwards holds only the wiki.

Configuration (environment):
  WIKI_DIR            folder with the WikiText files       (default: /raw_data)
  WIKI_FILES          comma-separated files to read        (default: wiki.test.txt)
  WIKI_MAX_ARTICLES   stop after this many articles; 0=all (default: 0)
  KG_MODEL, CHUNK_SIZE, CHUNK_OVERLAP and the database settings as for the query processor.
"""
import logging
import os
import re
import sys
import time
import uuid
from datetime import datetime, timezone

from pymilvus import utility

from corpus import WIKI_CORPUS_ID, collection_for_corpus
from corpus_indexer import CorpusIndexer
from embedding import EmbeddingService
from entity_resolution import EntityResolver
from knowledge_graph import KnowledgeGraphBuilder, extract_facts

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("build_wiki")

EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # must match the query processor (main.py)
ARTICLE_NAMESPACE = uuid.UUID("6f1c3c2e-7c1a-4f0e-9a51-2b3d2f8e9a10")
OLLAMA_RETRY_SECONDS = 30

_TITLE = re.compile(r"^ = (?!=)(.+?) = $")
_HEADING = re.compile(r"^ (?:= ){2,}(.+?)(?: =){2,} $")


# --- Reading WikiText ----------------------------------------------------------

def detokenize(text):
    """'a guest @-@ starring role , in 2000 .' -> 'a guest-starring role, in 2000.'"""
    text = text.replace(" @-@ ", "-").replace(" @,@ ", ",").replace(" @.@ ", ".")
    text = re.sub(r" ([,.;:!?%)\]}])", r"\1", text)
    text = re.sub(r"([(\[{$]) ", r"\1", text)
    text = re.sub(r" (n't|'s|'re|'ve|'ll|'d|'m)\b", r"\1", text)
    return re.sub(r"[ \t]{2,}", " ", text).strip()


def read_articles(path):
    """[(title, text)] in file order. Section headings become their own paragraph."""
    articles, title, paragraphs = [], None, []

    def finish():
        if title is not None:
            body = "\n\n".join(p for p in paragraphs if p)
            if body.strip():
                articles.append((title, body))

    with open(path, encoding="utf-8") as handle:
        for raw in handle:
            line = raw.rstrip("\n")
            match = _TITLE.match(line)
            if match:
                finish()
                title, paragraphs = detokenize(match.group(1)), []
                continue
            heading = _HEADING.match(line)
            if heading:
                paragraphs.append(detokenize(heading.group(1)))
            elif line.strip():
                paragraphs.append(detokenize(line))
    finish()
    return articles


def safe_filename(title):
    name = re.sub(r'[\\/:*?"<>|\x00-\x1f]', " ", title)
    name = " ".join(name.split())[:150] or "Untitled"
    return f"{name}.txt"


def select_articles():
    """[(file_id, filename, text)] for the configured files, in order."""
    folder = os.getenv("WIKI_DIR", "/raw_data")
    files = [f.strip() for f in os.getenv("WIKI_FILES", "wiki.test.txt").split(",") if f.strip()]
    limit = int(os.getenv("WIKI_MAX_ARTICLES", "0"))
    selected, seen = [], {}
    for name in files:
        path = os.path.join(folder, name)
        if not os.path.exists(path):
            sys.exit(f"WikiText file not found: {path} (set WIKI_DIR / WIKI_FILES)")
        for title, text in read_articles(path):
            # Stable ids, so a rerun finds what an earlier run indexed
            n = seen[(name, title)] = seen.get((name, title), 0) + 1
            file_id = str(uuid.uuid5(ARTICLE_NAMESPACE, f"{name}\n{title}\n{n}"))
            selected.append((file_id, safe_filename(title), text))
            if limit and len(selected) >= limit:
                return selected
    return selected


# --- Building -------------------------------------------------------------------

def check_project_is_clean(driver):
    """Refuses to run on a database that holds the old wiki or upload workspaces: the
    snapshot is exported from this database, so it must contain the wiki only."""
    with driver.session() as session:
        old = session.run(
            "MATCH (c:Chunk) WHERE c.corpus_id IS NULL OR NOT c.chunk_id CONTAINS ':' RETURN count(c) AS n"
        ).single()["n"]
        uploads = session.run(
            "MATCH (c:Chunk) WHERE c.corpus_id IS NOT NULL AND c.corpus_id <> $wiki RETURN count(c) AS n",
            wiki=WIKI_CORPUS_ID,
        ).single()["n"]
    if old or uploads:
        sys.exit(
            f"This database already holds {old} chunk(s) of the old wiki and {uploads} uploaded "
            "chunk(s). Build the wiki in an empty project (docker-compose.wiki-build.yml), so the "
            "snapshot contains the wiki only."
        )


def indexed_file_ids(driver):
    with driver.session() as session:
        rows = session.run(
            "MATCH (c:Chunk {corpus_id: $wiki}) RETURN DISTINCT c.file_id AS f", wiki=WIKI_CORPUS_ID
        ).data()
    return {r["f"] for r in rows}


def ensure_collection(indexer, dim):
    """ai_docs with an exact (FLAT) index: the wiki is small enough to search exhaustively."""
    name = collection_for_corpus(WIKI_CORPUS_ID)
    indexer._connect_milvus()
    if not utility.has_collection(name):
        indexer.create_collection(name, dim, {"metric_type": "COSINE", "index_type": "FLAT", "params": {}})


def check_model():
    """Fails early, with a clear message, if the extraction model can't be reached."""
    model = os.getenv("KG_MODEL", "qwen2.5:1.5b")
    try:
        extract_facts("Robert Boulter is an English film actor.")
    except Exception as exc:
        sys.exit(f"Knowledge-graph model {model} is not usable: {exc}")
    logger.info("Extraction model %s is reachable", model)


def record_build(driver, articles, started):
    """Build settings, read by the snapshot export (ingestion_pipeline/wiki_snapshot.py)."""
    with driver.session() as session:
        counts = session.run(
            """
            MATCH (c:Chunk {corpus_id: $wiki})
            WITH count(c) AS chunks, sum(CASE WHEN c.kg_status IS NULL THEN 1 ELSE 0 END) AS pending
            OPTIONAL MATCH (e:Entity {corpus_id: $wiki})
            WITH chunks, pending, count(e) AS entities
            OPTIONAL MATCH ()-[f:FACT {corpus_id: $wiki}]->()
            RETURN chunks, pending, entities, count(f) AS facts
            """,
            wiki=WIKI_CORPUS_ID,
        ).single().data()
        session.run(
            """
            MERGE (m:WikiBuild {name: 'wiki'})
            SET m += $props
            """,
            props={
                **counts,
                "articles": len(articles),
                "files": os.getenv("WIKI_FILES", "wiki.test.txt"),
                "embedding_model": EMBEDDING_MODEL,
                "kg_model": os.getenv("KG_MODEL", "qwen2.5:1.5b"),
                "kg_enabled": os.getenv("KG_ENABLED", "true"),
                "chunk_size": int(os.getenv("CHUNK_SIZE", "1500")),
                "chunk_overlap": int(os.getenv("CHUNK_OVERLAP", "200")),
                "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "build_seconds": round(time.time() - started),
            },
        ).consume()
    return counts


def main():
    started = time.time()
    articles = select_articles()
    if not articles:
        sys.exit("No articles found")
    logger.info("Wiki build: %d article(s), %d characters",
                len(articles), sum(len(text) for _, _, text in articles))

    embedding = EmbeddingService(model_name=EMBEDDING_MODEL)
    indexer = CorpusIndexer(embedding)
    driver = indexer.neo4j_driver
    check_project_is_clean(driver)
    ensure_collection(indexer, len(embedding.generate_embedding("dimension probe")))

    # 1. Chunk, embed and store every article (fast), so the extraction below can be stopped
    #    and resumed at any point
    done = indexed_file_ids(driver)
    for i, (file_id, filename, text) in enumerate(articles, 1):
        if file_id in done:
            continue
        result = indexer.index_text(WIKI_CORPUS_ID, file_id, filename, text)
        logger.info("Indexed %d/%d %s (%d chunks)", i, len(articles), filename, result["chunk_count"])

    # 2. Knowledge graph, sentence by sentence (slow: roughly 4 s per sentence on a CPU)
    builder = KnowledgeGraphBuilder(
        driver, resolver=EntityResolver(driver, embedding.generate_batch_embeddings), background=False
    )
    if builder.enabled:
        check_model()
        extract_started, total_facts = time.time(), 0
        for i, (file_id, filename, _) in enumerate(articles, 1):
            while True:
                try:
                    total_facts += builder.process_file(WIKI_CORPUS_ID, file_id)
                    break
                except ConnectionError as exc:
                    logger.warning("%s. Retrying in %ds", exc, OLLAMA_RETRY_SECONDS)
                    time.sleep(OLLAMA_RETRY_SECONDS)
            elapsed = time.time() - extract_started
            remaining = elapsed / i * (len(articles) - i)
            logger.info("Knowledge graph %d/%d done (%s); %d fact(s) so far; about %.1f h left",
                        i, len(articles), filename, total_facts, remaining / 3600)
    else:
        logger.info("KG_ENABLED is false: skipping knowledge-graph extraction")

    counts = record_build(driver, articles, started)
    logger.info("Wiki build complete: %s", counts)
    indexer.close()


if __name__ == "__main__":
    main()
