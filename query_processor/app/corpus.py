import re
import uuid


WIKI_CORPUS_ID = "wiki"


def normalize_corpus_id(corpus_id: str) -> str:
    if not corpus_id or corpus_id == WIKI_CORPUS_ID:
        return WIKI_CORPUS_ID

    try:
        return str(uuid.UUID(corpus_id))
    except (ValueError, AttributeError) as exc:
        raise ValueError("corpus_id must be 'wiki' or a valid workspace UUID") from exc


def normalize_file_id(file_id: str) -> str:
    """File ids are UUIDs; validating them also makes them safe to embed in Milvus
    filter expressions (e.g. doc_id like "<file_id>:%")."""
    try:
        return str(uuid.UUID(file_id))
    except (ValueError, AttributeError, TypeError) as exc:
        raise ValueError("file_id must be a valid UUID") from exc


def file_filter_expr(file_ids) -> str:
    """Milvus expression matching chunks of the given files. Upload chunk ids are
    '<file_id>:<chunk_index>', so a prefix match selects every chunk of a file."""
    ids = [normalize_file_id(f) for f in file_ids]
    return " or ".join(f'doc_id like "{f}:%"' for f in ids)


def collection_for_corpus(corpus_id: str) -> str:
    normalized_id = normalize_corpus_id(corpus_id)
    if normalized_id == WIKI_CORPUS_ID:
        return "ai_docs"
    return f"corpus_{re.sub('-', '', normalized_id)}"
