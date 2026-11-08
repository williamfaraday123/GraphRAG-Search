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


def collection_for_corpus(corpus_id: str) -> str:
    normalized_id = normalize_corpus_id(corpus_id)
    if normalized_id == WIKI_CORPUS_ID:
        return "ai_docs"
    return f"corpus_{re.sub('-', '', normalized_id)}"
