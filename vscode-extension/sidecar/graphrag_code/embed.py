"""
Symbol embeddings for semantic search.

"auto" (the default) uses a code-aware sentence-transformers model, trained on code search
(natural-language query -> function body), when sentence-transformers is installed and the
model can be loaded; otherwise a hashing embedder over identifier parts ("parseConfigFile" ->
parse, config, file), which needs no model and no download. Vectors are L2-normalised, so a
dot product is the cosine similarity.
"""
import hashlib
import logging
import math
import re
from typing import List, Sequence

logger = logging.getLogger(__name__)

DEFAULT_CODE_MODEL = "flax-sentence-embeddings/st-codesearch-distilroberta-base"
HASH_DIM = 1024

_WORDS = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")
_STOP = {"self", "the", "def", "return", "none", "true", "false", "and", "not", "for", "in",
         "is", "if", "else", "import", "from", "class", "of", "to", "a", "an", "or", "with"}


def identifier_words(text: str) -> List[str]:
    words = []
    for token in re.findall(r"[A-Za-z0-9_]+", text):
        for part in token.split("_"):
            words += [w.lower() for w in _WORDS.findall(part)]
    return [w for w in words if len(w) > 1 and w not in _STOP]


class HashingEmbedder:
    name = f"hashing-{HASH_DIM}"
    dim = HASH_DIM

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        out = []
        for text in texts:
            vec = [0.0] * HASH_DIM
            counts = {}
            for w in identifier_words(text):
                counts[w] = counts.get(w, 0) + 1
            for w, c in counts.items():
                h = int.from_bytes(hashlib.blake2b(w.encode(), digest_size=8).digest(), "little")
                vec[h % HASH_DIM] += (1.0 if (h >> 63) & 1 else -1.0) * (1.0 + math.log(c))
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer  # optional dependency
        self.model = SentenceTransformer(model_name)
        self.name = model_name
        self.dim = self.model.get_sentence_embedding_dimension()

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        vectors = self.model.encode(list(texts), batch_size=32, normalize_embeddings=True,
                                    convert_to_numpy=True, show_progress_bar=False)
        return vectors.tolist()


def make_embedder(spec: str = "auto"):
    """'hashing', 'auto' (code model if it loads, else hashing) or a sentence-transformers
    model name (falls back to hashing if it can't be loaded)."""
    if spec == "hashing":
        return HashingEmbedder()
    model = DEFAULT_CODE_MODEL if spec in ("", "auto") else spec
    try:
        return SentenceTransformerEmbedder(model)
    except Exception as e:  # not installed, offline, blocked DLL, ...
        logger.warning("Embedding model %s unavailable (%s); using identifier hashing", model, e)
        return HashingEmbedder()


def symbol_text(sym) -> str:
    """What a symbol is embedded as: where it is, its signature, docstring and code."""
    return f"{sym.kind} {sym.qualname or sym.name} in {sym.path}\n{sym.signature}\n{sym.doc}\n{sym.text[:1500]}"
