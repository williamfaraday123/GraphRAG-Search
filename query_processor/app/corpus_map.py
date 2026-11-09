"""
Corpus map: entities, relationships and themed chunk clusters for an upload workspace.

Everything here is computed from the indexed chunks without an LLM, so a map of a few
hundred chunks builds in seconds on CPU:

- Entities: proper names (capitalised phrases and acronyms) plus recurring key
  concepts (repeated noun-like word pairs and words), counted per chunk.
- Relationships: two entities mentioned in the same sentence. The weight is the number
  of sentences that link them, and up to two of those sentences are kept as evidence.
- Clusters: chunks grouped by embedding similarity (Louvain communities on a
  nearest-neighbour graph), labelled by the entities most specific to each group.
  A cluster that spans several files shows where documents discuss the same theme.
"""
import math
import re
from collections import Counter, defaultdict

import networkx as nx
import numpy as np

MAX_ENTITIES = 40
MAX_RELATIONSHIPS = 80
MAX_CONCEPTS = 15
MAX_EXAMPLES = 2
MAX_FACT_EXAMPLES = 4
SNIPPET_CHARS = 220

STOPWORDS = set("""
a about above after again against all also although am an and any are as at be because been
before being below between both but by can could did do does doing down during each either
else even ever every few for from further had has have having he her here hers herself him
himself his how however i if in into is it its itself just least less like made make many
may me might more most much must my myself neither no nor not now of off often on once one
only or other others our ours ourselves out over own per perhaps quite rather same she should
since so some such than that the their theirs them themselves then there these they this
those though through thus to too under until up upon us very was we were what when where
whether which while who whom whose why will with within without would yet you your yours
yourself yourselves first second new well still way ways part parts kind lot lots thing
things something anything everything nothing time times year years day days also already
almost always never sometimes usually another around across along among toward towards onto
get got go goes going gone come came use used using said say says see seen take taken give
given become became becoming find found know known let put show shown tell told think try
took began begun brought held left felt went thought gave led lost won sold bought paid meant
want wanted need needed seem seemed seems keep kept able unable various several certain
""".split())

# Capitalised words that start sentences or headings without naming anything.
# Ordinals stay allowed so "First World War" keeps its first word.
NON_NAMES = (STOPWORDS - {"first", "second"}) | {
    "mr", "mrs", "ms", "dr", "chapter", "section", "figure", "table", "page", "note",
    "yes", "ok", "okay", "dear", "regards", "thanks", "thank", "please",
}
NAME_CONNECTORS = {"of", "the", "de", "du", "da", "del", "von", "van", "and", "&", "for"}

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])|\n{2,}|\n(?=[-*•\d])")
# Names and concept pairs never run across punctuation ("In Scandinavia, the Swedish")
_CLAUSE_SPLIT = re.compile(r"[,;:()\[\]\"“”–—?!]|\s-\s|\.(?=\s)")
_TOKEN = re.compile(r"[A-Za-z][A-Za-z'’\-]*[A-Za-z]|[A-Za-z]|\d+(?:[.,]\d+)*")


def split_sentences(text):
    return [s.strip() for s in _SENTENCE_SPLIT.split(text) if s and s.strip()]


def _clauses(sentence):
    """Token lists per clause; empty clauses dropped."""
    out = []
    for clause in _CLAUSE_SPLIT.split(sentence):
        tokens = [t for t in (_clean_token(t) for t in _TOKEN.findall(clause)) if t]
        if tokens:
            out.append(tokens)
    return out


def _squash(text):
    return " ".join(text.split()).lower()


def _clean_token(token):
    token = re.sub(r"['’]s$", "", token)  # possessive: Hollywood's -> Hollywood
    return token.strip("'’-")


def _is_capitalised(token):
    return token[:1].isupper()


def _is_acronym(token):
    return len(token) >= 2 and token.isupper() and token.isalpha()


def _name_spans(tokens):
    """Yields (start_index, phrase) for runs of capitalised tokens, allowing short
    connectors inside a run ('First World War', 'Bank of England')."""
    i = 0
    while i < len(tokens):
        if not _is_capitalised(tokens[i]) or tokens[i].isdigit():
            i += 1
            continue
        start, j = i, i + 1
        while j < len(tokens):
            if _is_capitalised(tokens[j]) and not tokens[j].isdigit():
                j += 1
            elif (tokens[j].lower() in NAME_CONNECTORS and j + 1 < len(tokens)
                  and _is_capitalised(tokens[j + 1])):
                j += 2
            else:
                break
        words = tokens[start:j]
        while words and words[0].lower() in NON_NAMES:  # "The Netherlands" -> "Netherlands"
            words = words[1:]
            start += 1
        while words and words[-1].lower() in NAME_CONNECTORS:
            words = words[:-1]
        if words:
            yield start, " ".join(words)
        i = j


class _EntityIndex:
    """Collects mentions under a case-insensitive key, keeping the most common spelling."""

    def __init__(self):
        self.spellings = defaultdict(Counter)
        self.kind = {}
        self.chunks = defaultdict(set)      # key -> chunk ids mentioning it
        self.mentions = Counter()           # key -> total mentions

    def add(self, key_text, kind, chunk_id):
        key = key_text.lower()
        self.spellings[key][key_text] += 1
        self.kind.setdefault(key, kind)
        if kind == "name":
            self.kind[key] = "name"  # a name sighting outranks a concept sighting
        self.chunks[key].add(chunk_id)
        self.mentions[key] += 1

    def label(self, key):
        return self.spellings[key].most_common(1)[0][0]


def _extract_names(chunks):
    """Returns {chunk_id: [(sentence, set(entity keys), clauses)]} and the entity index.

    Chunks overlap by a couple of hundred characters, so a sentence that already
    appeared in the previous chunk of the same file is skipped; otherwise every
    phrase in an overlap would look like it was repeated."""
    mid_sentence_caps, lowercase_seen = set(), set()
    parsed = {}
    previous = {}  # file_id -> squashed text of the previous chunk
    for chunk in chunks:
        prior = previous.get(chunk["file_id"], "")
        sentences = []
        for sentence in split_sentences(chunk["content"]):
            if prior and _squash(sentence).rstrip(".!?\"' ") in prior:
                continue
            clauses = _clauses(sentence)
            sentences.append((sentence, clauses))
            for c, tokens in enumerate(clauses):
                for k, token in enumerate(tokens):
                    if (c > 0 or k > 0) and _is_capitalised(token):
                        mid_sentence_caps.add(token.lower())
                    elif token.islower():
                        lowercase_seen.add(token)
        parsed[chunk["chunk_id"]] = sentences
        previous[chunk["file_id"]] = _squash(chunk["content"])

    index = _EntityIndex()
    per_chunk = {}
    for chunk in chunks:
        chunk_id = chunk["chunk_id"]
        out = []
        for sentence, clauses in parsed[chunk_id]:
            keys = set()
            for c, tokens in enumerate(clauses):
                for start, phrase in _name_spans(tokens):
                    lower = phrase.lower()
                    if " " not in phrase:
                        if lower in NON_NAMES or len(lower) < 2:
                            continue
                        if not _is_acronym(phrase):
                            if c == 0 and start == 0 and lower not in mid_sentence_caps:
                                continue  # just a capitalised sentence opener
                            if lower in lowercase_seen and lower not in mid_sentence_caps:
                                continue
                    index.add(phrase, "name", chunk_id)
                    keys.add(lower)
            out.append((sentence, keys, clauses))
        per_chunk[chunk_id] = out
    return per_chunk, index


def _is_concept_word(word, name_words):
    """Rough noun filter without a tagger: drops stopwords, names, adverbs (-ly)
    and past-tense verbs (-ed)."""
    return (len(word) >= 4 and word not in STOPWORDS and word not in name_words
            and not word.endswith("ly") and not word.endswith("ed"))


def _add_concepts(per_chunk, index, chunk_count):
    """Adds recurring concepts: word pairs seen in 2+ chunks, then single words in 3+.
    Keeps them specific to this corpus by preferring terms that are not in every chunk."""
    name_words = {w for key in index.kind for w in key.split()}
    pair_chunks, word_chunks = defaultdict(set), defaultdict(set)
    for chunk_id, sentences in per_chunk.items():
        for _, _, clauses in sentences:
            for tokens in clauses:
                words = [t if t.islower() and _is_concept_word(t, name_words) else None for t in tokens]
                for w in words:
                    if w:
                        word_chunks[w].add(chunk_id)
                for a, b in zip(words, words[1:]):
                    if a and b:
                        pair_chunks[f"{a} {b}"].add(chunk_id)

    def score(chunk_ids):
        df = len(chunk_ids)
        return df * math.log(1 + chunk_count / df) if df else 0

    pairs = [(p, ids) for p, ids in pair_chunks.items() if len(ids) >= 2]
    pairs.sort(key=lambda item: score(item[1]), reverse=True)
    chosen = pairs[: MAX_CONCEPTS // 2]
    covered = {w for p, _ in chosen for w in p.split()}
    min_df = 3 if chunk_count >= 3 else 2
    words = [(w, ids) for w, ids in word_chunks.items() if len(ids) >= min_df and w not in covered]
    words.sort(key=lambda item: score(item[1]), reverse=True)
    chosen += words[: MAX_CONCEPTS - len(chosen)]

    concepts = {term for term, _ in chosen}
    pair_terms = {t for t in concepts if " " in t}
    for chunk_id, sentences in per_chunk.items():
        for sentence, keys, clauses in sentences:
            padded = " | ".join(" ".join(t.lower() for t in tokens) for tokens in clauses)
            padded = f" {padded} "
            for term in concepts:
                if f" {term} " in padded:
                    # a single word inside a chosen pair is counted by the pair
                    if " " not in term and any(term in p.split() and f" {p} " in padded for p in pair_terms):
                        continue
                    index.add(term, "concept", chunk_id)
                    keys.add(term)


def _snippet(text, limit=SNIPPET_CHARS):
    text = " ".join(text.split()).lstrip(".,;:!?)]}\"' ")  # chunk overlap can start mid-punctuation
    return text if len(text) <= limit else text[: limit - 1].rsplit(" ", 1)[0] + "…"


def _component_layout(graph, nodes):
    """Positions one connected component inside a unit box."""
    sub = graph.subgraph(nodes)
    if len(nodes) == 1:
        return {nodes[0]: (0.5, 0.5)}
    if len(nodes) == 2:
        return {nodes[0]: (0.2, 0.5), nodes[1]: (0.8, 0.5)}
    # Kamada-Kawai spreads small graphs evenly (fewer overlapping labels than spring)
    pos = nx.kamada_kawai_layout(sub, weight=None)
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    span_x = (max(xs) - min(xs)) or 1
    span_y = (max(ys) - min(ys)) or 1
    return {k: ((p[0] - min(xs)) / span_x, (p[1] - min(ys)) / span_y) for k, p in pos.items()}


def _layout(entity_keys, edges):
    """Deterministic 0-1 coordinates for the entity graph.

    The largest connected group gets the main area on the left. Smaller groups and
    unlinked entities fill a grid on the right, sized by how many entities they hold,
    so neither side gets squeezed (a fact graph often has many small groups)."""
    graph = nx.Graph()
    graph.add_nodes_from(entity_keys)
    graph.add_edges_from(edges)
    if not graph.number_of_nodes():
        return {}
    components = sorted((sorted(c) for c in nx.connected_components(graph)), key=lambda c: (-len(c), c[0]))
    main, rest = components[0], components[1:]
    if not rest:
        return _component_layout(graph, main)

    total = graph.number_of_nodes()
    side = min(0.6, max(0.25, 1 - len(main) / total))  # width share of the grid
    gap = 0.05
    positions = {}
    for key, (x, y) in _component_layout(graph, main).items():
        positions[key] = (x * (1 - side - gap), y)

    # Grid cells: groups first (largest first), then single entities two to a cell
    groups = [c for c in rest if len(c) > 1]
    singles = [c[0] for c in rest if len(c) == 1]
    cells = groups + [singles[i:i + 2] for i in range(0, len(singles), 2)]
    columns = 2 if side >= 0.35 and len(cells) > 3 else 1
    rows = math.ceil(len(cells) / columns)
    cell_w, cell_h = side / columns, 1 / rows
    left = 1 - side
    for i, cell in enumerate(cells):
        col, row = i % columns, i // columns
        x0, y0 = left + col * cell_w, row * cell_h
        if len(cell) > 1 and graph.subgraph(cell).number_of_edges():
            layout = _component_layout(graph, cell)
        else:  # unlinked singles side by side
            layout = {key: (0.2 + 0.6 * j / max(1, len(cell) - 1), 0.5) for j, key in enumerate(cell)}
        for key, (x, y) in layout.items():
            # Inset within the cell so neighbouring cells' labels don't touch
            positions[key] = (x0 + cell_w * (0.12 + 0.76 * x), y0 + cell_h * (0.3 + 0.55 * y))
    return positions


def _cluster(chunks, embeddings):
    """Louvain communities on a graph linking each chunk to its most similar chunks."""
    n = len(chunks)
    if n == 0:
        return []
    if n == 1:
        return [[0]]
    vectors = np.asarray(embeddings, dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-9
    sims = vectors @ vectors.T
    np.fill_diagonal(sims, -1)

    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    neighbours = min(3, n - 1)
    for i in range(n):
        for rank, j in enumerate(np.argsort(-sims[i])[:neighbours]):
            # The nearest neighbour joins at a looser threshold so short chunks on
            # one topic don't end up as single-chunk clusters
            if sims[i, j] >= (0.2 if rank == 0 else 0.35):
                graph.add_edge(i, int(j), weight=float(sims[i, j]))
    # Sequential chunks of one file usually continue a topic; a weak link keeps them together
    for i in range(n - 1):
        if chunks[i]["file_id"] == chunks[i + 1]["file_id"] and sims[i, i + 1] >= 0.25:
            w = graph.get_edge_data(i, i + 1, {"weight": 0})["weight"]
            graph.add_edge(i, i + 1, weight=max(w, float(sims[i, i + 1])))

    communities = nx.community.louvain_communities(graph, weight="weight", resolution=1.0, seed=42)
    return sorted((sorted(c) for c in communities), key=lambda c: (-len(c), c[0]))


class _Graph:
    """What both graph sources produce: entities, their chunk mentions, and edges."""

    def __init__(self):
        self.labels, self.kinds = {}, {}
        self.chunks = defaultdict(set)          # entity key -> chunk ids
        self.mentions = Counter()               # entity key -> mention count
        self.per_chunk = defaultdict(set)       # chunk id -> entity keys (labels clusters)
        self.edges = {}                         # (a, b) sorted -> {weight, files, examples}
        self.rank = {}                          # entity key -> sort position (lower = more important)


def _keyword_graph(chunks, filename):
    """Statistical fallback: capitalised names + recurring concepts, linked when they
    share a sentence."""
    per_chunk, index = _extract_names(chunks)
    _add_concepts(per_chunk, index, len(chunks))
    g = _Graph()
    ranked = sorted(index.kind, key=lambda k: (-len(index.chunks[k]), -index.mentions[k],
                                               index.kind[k] != "name", k))
    for position, key in enumerate(ranked[:MAX_ENTITIES]):
        g.labels[key], g.kinds[key] = index.label(key), index.kind[key]
        g.chunks[key] = set(index.chunks[key])
        g.mentions[key] = index.mentions[key]
        g.rank[key] = position
    for chunk in chunks:
        for sentence, keys, _ in per_chunk[chunk["chunk_id"]]:
            present = sorted(k for k in keys if k in g.labels)
            g.per_chunk[chunk["chunk_id"]].update(present)
            for i, a in enumerate(present):
                for b in present[i + 1:]:
                    edge = g.edges.setdefault((a, b), {"weight": 0, "examples": [], "files": set()})
                    edge["weight"] += 1
                    edge["files"].add(chunk["file_id"])
                    if len(edge["examples"]) < MAX_EXAMPLES:
                        edge["examples"].append({
                            "chunkId": chunk["chunk_id"],
                            "filename": filename.get(chunk["file_id"], "Unknown file"),
                            "sentence": _snippet(sentence, 300),
                        })
    return g


def _knowledge_graph(chunks, filename, kg):
    """Facts the LLM extracted (see knowledge_graph.py), drawn as directed, labelled edges."""
    chunk_ids = {c["chunk_id"] for c in chunks}
    g = _Graph()
    for row in kg["mentions"]:
        if row["chunk_id"] not in chunk_ids:
            continue
        g.labels.setdefault(row["key"], row["name"])
        g.kinds.setdefault(row["key"], row["kind"] or "concept")
        g.chunks[row["key"]].add(row["chunk_id"])

    # Names extracted from one file often appear in others; find those mentions by text
    # so entities connect files even where the model didn't pick them out.
    for key, label in g.labels.items():
        if g.kinds[key] != "name":
            continue
        pattern = re.compile(rf"(?<!\w){re.escape(label)}(?!\w)", re.IGNORECASE)
        for chunk in chunks:
            if chunk["chunk_id"] not in g.chunks[key] and pattern.search(chunk["content"]):
                g.chunks[key].add(chunk["chunk_id"])

    degree = Counter()
    for fact in kg["facts"]:
        if fact["chunk_id"] in chunk_ids and fact["source"] in g.labels and fact["target"] in g.labels:
            degree[fact["source"]] += 1
            degree[fact["target"]] += 1
    # Entities in the most facts first; then the most widely mentioned
    ranked = sorted(g.labels, key=lambda k: (-degree[k], -len(g.chunks[k]), k))
    kept = set(ranked[:MAX_ENTITIES])
    for position, key in enumerate(ranked):
        g.rank[key] = position
    for key in list(g.labels):
        if key not in kept:
            del g.labels[key], g.kinds[key], g.chunks[key]
    for key, ids in g.chunks.items():
        g.mentions[key] = len(ids)
        for cid in ids:
            g.per_chunk[cid].add(key)

    for fact in sorted(kg["facts"], key=lambda f: (f["chunk_id"], f["source"], f["target"])):
        a, b = fact["source"], fact["target"]
        if fact["chunk_id"] not in chunk_ids or a not in kept or b not in kept:
            continue
        edge = g.edges.setdefault(tuple(sorted((a, b))), {"weight": 0, "examples": [], "files": set()})
        edge["weight"] += 1
        edge["files"].add(fact["file_id"])
        if len(edge["examples"]) < MAX_FACT_EXAMPLES:
            edge["examples"].append({
                "chunkId": fact["chunk_id"],
                "filename": filename.get(fact["file_id"], "Unknown file"),
                "sentence": _snippet(fact["evidence"] or "", 300),
                "from": a,
                "relation": fact["relation"],
                "to": b,
            })
    return g


def build_corpus_map(chunks, embed, files, kg=None, progress=None):
    """
    chunks:   [{chunk_id, file_id, chunk_index, content}] in document order.
    embed:    callable(list[str]) -> list[vector].
    files:    [{file_id, filename}] in workspace order (fixes each file's colour slot).
    kg:       stored LLM facts ({"mentions", "facts"} from knowledge_graph.load_graph), or None.
    progress: knowledge-graph extraction progress, passed through to the UI.

    Uses the LLM knowledge graph once it has any facts for these chunks; until then
    (or when extraction is off) falls back to the statistical keyword graph.
    """
    chunk_by_id = {c["chunk_id"]: c for c in chunks}
    filename = {f["file_id"]: f["filename"] for f in files}

    g = _knowledge_graph(chunks, filename, kg) if kg and kg.get("facts") else None
    mode = "knowledge_graph" if g and g.edges else "keywords"
    if mode == "keywords":
        g = _keyword_graph(chunks, filename)
    kept = set(g.labels)

    top_edges = dict(sorted(g.edges.items(), key=lambda kv: (-kv[1]["weight"], kv[0]))[:MAX_RELATIONSHIPS])
    positions = _layout(sorted(kept), top_edges)

    entities = []
    for key in sorted(kept, key=lambda k: (-len(g.chunks[k]), g.rank.get(k, 0), k)):
        ids = sorted(g.chunks[key], key=lambda c: (chunk_by_id[c]["file_id"], chunk_by_id[c]["chunk_index"]))
        x, y = positions.get(key, (0.5, 0.5))
        entities.append({
            "id": key,
            "label": g.labels[key],
            "type": g.kinds[key],
            "mentions": g.mentions[key],
            "chunkCount": len(ids),
            "fileIds": sorted({chunk_by_id[c]["file_id"] for c in ids}),
            "chunkIds": ids,
            "x": round(x, 4),
            "y": round(y, 4),
        })

    relationships = [
        {
            "source": a,
            "target": b,
            "weight": data["weight"],
            "fileIds": sorted(data["files"]),
            "examples": data["examples"],
        }
        for (a, b), data in top_edges.items()
    ]

    clusters = []
    groups = _cluster(chunks, embed([c["content"] for c in chunks])) if chunks else []
    corpus_df = {k: max(1, len(g.chunks[k])) for k in kept}
    for number, members in enumerate(groups, start=1):
        member_ids = [chunks[i]["chunk_id"] for i in members]
        counts = Counter(k for cid in member_ids for k in g.per_chunk.get(cid, ()) if k in kept)
        # Specific to this cluster: frequent here, relatively rare elsewhere
        top = sorted(counts, key=lambda k: (-(counts[k] * counts[k] / corpus_df[k]), g.rank.get(k, 0), k))[:4]
        file_counts = Counter(chunk_by_id[cid]["file_id"] for cid in member_ids)
        clusters.append({
            "id": number,
            "label": ", ".join(g.labels[k] for k in top[:3]) or f"Theme {number}",
            "entityIds": top,
            "files": [
                {"fileId": f["file_id"], "filename": f["filename"], "chunkCount": file_counts[f["file_id"]]}
                for f in files if file_counts.get(f["file_id"])
            ],
            "chunks": [
                {
                    "chunkId": chunks[i]["chunk_id"],
                    "fileId": chunks[i]["file_id"],
                    "filename": filename.get(chunks[i]["file_id"], "Unknown file"),
                    "chunkIndex": chunks[i]["chunk_index"],
                    "snippet": _snippet(chunks[i]["content"]),
                }
                for i in members
            ],
        })

    chunk_payload = {
        c["chunk_id"]: {
            "fileId": c["file_id"],
            "filename": filename.get(c["file_id"], "Unknown file"),
            "chunkIndex": c["chunk_index"],
            "snippet": _snippet(c["content"]),
        }
        for c in chunks
    }

    return {
        "mode": mode,
        "progress": progress,
        "stats": {
            "files": len({c["file_id"] for c in chunks}),
            "chunks": len(chunks),
            "entities": len(entities),
            "relationships": len(relationships),
            "clusters": len(clusters),
        },
        "files": [{"fileId": f["file_id"], "filename": f["filename"]} for f in files],
        "entities": entities,
        "relationships": relationships,
        "clusters": clusters,
        "chunks": chunk_payload,
    }
