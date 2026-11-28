"""
Context for a question or an edit: which symbols the model reads.

1. Seeds: the symbol under the cursor, symbols the question names (full weight only when written
   like code: `push_ppr`, LocalPPR, local_ppr.py; an ordinary word that happens to be a name is
   a weak hint), folders it names ("core", "vscode-extension": their modules), the best vector
   hits, and (lightly) the open files, so the context follows what you're working on.
2. Local push PPR from the seeds over the code graph (graphrag_core.ppr): how strongly each
   symbol is connected to them, at any distance, without walking the whole call graph.
3. Hybrid ranking (graphrag_core.ranking): semantic similarity + PPR + named-in-question.
   Every named folder gets a few of its best symbols in the context.
4. Coverage check (graphrag_core.coverage): every symbol that must be read is in the context;
   for an edit, that's every caller of the symbols being changed.
5. Relationships: the graph's own edges for the model to read beside the code (as the text
   search gives its model knowledge-graph facts): every edge between two named folders, then
   edges across files among the best-ranked symbols.

With the architecture (architecture.py), retrieval goes top-down: components are ranked first
(named, holding the seeds, similar to the question, then PPR over the component graph), the
code at the boundaries between the top ones joins the seeds, and a symbol's component score is
a fourth, light ranking signal: a boost, never a filter, so a wrong first pick costs little.
"""
from collections import Counter
from typing import Dict, List, Optional

from graphrag_core.coverage import required_by, uncovered
from graphrag_core.ppr import push_ppr, top
from graphrag_core.ranking import hybrid_rank

from .index import Index

WEIGHTS = {"semantic": 0.45, "ppr": 0.4, "named": 0.15, "component": 0.1}
TOP_COMPONENTS = 3         # components whose boundaries are seeded (top-down retrieval)
TOP_COMPONENT_SCORE = 0.3  # of the best component's score
BOUNDARY_SEED = 0.3
FOCUS_SEED = 1.0
NAMED_SEED = 1.0
OPEN_FILE_SEED = 0.15
VECTOR_SEEDS = 6
MAX_CONTEXT = 12
CONTEXT_CHARS = 24_000
MAX_REQUIRED = 12          # callers added by the coverage check, at most
GRAPH_VIEW_NODES = 60
IMPACT_RESULTS = 15
FOLDER_SEED = 1.0          # a named folder weighs as much as a named symbol, spread over its modules
FOLDER_MIN_CONTEXT = 2     # symbols from each named folder the context always has
MAX_RELATIONS = 24         # relationship lines the model reads
RELATION_POOL = 40         # best-ranked symbols whose edges count as relevant
# Which edges say most about how code depends on code
_KIND_ORDER = {"calls": 0, "renders": 0, "extends": 1, "implements": 1, "imports": 2, "uses": 2, "uses_type": 3}


def _seeds(index: Index, query: str, focus: Optional[dict], open_files: List[str]):
    g = index.graph
    named = g.seed_candidates(query) if query else {}
    folders = g.named_folders(query) if query else {}
    focus_id = g.symbol_at(index.rel(focus["path"]), int(focus["line"])) if focus and focus.get("path") else None
    if focus and focus.get("path") and not focus_id:
        # A cursor may be in whitespace, an import or another file region that belongs to the
        # module rather than a named symbol. Seed the module so "around cursor" still has a graph.
        module_id = f"{index.rel(focus['path'])}::"
        if module_id in g.symbols:
            focus_id = module_id
    vector = index.vector_search(query, k=30) if query else {}
    seeds: Dict[str, float] = {}
    for sid, sim in list(vector.items())[:VECTOR_SEEDS]:
        seeds[sid] = max(seeds.get(sid, 0.0), sim)
    for sid, weight in named.items():
        seeds[sid] = max(seeds.get(sid, 0.0), NAMED_SEED * weight)
    for folder, modules in folders.items():
        share = FOLDER_SEED / max(len(modules), 1)
        for mid in modules:
            seeds[mid] = max(seeds.get(mid, 0.0), share)
        fid = g.folder_ids.get(folder)  # the folder's own node too (it shows in the graph view)
        if fid:
            seeds[fid] = max(seeds.get(fid, 0.0), FOLDER_SEED / 2)
    if focus_id:
        seeds[focus_id] = max(seeds.get(focus_id, 0.0), FOCUS_SEED)
    for path in open_files or []:
        mid = f"{index.rel(path)}::"
        if mid in g.symbols:
            seeds[mid] = max(seeds.get(mid, 0.0), OPEN_FILE_SEED)
    return seeds, named, focus_id, vector, folders


def is_test(path: str) -> bool:
    name = path.rsplit("/", 1)[-1]
    return ("/tests/" in f"/{path}" or "/test/" in f"/{path}" or "/__tests__/" in f"/{path}"
            or name.startswith("test_") or ".test." in name or ".spec." in name or name.endswith("_test.py"))


def _in_folder(path: str, folder: str) -> bool:
    return path.startswith(folder + "/")


def _label(g, sid: str) -> str:
    s = g.symbols[sid]
    return f"{s.path}::{s.qualname}" if s.qualname else s.path


def relation(g, e) -> dict:
    return {"source": e.src, "target": e.dst, "kind": e.kind, "guessed": e.guessed,
            "text": f"{_label(g, e.src)} --{e.kind}{' (guessed)' if e.guessed else ''}--> {_label(g, e.dst)}"}


def folder_links(index: Index, folders: List[str], ppr: Dict[str, float], per_direction: int) -> tuple:
    """Every edge from one named folder into another (both directions): counts by kind, and
    the most telling edges (calls first, then by PPR)."""
    g = index.graph
    summary, picked = [], []
    for a in folders:
        for b in folders:
            if a == b or _in_folder(a + "/", b) or _in_folder(b + "/", a):
                continue
            edges = [e for e in g.edges if e.kind not in ("contains", "contains_dir")
                     and _in_folder(g.symbols[e.src].path, a) and _in_folder(g.symbols[e.dst].path, b)]
            kinds = Counter(e.kind for e in edges)
            summary.append({"from": a, "to": b, "count": len(edges), "kinds": dict(kinds),
                            "text": f"{a} -> {b}: {len(edges)} edge(s)"
                                    + (f" ({', '.join(f'{n} {k}' for k, n in kinds.most_common())})" if edges else "")})
            edges.sort(key=lambda e: (_KIND_ORDER.get(e.kind, 4), e.guessed, -(ppr.get(e.src, 0.0) + ppr.get(e.dst, 0.0))))
            picked += edges[:per_direction]
    return summary, picked


def relationships(index: Index, ppr: Dict[str, float], context_ids: List[str], folders: List[str],
                  limit: int = MAX_RELATIONS) -> dict:
    """The graph's edges the model reads beside the code (see the module docstring)."""
    g = index.graph
    out, seen = [], set()

    def add(e):
        key = (e.src, e.dst, e.kind)
        if key not in seen and len(out) < limit:
            seen.add(key)
            out.append(relation(g, e))

    summary = []
    if len(folders) >= 2:
        pairs = max(len(folders) * (len(folders) - 1), 1)
        summary, picked = folder_links(index, folders, ppr, per_direction=max(limit // pairs, 3))
        for e in picked:
            add(e)
    pool = set(context_ids) | set(top(ppr, RELATION_POOL))
    nearby = [e for e in g.edges if e.src in pool and e.dst in pool and e.kind not in ("contains", "contains_dir")
              and g.symbols[e.src].path != g.symbols[e.dst].path]
    # Tests last: they use everything, and say little about how the code itself fits together
    nearby.sort(key=lambda e: (is_test(g.symbols[e.src].path), -(ppr.get(e.src, 0.0) + ppr.get(e.dst, 0.0)),
                               _KIND_ORDER.get(e.kind, 4)))
    for e in nearby:
        add(e)
    return {"relations": out, "folderLinks": summary}


def _top_down(index: Index, arch, query: str, seeds: Dict[str, float]) -> tuple:
    """Architecture first: rank the components (architecture.rank_components), then seed the
    code at the boundaries between the top ones, so a question that spans components reaches
    the code that connects them (the fetch call and the route handler) and not only code
    near the names it mentions. Returns (component scores, the top code components, whether
    the model should read the architecture)."""
    from . import architecture as A
    scores = A.rank_components(index, arch, query, seeds)
    chosen = [c for c, s in scores.items() if s >= TOP_COMPONENT_SCORE
              and arch.components[c]["kind"] == "code"][:TOP_COMPONENTS]
    broad = A.is_architecture_question(query)
    if len(chosen) >= 2 or broad:
        for sid in A.boundary_symbols(index, arch, chosen):
            seeds[sid] = max(seeds.get(sid, 0.0), BOUNDARY_SEED)
    return scores, chosen, broad or len(chosen) >= 2


def retrieve(index: Index, query: str, focus: Optional[dict] = None, open_files: List[str] = (),
             mode: str = "ask", targets: List[str] = (), k: int = MAX_CONTEXT,
             budget: int = CONTEXT_CHARS, arch=None) -> dict:
    """mode 'ask': the named/focused symbols must be in the context.
    mode 'edit': so must every caller of them (and of `targets`), before anything changes.
    arch: the architecture (architecture.py), for top-down ranking: components first."""
    g = index.graph
    seeds, named, focus_id, vector, folders = _seeds(index, query, focus, list(open_files))
    for t in targets:
        if t in g.symbols:
            seeds[t] = max(seeds.get(t, 0.0), FOCUS_SEED)
    comp_scores, top_comps, show_arch = _top_down(index, arch, query, seeds) if arch is not None and query else ({}, [], False)
    ppr, fetched = push_ppr(seeds, g.neighbours) if seeds else ({}, 0)

    candidates = [s for s in dict.fromkeys(list(vector) + top(ppr, 40) + list(named) + list(seeds))
                  if s in g.symbols and g.symbols[s].kind not in ("module", "folder")]
    in_component = {}
    if comp_scores:
        in_component = {s: comp_scores.get(arch.component_of(g.symbols[s].path) or "", 0.0) for s in candidates}
    ranked = hybrid_rank(candidates, {"semantic": vector, "ppr": ppr, "named": named, "component": in_component},
                         WEIGHTS)

    # The symbols this request is about: always read (names written as code; an ordinary word
    # that matched a name only helps the ranking)
    strong = [s for s, w in named.items() if w >= 1.0]
    about = [s for s in dict.fromkeys(([focus_id] if focus_id else []) + strong + list(targets))
             if s in g.symbols and g.symbols[s].kind not in ("module", "folder")]
    context, used = [], 0
    reasons: Dict[str, str] = {s: "the question is about it" for s in about}

    def add(sid: str, row: Optional[dict] = None):
        nonlocal used
        sym = g.symbols[sid]
        size = len(sym.text) + 200
        if sid in [c["id"] for c in context] or (context and used + size > budget):
            return False
        used += size
        row = row or {"id": sid, "score": 0.0, "semantic": 0.0, "ppr": 0.0, "named": 0.0}
        context.append({**row, "path": sym.path, "start": sym.start, "end": sym.end, "kind": sym.kind,
                        "qualname": sym.qualname, "signature": sym.signature,
                        "reason": reasons.get(sid, "ranked")})
        return True

    rows = {r["id"]: r for r in ranked}
    for sid in about:
        add(sid, rows.get(sid))
    for row in ranked:
        if len(context) >= k:
            break
        add(row["id"], row)

    # Two named folders: the code that connects them comes first (both ends of the most telling
    # edges between them, e.g. retrieve -> push_ppr for "core" and "vscode-extension")
    if len(folders) >= 2:
        _, bridges = folder_links(index, list(folders), ppr, per_direction=2)
        for e in bridges:
            for sid in (e.src, e.dst):
                if g.symbols[sid].kind not in ("module", "folder"):
                    reasons.setdefault(sid, "connects folders the question names")
                    add(sid, rows.get(sid))

    # Each folder the question names: its best symbols, even past k
    for folder in folders:
        have = sum(1 for c in context if _in_folder(c["path"], folder))
        inside = [r for r in ranked if _in_folder(g.symbols[r["id"]].path, folder)]
        if len(inside) < FOLDER_MIN_CONTEXT:  # nothing ranked there: its best by PPR
            inside += [{"id": s, "score": 0.0, "semantic": 0.0, "ppr": 0.0, "named": 0.0}
                       for s in top(ppr, 200) if s in g.symbols and g.symbols[s].kind not in ("module", "folder")
                       and _in_folder(g.symbols[s].path, folder)][:FOLDER_MIN_CONTEXT]
        for row in inside:
            if have >= FOLDER_MIN_CONTEXT:
                break
            reasons.setdefault(row["id"], f"in {folder}/, which the question names")
            if add(row["id"], rows.get(row["id"], row)):
                have += 1

    # Coverage: before changing a function, all its callers must be in context. With a language
    # server, its exact callers first replace the parser's guesses.
    missing = []
    settled = index.settle_callers(about) if mode == "edit" and about else None
    if mode == "edit" and about:
        need = required_by(about, g.callers, limit=MAX_REQUIRED * 3)
        missing = uncovered(list(need), [c["id"] for c in context])
        for sid in missing[:MAX_REQUIRED]:
            names = ", ".join(g.symbols[t].qualname for t in need[sid])
            reasons[sid] = f"calls {names} (coverage check)"
            add(sid, rows.get(sid))
    links = relationships(index, ppr, [c["id"] for c in context], list(folders))
    components = [{"id": c, "name": arch.components[c]["name"], "score": round(s, 3)}
                  for c, s in list(comp_scores.items())[:5]] if comp_scores else []
    arch_text = boundary_code = ""
    if show_arch:
        from . import architecture as A
        arch_text = A.text_for(arch, top_comps)
        boundary_code = A.boundary_snippets(index, arch, top_comps, query)
    return {
        "components": components,
        "architecture": arch_text,
        "boundaryCode": boundary_code,
        "context": context,
        "about": about,
        "focus": focus_id,
        "named": list(named),
        "folders": list(folders),
        "relations": links["relations"],
        "folderLinks": links["folderLinks"],
        "ppr": ppr,
        "fetched": fetched,
        "uncoveredCallers": [m for m in missing if m not in {c["id"] for c in context}],
        "settled": settled,
    }


def graph_view(index: Index, scores: Dict[str, float], highlight: List[str] = (),
               limit: int = GRAPH_VIEW_NODES) -> dict:
    """Nodes (best PPR first, plus the highlighted ones) and the edges between them, for the
    webview."""
    g = index.graph
    chosen = [s for s in dict.fromkeys(list(highlight) + top(scores, limit)) if s in g.symbols][:limit]
    keep = set(chosen)
    best = max([scores.get(s, 0.0) for s in chosen], default=0.0) or 1.0
    nodes = []
    for sid in chosen:
        s = g.symbols[sid]
        nodes.append({"id": sid, "label": s.qualname or s.name, "kind": s.kind, "path": s.path,
                      "line": s.start, "score": scores.get(sid, 0.0) / best,
                      "highlight": sid in highlight})
    edges = [e.to_json() for e in g.edges if e.src in keep and e.dst in keep]
    return {"nodes": nodes, "edges": edges}


def impact(index: Index, changed: List[str], k: int = IMPACT_RESULTS) -> List[dict]:
    """Likely-to-break code: PPR from the changed symbols, pushed only backwards along
    dependencies (callers, subclasses, type users, importers). Direct dependents first."""
    g = index.graph
    seeds = {s: 1.0 for s in changed if s in g.symbols}
    if not seeds:
        return []
    scores, _ = push_ppr(seeds, g.dependents_of)
    direct = {d for s in seeds for d in g.dependents_of([s])[s]}
    out = []
    for sid in top(scores, k * 2, exclude=seeds):
        s = g.symbols[sid]
        if s.kind in ("module", "folder") and sid not in direct:
            continue
        out.append({"id": sid, "path": s.path, "line": s.start, "qualname": s.qualname or s.name,
                    "kind": s.kind, "score": scores[sid], "direct": sid in direct})
    out.sort(key=lambda r: (not r["direct"], -r["score"]))
    return out[:k]


def context_block(index: Index, item: dict, max_chars: int = 12_000) -> str:
    """A context item as the model reads it: where it is, then its code as it is on disk now
    (in full: the indexed copy is shortened, and an edit must see every line it replaces)."""
    sym = index.graph.symbols[item["id"]]
    try:
        code = "".join(index.read(sym.path).splitlines(keepends=True)[sym.start - 1:sym.end])
    except OSError:
        code = sym.text
    if len(code) > max_chars and item.get("reason") != "the question is about it":
        code = code[:max_chars] + "\n# ... (cut)\n"
    # "path :: Qualified.name" is also how edit blocks name a symbol (edits.py): a model that
    # copies the header gets it right
    return (f"### {sym.path} :: {sym.qualname}   ({sym.kind}, lines {sym.start}-{sym.end})\n"
            f"```{index.fence_language(sym.path)}\n{code.rstrip()}\n```")
