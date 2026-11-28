"""
Structural questions, answered from the code graph instead of from retrieved code:

  callers   "who calls push_ppr?", "where is format_name used?", "callers of LocalPPR.scores"
  callees   "what does retrieve call?", "what does the core depend on?"
  impact    "what depends on push_ppr?", "what breaks if I change Index.update?"
  between   "does the vscode-extension use core?", "are functions in core called by query_processor?"

A target is a symbol written as code (`push_ppr`, LocalPPR.scores), a folder ("core",
"vscode-extension/sidecar"), or the symbol under the cursor ("who calls this?"). The result is
exact for what static analysis resolved (edges it guessed are marked), it covers the whole
repository rather than the dozen symbols a context holds, and the model only has to explain it.
Questions that don't match go through the normal search; a matching question whose target
can't be found does too.
"""
import re
from collections import Counter
from typing import Dict, List, Optional, Set

from . import retrieve as R
from .index import Index

MAX_FACTS = 40
_DEPENDENCY = ("calls", "renders", "uses", "extends", "implements", "uses_type", "imports")
_USE_WORDS = r"(?:call|calls|called|calling|use|uses|used|using|invoke|invokes|invoked|import|imports|imported|depend|depends|rel(?:y|ies) on|render|renders|rendered)"

_IMPACT = re.compile(r"\b(?:what|which|who)\b[^?]*\b(?:depends? on|breaks?|affected|impacted|relies on)\b"
                     r"|\bimpact of\b|\bif (?:i|we|you) (?:change|modify|remove|rename|delete)\b", re.I)
_CALLEES = re.compile(r"\b(?:what|which)\b[^?]*\b(?:does|do|did)\b[^?]*\b(?:call|use|invoke|depend on|import|render)\s*\??\s*$"
                      r"|\bcallees? of\b|\bwhat is called by\b", re.I)
_CALLERS = re.compile(r"\b(?:who|what|which)(?:\s+\w+){0,3}\s+(?:calls?|uses?|invokes?|renders?|imports?)\b"
                      r"|\bcallers? of\b|\bwhere\b[^?]*\b(?:called|used|invoked|rendered|imported)\b"
                      r"|\b(?:called|used|invoked|imported|referenced|needed|required)\s+(?:in|by|from|within|across)\b"
                      r"|\bwhich\b[^?]*\b(?:use|uses|used|call|calls|called|import|imports|imported|relies|rely)\b"
                      r"|\b(?:users|consumers|dependents)\s+of\b", re.I)
_BETWEEN = re.compile(r"^\s*(?:does|do|did|is|are|was|were|can|will)\b[^?]*\b" + _USE_WORDS + r"\b", re.I)
_THIS = re.compile(r"\b(?:this|it|here|the cursor|current (?:function|method|class|symbol))\b", re.I)
# Where a structural question puts its target: "what does ___ call", "who calls ___"
_NAME = r"`?([A-Za-z_][\w.\-/]*)`?"
_SLOTS = [
    re.compile(r"\b(?:what|which)(?:\s+\w+)?\s+(?:does|do|did)\s+(?:the\s+)?" + _NAME, re.I),
    re.compile(r"\b(?:calls?|uses?|invokes?|renders?|imports?)\s+(?:the\s+)?" + _NAME, re.I),
    re.compile(r"\b(?:callers?|callees?|impact|dependents?)\s+of\s+(?:changing\s+)?(?:the\s+)?" + _NAME, re.I),
    re.compile(r"\bwhere\s+(?:is|are)\s+(?:the\s+)?" + _NAME, re.I),
    re.compile(r"\bdepends?\s+on\s+(?:the\s+)?" + _NAME, re.I),
    re.compile(r"\b(?:change|modify|remove|rename|delete)\s+(?:the\s+)?" + _NAME, re.I),
]
_NOT_TARGETS = {"this", "it", "here", "that", "what", "which", "who", "a", "an", "function", "method", "class", "code"}
# Static analysis can miss calls whose receiver type it can't tell (Python, JavaScript)
_CAVEAT = ("static analysis can miss calls through an object whose type it can't tell "
           "(obj.method()), and calls from outside the repository")


class Target:
    """A symbol (with what it contains, e.g. a class's methods) or a folder."""

    def __init__(self, index: Index, label: str, symbol: str = "", folder: str = ""):
        self.index, self.label, self.symbol, self.folder = index, label, symbol, folder

    def members(self) -> Set[str]:
        g = self.index.graph
        if self.folder:
            return {sid for sid, s in g.symbols.items() if R._in_folder(s.path, self.folder)}
        out, todo = set(), [self.symbol]
        while todo:  # the symbol and everything nested in it
            sid = todo.pop()
            if sid in out:
                continue
            out.add(sid)
            todo += [e.dst for e in g.out.get(sid, []) if e.kind == "contains"]
        return out

    def to_json(self) -> dict:
        return {"label": self.label, "symbol": self.symbol, "folder": self.folder}


def _slot_target(index: Index, query: str, focus_id: Optional[str]) -> Optional[Target]:
    """The target in the question's slot, by any name (even a plain word: "what does retrieve
    call?" means the symbol `retrieve`)."""
    g = index.graph
    for pattern in _SLOTS:
        m = pattern.search(query)
        if not m:
            continue
        word = m.group(1).strip(".").rstrip("/")
        if word.lower() in _NOT_TARGETS:
            if focus_id and g.symbols[focus_id].kind != "module":
                return Target(index, g.symbols[focus_id].qualname or focus_id, symbol=focus_id)
            continue
        folders = g.named_folders(word)
        if folders:
            f = next(iter(folders))
            return Target(index, f"{f}/", folder=f)
        hits = [s for s in g.seed_candidates(word) if g.symbols[s].kind != "module"]
        if len(hits) == 1:
            s = hits[0]
            return Target(index, g.symbols[s].qualname or s, symbol=s)
    return None


def _targets(index: Index, query: str, focus_id: Optional[str]) -> List[Target]:
    g = index.graph
    found = []
    # Folders first, in the order the question names them
    folders = g.named_folders(query)
    order = sorted(folders, key=lambda f: _position(query, f))
    found += [Target(index, f"{f}/", folder=f) for f in order]
    # Symbols written as code (an ordinary word that happens to be a name isn't a target)
    named = [s for s, w in g.seed_candidates(query).items() if w >= 1.0 and g.symbols[s].kind != "module"]
    named.sort(key=lambda s: _position(query, g.symbols[s].name))
    found += [Target(index, g.symbols[s].qualname or g.symbols[s].path, symbol=s) for s in named]
    found.sort(key=lambda t: _position(query, t.folder or g.symbols[t.symbol].name))
    if focus_id and (not found or _THIS.search(query)) and g.symbols[focus_id].kind != "module":
        found.insert(0, Target(index, g.symbols[focus_id].qualname or focus_id, symbol=focus_id))
    return found


def _position(query: str, name: str) -> int:
    i = query.lower().find(name.split("/")[-1].lower())
    return i if i >= 0 else len(query)


def classify(index: Index, query: str, focus_id: Optional[str] = None) -> Optional[dict]:
    """The structural question this is, with its targets, or None."""
    targets = _targets(index, query, focus_id)
    if not targets:
        slot = _slot_target(index, query, focus_id)
        targets = [slot] if slot else []
    if not targets:
        return None
    if len(targets) >= 2 and _BETWEEN.search(query):
        return {"kind": "between", "targets": targets[:2]}
    for kind, pattern in (("impact", _IMPACT), ("callees", _CALLEES), ("callers", _CALLERS)):
        if pattern.search(query):
            slot = _slot_target(index, query, focus_id)
            return {"kind": kind, "targets": [slot] if slot else targets[:1]}
    if len(targets) >= 2 and re.search(r"\b" + _USE_WORDS + r"\b", query, re.I):
        return {"kind": "between", "targets": targets[:2]}
    return None


def component_of(path: str) -> str:
    """The top-level folder a file is in ("query_processor/"), or the file itself at the root."""
    return path.split("/", 1)[0] + "/" if "/" in path else path


def _spread(edges: list, key, limit: int) -> list:
    """Up to `limit` edges, taken in turn from each group (component), so one component with
    many edges can't fill the list on its own."""
    groups: Dict[str, list] = {}
    for e in edges:
        groups.setdefault(key(e), []).append(e)
    out, queues = [], list(groups.values())
    while queues and len(out) < limit:
        for q in list(queues):
            if q:
                out.append(q.pop(0))
            if not q:
                queues.remove(q)
            if len(out) >= limit:
                break
    return out


def _by_component(paths: List[str]) -> str:
    """'query_processor/ (13 in 4 files: app/llm_client.py, ...); vscode-extension/ (...)'."""
    comps: Dict[str, Counter] = {}
    for p in paths:
        comps.setdefault(component_of(p), Counter())[p] += 1
    parts = []
    for comp, files in sorted(comps.items(), key=lambda kv: -sum(kv[1].values())):
        names = [f[len(comp):] if f.startswith(comp) else f for f, _ in files.most_common(6)]
        more = f", +{len(files) - 6} more" if len(files) > 6 else ""
        parts.append(f"{comp} ({sum(files.values())} in {len(files)} file(s): {', '.join(names)}{more})")
    return "; ".join(parts)


def _fact(index: Index, e, why: str = "") -> dict:
    fact = R.relation(index.graph, e)
    s = index.graph.symbols[e.src]
    fact.update(path=s.path, line=s.start, why=why)
    return fact


def _runtime_facts(arch, kind: str, targets: List[Target]) -> List[dict]:
    """Runtime connections (architecture.py) into, out of or between the target folders."""
    from .architecture import connection_text

    def side(cid: str, folder: str, path_of_side: List[str]) -> bool:
        comp = arch.components.get(cid) or {}
        p = comp.get("path") or ""
        if p and (p == folder or p.startswith(folder + "/")):
            return True  # the component is (in) the folder
        return any(R._in_folder(x, folder) for x in path_of_side)  # the folder is part of it

    out = []
    for c in arch.connections:
        if c["kind"] in ("code", "compose"):
            continue
        ev = c["evidence"]
        src_paths = [e["path"] for e in ev if arch.component_of(e["path"]) == c["from"]]
        dst_paths = [e["path"] for e in ev if arch.component_of(e["path"]) == c["to"]]
        a = targets[0].folder
        if kind == "between" and len(targets) > 1 and targets[1].folder:
            b = targets[1].folder
            hit = (side(c["from"], a, src_paths) and side(c["to"], b, dst_paths)) or \
                  (side(c["from"], b, src_paths) and side(c["to"], a, dst_paths))
        elif kind in ("callers", "impact"):
            hit = bool(a) and side(c["to"], a, dst_paths) and not side(c["from"], a, src_paths)
        elif kind == "callees":
            hit = bool(a) and side(c["from"], a, src_paths) and not side(c["to"], a, dst_paths)
        else:
            hit = False
        if hit:
            first = ev[0] if ev else {}
            out.append({"source": c["from"], "target": c["to"], "kind": c["kind"], "guessed": True,
                        "text": connection_text(arch, c), "path": first.get("path", ""), "line": first.get("line", 0),
                        "why": "runtime connection (detected)"})
    return out


def answer(index: Index, question: dict, arch=None) -> dict:
    """Runs the graph query: {kind, targets, summary, facts: [relation + path/line], symbols}.
    With the architecture, questions about folders also get runtime connections."""
    g = index.graph
    kind, targets = question["kind"], question["targets"]
    t = targets[0]
    inside = t.members()
    facts, summary = [], ""
    settled = None
    if kind in ("callers", "impact") and t.symbol:
        # With a language server: its exact callers replace the parser's guesses first
        settled = index.settle_callers([t.symbol])

    if kind in ("callers", "callees"):
        incoming = kind == "callers"
        edges = []
        for sid in inside:
            g.callers(sid)  # loads language-server edges for symbols only a server knows
            for e in (g.into.get(sid, []) if incoming else g.out.get(sid, [])):
                other = e.src if incoming else e.dst
                if e.kind in _DEPENDENCY and other not in inside:
                    edges.append(e)
        edges.sort(key=lambda e: (e.guessed, R._KIND_ORDER.get(e.kind, 4), e.src, e.dst))
        sure = sum(1 for e in edges if not e.guessed)
        kinds = Counter(e.kind for e in edges)
        if incoming:
            summary = (f"{len(edges)} reference(s) to {t.label} from outside it"
                       f" ({', '.join(f'{n} {k}' for k, n in kinds.most_common())})" if edges
                       else f"Static analysis found nothing in the indexed code that calls or uses {t.label}")
        else:
            summary = (f"{t.label} uses {len(edges)} other symbol(s)"
                       f" ({', '.join(f'{n} {k}' for k, n in kinds.most_common())})" if edges
                       else f"Static analysis found no calls or uses from {t.label} to other indexed code")
        if edges and sure < len(edges):
            summary += f"; {len(edges) - sure} of them guessed (receiver type unknown)"
        # Where they are, by component (top-level folder) and file
        side = (lambda e: g.symbols[e.src].path) if incoming else (lambda e: g.symbols[e.dst].path)
        if edges:
            summary += (". From " if incoming else ". Into ") + _by_component([side(e) for e in edges])
        facts = [_fact(index, e) for e in _spread(edges, lambda e: component_of(side(e)), MAX_FACTS)]

    elif kind == "impact":
        seeds = sorted(inside)[:200]
        affected = [a for a in R.impact(index, seeds, k=MAX_FACTS) if a["id"] not in inside]
        direct = [a for a in affected if a["direct"]]
        summary = (f"{len(direct)} direct and {len(affected) - len(direct)} indirect dependent(s) of {t.label}"
                   f". In {_by_component([a['path'] for a in affected])}"
                   if affected else f"Static analysis found nothing in the indexed code that depends on {t.label}")
        facts = [{"source": a["id"], "target": t.symbol or t.folder, "kind": "depends on",
                  "text": f"{a['path']}::{a['qualname']} {'directly' if a['direct'] else 'indirectly'} depends on {t.label}",
                  "path": a["path"], "line": a["line"], "guessed": False, "why": ""} for a in affected]

    else:  # between
        a, b = targets[0], targets[1]
        ma, mb = a.members(), b.members()
        lines = []
        for x, y, mx, my in ((a, b, ma, mb), (b, a, mb, ma)):
            edges = [e for sid in mx for e in g.out.get(sid, []) if e.dst in my and e.kind in _DEPENDENCY]
            edges.sort(key=lambda e: (e.guessed, R._KIND_ORDER.get(e.kind, 4), e.src, e.dst))
            kinds = Counter(e.kind for e in edges)
            lines.append(f"{x.label} -> {y.label}: {len(edges)} edge(s)"
                         + (f" ({', '.join(f'{n} {k}' for k, n in kinds.most_common())})" if edges else ""))
            facts += [_fact(index, e) for e in edges[:MAX_FACTS // 2]]
        summary = "; ".join(lines)

    # Folders: connections no import shows (HTTP, gRPC, processes, datastores), from the architecture
    if arch is not None and any(x.folder for x in targets):
        runtime = _runtime_facts(arch, kind, targets)
        if runtime:
            summary += "; runtime connections (detected, not imports): " + "; ".join(f["text"] for f in runtime[:6])
            facts = runtime + facts

    exact = bool(settled and settled.get("checked"))
    if exact:
        summary += " (callers checked with the language server)"
    elif kind != "between" or not facts:
        summary += f"; {_CAVEAT}"
    symbols = [t.symbol] if t.symbol and g.symbols[t.symbol].kind not in ("module", "folder") else []
    for f in facts:
        for sid in (f["source"], f["target"]):
            if sid in g.symbols and g.symbols[sid].kind not in ("module", "folder") and sid not in symbols:
                symbols.append(sid)
    return {"kind": kind, "targets": [x.to_json() for x in targets], "summary": summary,
            "facts": facts, "symbols": symbols, "exact": exact}


def answer_text(result: dict) -> str:
    """The graph's answer as the model reads it."""
    lines = [f"Graph query ({result['kind']}) over the whole repository: {result['summary']}."]
    lines += [f"- {f['text']}" for f in result["facts"]]
    if len(result["facts"]) >= MAX_FACTS:
        lines.append(f"- (only the first {MAX_FACTS} are listed)")
    return "\n".join(lines)
