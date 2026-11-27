"""
The code graph: symbols as nodes, code relationships as edges, rebuilt in memory from every
file's parse result (parse_python.py) whenever files change. Resolution is static:
  calls       a function/method calls another, or instantiates a class (`new` in JS/TS)
  renders     a React component renders another (<Child />)
  imports     a module imports a module or a symbol
  extends     a class subclasses another; an interface extends another
  implements  a TS class implements an interface
  contains    module -> class/function, class -> method, function -> nested function
  uses_type   a signature or annotation names a class, interface or type alias

Python modules are found by dotted name ("pkg.models"), JS/TS modules by path ("src/api",
"src/ui" for src/ui/index.ts), through `export * from` barrels and default exports.

Python and JavaScript are dynamic: a call on a receiver whose type isn't known ("obj.save()") is
linked to every method of that name only when there are at most MAX_GUESSES of them, as a
guessed edge at half weight. Calls into the standard library or packages outside the
repository are dropped.

For PPR the graph is undirected but direction-aware: each relationship has one weight from
the source (caller -> callee) and another back (callee -> caller), scaled by how often the
source uses the target.

Files no parser reads (Java, Go, C#, ... : whatever VS Code has a language server for) come
in as symbols only (parser "lsp"). Their edges are loaded lazily from the language server
(`external`, set by index.py) the first time PPR, the coverage check or impact analysis
reaches one of their symbols: push PPR only ever asks for the neighbours of nodes it reaches,
so the whole graph never has to be built. Those edges are `calls` (call hierarchy), `extends`
(type hierarchy) and `uses` (references, when a server has no call hierarchy).
"""
import math
import re
from collections import defaultdict
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

from .parse_python import FileResult, Symbol

# (forward weight, backward weight) per relationship
EDGE_WEIGHTS = {
    "calls": (1.0, 0.7),
    "renders": (0.9, 0.6),
    "extends": (0.8, 0.5),
    "implements": (0.7, 0.4),
    "uses_type": (0.5, 0.3),
    "uses": (0.6, 0.4),       # a language server's references: a call or a type use
    "contains": (0.5, 0.8),
    "contains_dir": (0.3, 0.1),  # folder -> subfolder / module: light, so a folder isn't a hub
    "imports": (0.3, 0.2),
}
TYPE_KINDS = ("class", "interface", "type")
# Reference kind (parser) -> edge kind
_EDGE_OF_REF = {"call": "calls", "render": "renders", "uses_type": "uses_type", "implements": "implements"}
GUESSED_FACTOR = 0.5
MAX_GUESSES = 3
# Method names too common to guess a receiver from ("x.get()" could be anything)
_TOO_COMMON = {"get", "set", "add", "put", "pop", "run", "update", "append", "extend", "items",
               "keys", "values", "join", "split", "strip", "format", "read", "write", "close",
               "open", "copy", "clear", "remove", "insert", "index", "count", "sort", "encode",
               "decode", "lower", "upper", "replace", "startswith", "endswith", "load", "dump",
               "send", "start", "stop", "__init__", "__call__", "__enter__", "__exit__",
               # JavaScript built-ins
               "log", "warn", "error", "then", "catch", "finally", "map", "filter", "forEach",
               "reduce", "push", "find", "includes", "some", "every", "slice", "splice", "concat",
               "toString", "json", "fetch", "emit", "on", "off", "has", "delete", "trim", "test",
               "exec", "match", "parse", "stringify", "assign", "entries", "resolve", "reject"}
# Edges along which a change can break the source: it depends on the target
DEPENDENCY_KINDS = ("calls", "renders", "extends", "implements", "uses_type", "uses", "imports")
CALLER_KINDS = ("calls", "renders", "uses")
# Loads edges for nodes from a language server: ids -> {id: [(source, target, kind, count)]}
External = Callable[[List[str]], Dict[str, List[Tuple[str, str, str, int]]]]
_JS_INDEX = "index"

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*")
_FOLDER_TOKEN = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_\-./]*")
# Folder names too common to mean one place in a repository
_GENERIC_FOLDERS = {"src", "app", "apps", "lib", "libs", "test", "tests", "spec", "util", "utils", "main",
                    "docs", "doc", "code", "data", "static", "media", "scripts", "script", "out", "build",
                    "dist", "bin", "the", "a", "an", "it", "this", "that", "is", "are", "in", "of", "to"}
PLAIN_WORD_WEIGHT = 0.3


def looks_like_code(token: str, query: str) -> bool:
    """An identifier written as code: snake_case, camelCase/PascalCase, dotted, with digits,
    or quoted in backticks."""
    if "_" in token or "." in token or any(c.isdigit() for c in token):
        return True
    if token[:1].isupper() or any(c.isupper() for c in token[1:]):
        return True
    return f"`{token}" in query or f"{token}(" in query


class Edge:
    __slots__ = ("src", "dst", "kind", "count", "guessed")

    def __init__(self, src, dst, kind, count=1, guessed=False):
        self.src, self.dst, self.kind, self.count, self.guessed = src, dst, kind, count, guessed

    def to_json(self) -> dict:
        return {"source": self.src, "target": self.dst, "kind": self.kind,
                "count": self.count, "guessed": self.guessed}


class CodeGraph:
    def __init__(self, files: Iterable[FileResult]):
        self.files: Dict[str, FileResult] = {f.path: f for f in files}
        self.symbols: Dict[str, Symbol] = {}
        self.by_name: Dict[str, List[str]] = defaultdict(list)
        self.members: Dict[str, Dict[str, str]] = defaultdict(dict)   # container id -> name -> id
        self.module_ids: Dict[str, str] = {}                           # Python: dotted module -> module id
        self.js_modules: Dict[str, str] = {}                           # JS/TS: path without extension -> module id
        for f in self.files.values():
            for s in f.symbols:
                self.symbols[s.id] = s
                if s.kind != "module":
                    self.by_name[s.name].append(s.id)
                    self.members[s.parent][s.name] = s.id
                elif f.parser == "lsp":
                    pass  # imports between these files aren't resolved here
                elif f.lang == "python":
                    self.module_ids[f.module] = s.id
                else:
                    self.js_modules[f.module] = s.id
        # Symbols whose edges come from a language server, on demand (see the module docstring)
        self.external: Optional[External] = None
        self.external_paths = {p for p, f in self.files.items() if f.parser == "lsp"}
        self._external_loaded: set = set()
        # "./ui" means ui/index.ts unless there's a ui.ts
        for key, mid in list(self.js_modules.items()):
            if key.endswith("/" + _JS_INDEX) or key == _JS_INDEX:
                self.js_modules.setdefault(key[:-len(_JS_INDEX)].rstrip("/"), mid)
        self._suffix_index = self._build_suffix_index()
        self.edges: List[Edge] = []
        self.out: Dict[str, List[Edge]] = defaultdict(list)
        self.into: Dict[str, List[Edge]] = defaultdict(list)
        self.unresolved = 0
        self._resolve_all()
        self._add_folders()

    # --- module lookup ---------------------------------------------------------------------
    def _build_suffix_index(self) -> Dict[str, List[str]]:
        """'a.b.c' is importable as 'c', 'b.c' or 'a.b.c' depending on where Python runs
        from: index every suffix."""
        index = defaultdict(list)
        for dotted, mid in self.module_ids.items():
            parts = dotted.split(".")
            for i in range(len(parts)):
                index[".".join(parts[i:])].append(mid)
        return index

    def _module(self, dotted: str, near: str) -> Optional[str]:
        """The module `dotted` names, as seen from the file `near` (the closest one when
        several match: the import that runs from near's own directory wins)."""
        found = self._suffix_index.get(dotted)
        if not found:
            return None
        if len(found) == 1:
            return found[0]
        near_dir = near.rsplit("/", 1)[0]

        def shared(mid):
            path = self.symbols[mid].path
            return len(_common_prefix(path.split("/"), near_dir.split("/")))
        return max(found, key=shared)

    # --- resolution ------------------------------------------------------------------------
    def _resolve_target(self, target: str, near: str, depth: int = 0) -> Optional[str]:
        """An import's target: "<module path>#<export>[.member...]" (JS/TS) or a dotted name."""
        if "#" not in target:
            return self._resolve_dotted(target, near, depth)
        key, _, name = target.partition("#")
        mid = self.js_modules.get(key)
        if mid is None:
            return None  # a package, or outside the repository
        parts = name.split(".")
        if parts[0] == "*":  # import * as api: the module itself
            return self._member_path(mid, parts[1:], near, depth) if len(parts) > 1 else mid
        return self._member_path(mid, parts, near, depth)

    def _js_export(self, module_id: str, name: str, near: str, depth: int) -> Optional[str]:
        """What a JS/TS module exports under `name`: a local symbol (also via `export default X`
        or `export { a as b }`), a re-export (`export { X } from`), or a star re-export."""
        f = self.files.get(self.symbols[module_id].path)
        if f is None or depth > 4:
            return None
        local = f.exports.get(name, name)
        hit = self.members.get(module_id, {}).get(local)
        if hit:
            return hit
        if local in f.imports:  # re-exported, or `export default X` of an imported X
            found = self._resolve_target(f.imports[local], f.path, depth + 1)
            if found:
                return found
        for key, star in f.imports.items():  # export * from "./x"
            if key.startswith("*"):
                found = self._resolve_target(star.replace("#*", f"#{name}"), f.path, depth + 1)
                if found:
                    return found
        return None

    def _resolve_dotted(self, dotted: str, near: str, depth: int = 0) -> Optional[str]:
        """A symbol for an absolute dotted name: the longest module prefix, then members."""
        parts = dotted.split(".")
        for cut in range(len(parts), 0, -1):
            mid = self._module(".".join(parts[:cut]), near)
            if mid:
                return self._member_path(mid, parts[cut:], near, depth)
        return None

    def _member_path(self, container: str, rest: List[str], near: str, depth: int) -> Optional[str]:
        current = container
        for name in rest:
            sym = self.symbols[current]
            f = self.files.get(sym.path)
            if sym.kind == "module" and f is not None and f.lang != "python":
                nxt = self._js_export(current, name, near, depth)
            else:
                nxt = self.members.get(current, {}).get(name)
            if nxt is None and sym.kind == "class":
                nxt = self._inherited(current, name)
            if nxt is None and sym.kind == "module" and depth < 3 and f is not None and f.lang == "python":
                # "from .models import User" re-exported by a package's __init__
                if name in f.imports:
                    nxt = self._resolve_dotted(f.imports[name], f.path, depth + 1)
            if nxt is None:
                return None
            current = nxt
        return current

    def _bases(self, class_id: str) -> List[str]:
        return [e.dst for e in self.out.get(class_id, []) if e.kind == "extends"]

    def _inherited(self, class_id: str, name: str, seen=None) -> Optional[str]:
        seen = seen or set()
        for base in self._bases(class_id):
            if base in seen:
                continue
            seen.add(base)
            found = self.members.get(base, {}).get(name) or self._inherited(base, name, seen)
            if found:
                return found
        return None

    def _enclosing_class(self, sym: Symbol) -> Optional[str]:
        current = sym
        while current and current.parent:
            parent = self.symbols.get(current.parent)
            if parent is None:
                return None
            if parent.kind == "class":
                return parent.id
            current = parent
        return None

    def _resolve_name(self, target: str, sym: Symbol) -> Tuple[Union[str, List[str], None], bool]:
        """(symbol id, guessed) for a reference written in `sym`; a guess may give a list."""
        f = self.files[sym.path]
        head, _, rest = target.partition(".")
        rest_parts = rest.split(".") if rest else []

        if head in ("self", "cls") and rest_parts:
            cls = self._enclosing_class(sym)
            if cls:
                found = self._member_path(cls, rest_parts, sym.path, 0)
                if found:
                    return found, False
            return self._guess(rest_parts[-1])
        if head == "super" and rest_parts:
            cls = self._enclosing_class(sym)
            if cls:
                for base in self._bases(cls):
                    found = self.members.get(base, {}).get(rest_parts[0]) or self._inherited(base, rest_parts[0])
                    if found:
                        return found, False
            return None, False
        if head == "?":
            return self._guess(rest_parts[-1]) if rest_parts else (None, False)

        # Enclosing scopes, innermost first: nested defs, then the module. A class body's
        # names aren't visible inside its methods (Python scoping), only in the body itself.
        scope = sym
        while scope is not None:
            if scope.kind != "class" or scope.id == sym.id:
                hit = self.members.get(scope.id, {}).get(head)
                if hit:
                    found = self._member_path(hit, rest_parts, sym.path, 0) if rest_parts else hit
                    if found:
                        return found, False
            scope = self.symbols.get(scope.parent) if scope.parent else None
        if head in f.imports:
            dotted = f.imports[head] + (f".{rest}" if rest else "")
            found = self._resolve_target(dotted, sym.path)
            if found:
                return found, False
            return None, False  # imported from outside the repository
        if rest_parts and f.lang == "python":
            # "module.func" written with the module's dotted name (import a.b; a.b.f())
            found = self._resolve_dotted(target, sym.path)
            if found:
                return found, False
            return self._guess(rest_parts[-1])
        if rest_parts:  # obj.method() in JS/TS with obj unknown
            return self._guess(rest_parts[-1])
        return None, False

    def _guess(self, method: str) -> Tuple[Optional[List[str]], bool]:
        """Every method of this name, when there are few enough to be a useful guess."""
        if method in _TOO_COMMON or len(method) < 3:
            return None, False
        candidates = [i for i in self.by_name.get(method, []) if self.symbols[i].kind == "method"]
        if 0 < len(candidates) <= MAX_GUESSES:
            return candidates, True
        return None, False

    def _add_edge(self, src, dst, kind, count=1, guessed=False):
        if src == dst:
            return
        e = Edge(src, dst, kind, count, guessed)
        self.edges.append(e)
        self.out[src].append(e)
        self.into[dst].append(e)

    def _add_folders(self):
        """A node per folder that holds indexed files ("core/graphrag_core/::"), with
        `contains_dir` edges to its subfolders and modules. Light edges: code in the same
        folder is a little related, and the graph view can show the project's layout. Folders
        aren't code: they're never embedded, read as context or counted as dependents."""
        self.folder_ids: Dict[str, str] = {}
        modules = [s for s in self.symbols.values() if s.kind == "module"]
        for m in modules:
            parts = m.path.split("/")[:-1]
            for i in range(1, len(parts) + 1):
                d = "/".join(parts[:i])
                if d not in self.folder_ids:
                    fid = f"{d}/::"
                    self.folder_ids[d] = fid
                    self.symbols[fid] = Symbol(id=fid, path=f"{d}/", kind="folder", name=parts[i - 1], qualname="",
                                               start=0, end=0, signature=d, doc="", text="")
        for d, fid in self.folder_ids.items():
            parent = d.rsplit("/", 1)[0] if "/" in d else ""
            if parent:
                self.symbols[fid].parent = self.folder_ids[parent]
                self._add_edge(self.folder_ids[parent], fid, "contains_dir")
        for m in modules:
            d = m.path.rsplit("/", 1)[0] if "/" in m.path else ""
            if d:
                self._add_edge(self.folder_ids[d], m.id, "contains_dir")

    def _remove_edge(self, e: Edge):
        self.edges.remove(e)
        self.out[e.src].remove(e)
        self.into[e.dst].remove(e)

    def _has_edge(self, src, dst, kind) -> bool:
        return any(e.dst == dst and e.kind == kind for e in self.out.get(src, []))

    # --- edges from a language server --------------------------------------------------------
    def _ensure_external(self, nodes: Iterable[str]):
        """Loads the language-server edges of these nodes, once per graph."""
        if self.external is None:
            return
        todo = [n for n in nodes if n not in self._external_loaded and n in self.symbols
                and self.symbols[n].path in self.external_paths and self.symbols[n].kind != "module"]
        if not todo:
            return
        self._external_loaded.update(todo)
        for rels in self.external(todo).values():
            for src, dst, kind, count in rels:
                if src in self.symbols and dst in self.symbols and src != dst and not self._has_edge(src, dst, kind):
                    self._add_edge(src, dst, kind, count)

    def apply_exact_callers(self, target: str, callers: Iterable[str]) -> Dict[str, int]:
        """A language server's callers of `target` settle the parser's guesses: guessed edges it
        confirms stay (no longer guessed), the others go, and callers the parser missed are
        added. Edges the parser resolved for certain are kept either way."""
        exact = {c for c in callers if c in self.symbols and c != target}
        confirmed = dropped = added = 0
        for e in list(self.into.get(target, [])):
            if e.kind != "calls":
                continue
            if e.src in exact:
                if e.guessed:
                    e.guessed = False
                    confirmed += 1
            elif e.guessed:
                self._remove_edge(e)
                dropped += 1
        have = {e.src for e in self.into.get(target, []) if e.kind in CALLER_KINDS}
        for c in exact - have:
            self._add_edge(c, target, "calls")
            added += 1
        return {"confirmed": confirmed, "dropped": dropped, "added": added}

    def _resolve_all(self):
        for s in self.symbols.values():
            if s.parent and s.parent in self.symbols:
                self._add_edge(s.parent, s.id, "contains")
        # Bases first: method lookups through inheritance need them
        pending = [r for f in self.files.values() for r in f.refs if r.kind != "extends"]
        for f in self.files.values():
            for r in f.refs:
                if r.kind == "extends":
                    found = self._resolve_name(r.target, self.symbols[r.src])[0]
                    if isinstance(found, str) and self.symbols[found].kind in TYPE_KINDS:
                        self._add_edge(r.src, found, "extends", r.count)
        for r in pending:
            sym = self.symbols.get(r.src)
            if sym is None:
                continue
            if r.kind == "import":
                found = self._resolve_target(r.target, sym.path)
                if found:
                    self._add_edge(r.src, found, "imports", r.count)
                continue
            found, guessed = self._resolve_name(r.target, sym)
            if not found:
                self.unresolved += 1
                continue
            kind = _EDGE_OF_REF.get(r.kind, r.kind)
            for target in (found if isinstance(found, list) else [found]):
                target_kind = self.symbols[target].kind
                if kind in ("uses_type", "implements") and target_kind not in TYPE_KINDS:
                    continue
                if kind == "renders" and target_kind not in ("component", "function", "class"):
                    continue
                self._add_edge(r.src, target, kind, r.count, guessed)

    # --- queries ---------------------------------------------------------------------------
    def neighbours(self, nodes: List[str]) -> Dict[str, Dict[str, float]]:
        """For graphrag_core.ppr.push_ppr: both directions, each with its own weight."""
        self._ensure_external(nodes)
        result = {}
        for n in nodes:
            out: Dict[str, float] = defaultdict(float)
            for e in self.out.get(n, []):
                out[e.dst] += _weight(e, forward=True)
            for e in self.into.get(n, []):
                out[e.src] += _weight(e, forward=False)
            result[n] = out
        return result

    def dependents_of(self, nodes: List[str]) -> Dict[str, Dict[str, float]]:
        """Only the symbols that depend on each node (callers, subclasses, users of a type):
        for impact analysis, a change spreads backwards along these. A class's dependents
        include its methods' callers; a module's, its importers."""
        self._ensure_external(nodes)
        result = {}
        for n in nodes:
            out: Dict[str, float] = defaultdict(float)
            for e in self.into.get(n, []):
                if e.kind in DEPENDENCY_KINDS:
                    out[e.src] += _weight(e, forward=True)
            result[n] = out
        return result

    def callers(self, node: str) -> List[str]:
        """Direct callers, and components that render it (for a class, interface or type: its
        subclasses, implementers and the code that uses it)."""
        self._ensure_external([node])
        sym = self.symbols.get(node)
        kinds = DEPENDENCY_KINDS if sym is not None and sym.kind in TYPE_KINDS else CALLER_KINDS
        seen = []
        for e in self.into.get(node, []):
            if e.kind in kinds and not e.guessed and e.src not in seen:
                seen.append(e.src)
        return seen

    def symbol_at(self, path: str, line: int) -> Optional[str]:
        """The innermost symbol containing a line of a file."""
        best = None
        f = self.files.get(path)
        if not f:
            return None
        for s in f.symbols:
            if s.start <= line <= s.end and (best is None or s.end - s.start <= best.end - best.start):
                best = s
        return best.id if best else None

    def seed_candidates(self, query: str, limit: int = 8) -> Dict[str, float]:
        """Symbols the question names: identifiers, Class.method, module or file names.
        Exact case first; case-insensitive for names of 4+ letters."""
        found: Dict[str, float] = {}
        lowered = defaultdict(list)
        for name, ids in self.by_name.items():
            lowered[name.lower()].extend(ids)
        for token in _IDENT.findall(query):
            hits = []
            if "." in token:
                hits = [i for i, s in self.symbols.items() if s.qualname and s.qualname.endswith(token)]
                if not hits:
                    mid = self._module(token[:-3] if token.endswith(".py") else token, "")
                    hits = [mid] if mid else []
                if not hits:  # a file name: "Button.tsx"
                    hits = [i for i, s in self.symbols.items()
                            if s.kind == "module" and (s.path == token or s.path.endswith("/" + token))]
            if not hits:
                hits = list(self.by_name.get(token, []))
            if not hits and len(token) >= 4:
                hits = list(lowered.get(token.lower(), []))
            if not hits and len(token) >= 4:
                hits = [m for d, m in self.module_ids.items() if d.split(".")[-1] == token]
            if 0 < len(hits) <= 5:  # a name used in many places identifies nothing
                # "running", "called": an ordinary word that happens to be a name is a weak hint;
                # push_ppr, Sidecar.running, LocalPPR, `scores`, local_ppr.py are meant as code
                weight = 1.0 if looks_like_code(token, query) else PLAIN_WORD_WEIGHT
                for h in hits:
                    found[h] = max(found.get(h, 0.0), weight)
            if len(found) >= limit:
                break
        return found

    def named_folders(self, query: str) -> Dict[str, List[str]]:
        """Folders the question names ("core", "vscode-extension", "query_processor/app") ->
        the modules in them. Generic folder names (src, app, tests, ...) don't count."""
        dirs: Dict[str, List[str]] = defaultdict(list)
        for s in self.symbols.values():
            if s.kind == "module":
                parts = s.path.split("/")[:-1]
                for i in range(len(parts)):
                    dirs["/".join(parts[:i + 1])].append(s.id)
        found = {}
        for token in _FOLDER_TOKEN.findall(query):
            t = token.strip("`'\"/").lower()
            if not t or t in _GENERIC_FOLDERS:
                continue
            for d, modules in dirs.items():
                if d.lower() == t or d.lower().endswith("/" + t):
                    found[d] = modules
        # A folder inside another named folder adds nothing
        return {d: m for d, m in found.items() if not any(o != d and d.startswith(o + "/") for o in found)}

    def stats(self) -> dict:
        kinds = defaultdict(int)
        for e in self.edges:
            kinds[e.kind] += 1
        sym_kinds = defaultdict(int)
        for s in self.symbols.values():
            if s.kind != "folder":
                sym_kinds[s.kind] += 1
        return {"files": len(self.files), "folders": len(self.folder_ids), "symbols": dict(sym_kinds), "edges": dict(kinds),
                "unresolved": self.unresolved, "languageServerFiles": len(self.external_paths),
                "syntaxErrors": [f.path for f in self.files.values() if f.error]}


def _weight(e: Edge, forward: bool) -> float:
    w = EDGE_WEIGHTS[e.kind][0 if forward else 1] * (1.0 + math.log(max(e.count, 1)))
    return w * GUESSED_FACTOR if e.guessed else w


def _common_prefix(a: List[str], b: List[str]) -> List[str]:
    out = []
    for x, y in zip(a, b):
        if x != y:
            break
        out.append(x)
    return out
