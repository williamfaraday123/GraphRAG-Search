"""
The high-level architecture of a repository, as a graph above the code graph (the C4 model's
levels: system -> containers -> components -> code):

  components   what can be built or deployed on its own: folders with a build file
               (package.json, pyproject.toml, requirements.txt, pom.xml, go.mod, Cargo.toml,
               Dockerfile, ...) and docker-compose services. With fewer than two of those, the
               module graph is clustered with Leiden (graphrag_core.leiden: software
               architecture recovery), with folder structure as a light extra signal.
  infrastructure
               databases, queues and model APIs the code talks to (Neo4j, Milvus, Redis, an
               OpenAI-compatible endpoint, ...), from client code and compose images.
  connections  - code: every symbol-level edge between two components, summed up;
               - runtime, which no import shows (heuristic, from detectors.json):
                 HTTP (a client's URL matched to a server's route), gRPC (stub <-> servicer),
                 datastores and model APIs (client constructors, URLs), processes started with
                 `-m module`, docker-compose depends_on and service names in environment values.
               Every connection keeps its evidence (file and line), so the diagram can be
               checked, and impact analysis can follow it across the boundary.

Inside a component, Leiden clusters its modules into sub-components (C4's "components"), and
those open into modules, then the symbol graph.

Everything here is static and recomputed when the code graph changes (cached on it). Optional
model-written summaries per component are cached on disk by a hash of what they summarise.
"""
import hashlib
import json
import math
import os
import re
import time
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

from graphrag_core.leiden import leiden
from graphrag_core.ppr import push_ppr

from . import miniyaml
from .index import SKIP_DIRS, Index, looks_minified

MANIFESTS = {"package.json": "npm", "pyproject.toml": "Python", "setup.py": "Python", "setup.cfg": "Python",
             "requirements.txt": "Python", "Pipfile": "Python", "pom.xml": "Maven", "build.gradle": "Gradle",
             "build.gradle.kts": "Gradle", "go.mod": "Go", "Cargo.toml": "Cargo", "Gemfile": "Ruby",
             "composer.json": "PHP", "mix.exs": "Elixir", "pubspec.yaml": "Dart", "Package.swift": "Swift",
             "Dockerfile": "Docker"}
MANIFEST_SUFFIXES = {".csproj": ".NET", ".fsproj": ".NET", ".sln": ".NET"}
COMPOSE = re.compile(r"^(?:docker-)?compose(?:\.[\w-]+)?\.ya?ml$")
LANGS = {".py": "python", ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript", ".cjs": "javascript",
         ".ts": "typescript", ".tsx": "typescript", ".mts": "typescript", ".cts": "typescript",
         ".java": "java", ".kt": "kotlin", ".kts": "kotlin", ".scala": "scala", ".go": "go", ".rs": "rust",
         ".cs": "csharp", ".fs": "fsharp", ".rb": "ruby", ".php": "php", ".swift": "swift", ".c": "c",
         ".h": "c", ".cc": "cpp", ".cpp": "cpp", ".hpp": "cpp", ".m": "objective-c", ".dart": "dart",
         ".ex": "elixir", ".exs": "elixir", ".vue": "vue", ".svelte": "svelte", ".proto": "protobuf"}
EXTRA_SKIP = {"target", "bin", "obj", ".gradle", "Pods", ".dart_tool", ".terraform", "generated"}
GENERATED = re.compile(r"(_pb2(_grpc)?\.py|\.pb\.go|_grpc\.pb\.go|Grpc\.java|\.min\.js|\.d\.ts|\.[0-9a-f]{8,}(\.chunk)?\.m?js)$")
# Languages that can call each other directly (a Python file can't call a JavaScript function)
_FAMILY = {"typescript": "js", "javascript": "js", "vue": "js", "svelte": "js", "kotlin": "jvm", "java": "jvm",
           "scala": "jvm", "c": "c", "cpp": "c", "objective-c": "c"}
_COMMENT = re.compile(r"^\s*(?:#|//|\*|/\*|<!--)")
MAX_SCAN_FILES = 20_000
MAX_SCAN_BYTES = 1_000_000
MAX_EVIDENCE = 4
CLUSTER_DRILL_MIN = 13         # a component with more modules than this opens into Leiden clusters
FOLDER_CLUSTER_WEIGHT = 0.5    # folder structure's weight next to code edges, for clustering
EDGE_KIND_WEIGHT = {"calls": 1.0, "renders": 1.0, "extends": 0.8, "implements": 0.8, "uses": 0.7,
                    "uses_type": 0.5, "imports": 0.5}
CACHE_SECONDS = 60             # files no parser reads (Java, ...) are re-scanned at most this often
SUMMARY_FILE = "architecture_summaries.json"

with open(os.path.join(os.path.dirname(__file__), "detectors.json"), encoding="utf-8") as _fh:
    DETECTORS = json.load(_fh)


def _compile(entries):
    return [dict(e, regex=re.compile(e["pattern"], re.M)) for e in entries]


_ROUTES = _compile(DETECTORS["http_routes"])
_PREFIXES = _compile(DETECTORS["route_prefixes"])
_CLIENTS = _compile(DETECTORS["http_clients"])
_URLS = dict(DETECTORS["url_literals"], regex=re.compile(DETECTORS["url_literals"]["pattern"]))
_GRPC_SERVERS = _compile(DETECTORS["grpc_servers"])
_GRPC_CLIENTS = _compile(DETECTORS["grpc_clients"])
_PROCESSES = _compile(DETECTORS["processes"])
_INFRA = _compile(DETECTORS["infrastructure"])
# A broad question about the whole system (not "what happens in the orchestration layer when ...",
# which is about one component and goes through the normal search, components first)
_ARCH_WORDS = re.compile(
    r"\b(architecture|architectural|high[- ]level|overview|big picture|system design|"
    r"how (?:does|do) (?:the )?(?:whole )?(?:system|project|codebase|repo(?:sitory)?|it all|everything) (?:work|fit|hang)|"
    r"(?:request|data|control) flows?|end[- ]to[- ]end|"
    r"(?:what|which) (?:are the )?(?:main )?(?:components|services|subsystems|parts|layers)\b(?! (?:use|call|import|depend))|"
    r"how (?:do|are) (?:the )?(?:components|services|subsystems|parts|layers) (?:connect|connected|interact|talk|communicate|fit))", re.I)
_QUERY_WORD = re.compile(r"[a-z]{3,}")
_STOP = {"the", "and", "for", "what", "when", "how", "does", "which", "with", "from", "into", "that", "this", "happens",
         "client", "layer", "system", "code", "request", "requests", "call", "calls"}
SNIPPET_LINES = 12


def _is_test(path: str) -> bool:
    p = f"/{path}"
    name = path.rsplit("/", 1)[-1]
    return ("/tests/" in p or "/test/" in p or "/__tests__/" in p or "/fixtures/" in p or "/testdata/" in p
            or name.startswith("test_") or ".test." in name or ".spec." in name or name.endswith(("_test.py", "_test.go")))


def _family(path: str) -> str:
    lang = LANGS.get(os.path.splitext(path)[1].lower(), "")
    return _FAMILY.get(lang, lang)


def _plausible(g, e) -> bool:
    """A code edge that can cross components: resolved for certain (a guess by method name is
    no evidence across a boundary) and between languages that can call each other."""
    if e.guessed:
        return False
    a, b = _family(g.symbols[e.src].path), _family(g.symbols[e.dst].path)
    return not a or not b or a == b


def _line_of(text: str, pos: int) -> int:
    return text.count("\n", 0, pos) + 1


def _line_text(text: str, line: int) -> str:
    lines = text.splitlines()
    return lines[line - 1].strip()[:160] if 0 < line <= len(lines) else ""


class Architecture:
    """Components, infrastructure and connections (see the module docstring)."""

    def __init__(self):
        self.method = ""                            # manifests | clusters | folders
        self.components: Dict[str, dict] = {}       # id -> component
        self.connections: List[dict] = []
        self.comp_of_file: Dict[str, str] = {}      # every source file on disk (and indexed) -> component
        self.notes: List[str] = []
        self.created = time.time()
        self._clusters: Dict[str, Dict[str, int]] = {}

    # --- lookups ---------------------------------------------------------------------------
    def component_of(self, path: str) -> Optional[str]:
        return self.comp_of_file.get(path)

    def code_components(self) -> List[dict]:
        return [c for c in self.components.values() if c["kind"] == "code"]

    def to_json(self) -> dict:
        return {"method": self.method, "notes": self.notes,
                "components": [{k: v for k, v in c.items() if k not in ("files",)} for c in self.components.values()],
                "connections": self.connections}


# --- building --------------------------------------------------------------------------------
def build(index: Index, mode: str = "auto", summaries: Optional[Dict[str, str]] = None) -> Architecture:
    """mode: auto (build files and compose services; Leiden when there are fewer than two),
    manifests, clusters (Leiden always) or folders (top-level folders)."""
    arch = Architecture()
    files, manifests, compose_files = _walk(index.root)
    indexed = set(index.graph.files)
    all_files = sorted(set(files) | indexed)
    services = _compose_services(index.root, compose_files)

    comp_dirs: Dict[str, List[str]] = defaultdict(list)
    for rel, kind in manifests:
        d = rel.rsplit("/", 1)[0] if "/" in rel else ""
        comp_dirs[d].append(kind)
    for svc in services.values():
        if svc.get("context") is not None:
            comp_dirs.setdefault(svc["context"], [])

    use_manifests = mode in ("auto", "manifests")
    code_dirs = [d for d in comp_dirs if any(_inside(f, d) for f in all_files)]
    if mode == "auto" and len([d for d in code_dirs if d]) < 2:
        use_manifests = False
    if use_manifests:
        arch.method = "manifests"
        _assign_by_dirs(arch, all_files, comp_dirs, services)
    elif mode == "folders":
        arch.method = "folders"
        _assign_by_dirs(arch, all_files, {}, services)
    else:
        arch.method = "clusters"
        _assign_by_clusters(arch, index, all_files)
    if mode == "auto" and arch.method == "clusters":
        arch.notes.append("Fewer than two folders have their own build file or compose service, so components "
                          "were found by clustering the module graph (Leiden).")

    for c in arch.code_components():
        _describe(index, c)
        if summaries and c["id"] in summaries:
            c["summary"] = summaries[c["id"]]
    _code_connections(index, arch)
    _runtime_connections(index, arch, all_files, services)
    _merge_connections(arch)
    return arch


def _inside(path: str, d: str) -> bool:
    return d == "" or path.startswith(d + "/")


def _walk(root: str) -> Tuple[List[str], List[Tuple[str, str]], List[str]]:
    """Source files, build files (path, kind) and compose files, skipping build output,
    dependencies and test fixtures."""
    files, manifests, compose = [], [], []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS and d not in EXTRA_SKIP
                             and not d.endswith(".egg-info") and not d.startswith("."))
        rel_dir = os.path.relpath(dirpath, root).replace(os.sep, "/")
        rel_dir = "" if rel_dir == "." else rel_dir
        for name in filenames:
            rel = f"{rel_dir}/{name}" if rel_dir else name
            ext = os.path.splitext(name)[1].lower()
            if name in MANIFESTS or ext in MANIFEST_SUFFIXES:
                if not _is_test(rel):
                    manifests.append((rel, MANIFESTS.get(name) or MANIFEST_SUFFIXES[ext]))
            elif COMPOSE.match(name) and not rel_dir:
                compose.append(rel)
            if ext in LANGS and not GENERATED.search(name):
                files.append(rel)
                if len(files) >= MAX_SCAN_FILES:
                    return files, manifests, compose
    return files, manifests, compose


def _compose_services(root: str, compose_files: List[str]) -> Dict[str, dict]:
    """docker-compose services: name -> {context (folder, or None for an image), image,
    depends_on, environment, aliases, file}."""
    services: Dict[str, dict] = {}
    for rel in sorted(compose_files, key=lambda r: (r != "docker-compose.yml", r)):
        try:
            with open(os.path.join(root, rel), encoding="utf-8", errors="replace") as fh:
                data = miniyaml.load(fh.read())
        except OSError:
            continue
        for name, svc in ((data or {}).get("services") or {}).items() if isinstance(data, dict) else []:
            if not isinstance(svc, dict) or name in services:
                continue
            build = svc.get("build")
            context = build.get("context") if isinstance(build, dict) else build
            if isinstance(context, str):
                context = os.path.normpath(context).replace(os.sep, "/").lstrip("./") if context not in (".", "./") else ""
                context = "" if context == "." else context
            else:
                context = None
            env = svc.get("environment") or {}
            if isinstance(env, list):
                env = dict(str(e).partition("=")[::2] for e in env)
            deps = svc.get("depends_on") or []
            aliases = []
            nets = svc.get("networks")
            if isinstance(nets, dict):
                for n in nets.values():
                    if isinstance(n, dict):
                        aliases += [str(a) for a in n.get("aliases") or []]
            if svc.get("container_name"):
                aliases.append(str(svc["container_name"]))
            services[name] = {"name": name, "context": context, "image": str(svc.get("image") or ""),
                              "depends_on": list(deps) if isinstance(deps, (list, dict)) else [],
                              "environment": {str(k): str(v) for k, v in env.items()},
                              "aliases": aliases, "file": rel}
    return services


def _new_component(cid: str, name: str, kind: str, source: str, path: str = "") -> dict:
    return {"id": cid, "name": name, "kind": kind, "source": source, "path": path, "files": [],
            "fileCount": 0, "symbols": 0, "languages": [], "description": "", "summary": "", "services": []}


def _assign_by_dirs(arch: Architecture, files: List[str], comp_dirs: Dict[str, List[str]], services: Dict[str, dict]):
    dirs = sorted(comp_dirs, key=lambda d: -len(d))   # deepest first
    for d in dirs:
        kinds = sorted(set(comp_dirs[d]))
        name = d or "(repository root)"
        source = ("build files: " + ", ".join(kinds)) if kinds else "docker-compose build context"
        arch.components[d or "."] = _new_component(d or ".", name, "code", source, d)
    for f in files:
        d = next((d for d in dirs if _inside(f, d)), None)
        if d is None:  # outside every component: its top-level folder
            d = f.split("/", 1)[0] if "/" in f else ""
            if (d or ".") not in arch.components:
                arch.components[d or "."] = _new_component(d or ".", d or "(repository root)", "code", "folder", d)
        arch.components[d or "."]["files"].append(f)
        arch.comp_of_file[f] = d or "."
    for cid in [c for c, comp in arch.components.items() if not comp["files"]]:
        del arch.components[cid]
    for svc in services.values():
        if svc.get("context") is not None and (svc["context"] or ".") in arch.components:
            arch.components[svc["context"] or "."]["services"].append(svc["name"])


def _module_graph(index: Index, files: Optional[set] = None) -> Dict[str, Dict[str, float]]:
    """Files as nodes, code edges between them summed up (undirected, weighted by kind), for
    clustering. Folders join as light extra nodes, so structure helps where code edges are few."""
    g = index.graph
    adj: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    counts: Dict[Tuple[str, str], float] = defaultdict(float)
    for e in g.edges:
        w = EDGE_KIND_WEIGHT.get(e.kind)
        if w is None:
            continue
        a, b = g.symbols[e.src].path, g.symbols[e.dst].path
        if a == b or (files is not None and (a not in files or b not in files)) or _family(a) != _family(b):
            continue
        counts[tuple(sorted((a, b)))] += w * (0.3 if e.guessed else 1.0) * max(e.count, 1)
    for (a, b), w in counts.items():
        w = 1.0 + math.log(w)  if w > 1 else w
        adj[a][b] += w
        adj[b][a] += w
    for path in (files if files is not None else g.files):
        adj[path]  # every file is a node
        parent = path.rsplit("/", 1)[0] if "/" in path else ""
        if parent:
            node = f"{parent}/"
            adj[path][node] += FOLDER_CLUSTER_WEIGHT
            adj[node][path] += FOLDER_CLUSTER_WEIGHT
    return adj


def cluster_files(index: Index, files: Optional[set] = None, resolution: float = 1.0) -> Dict[str, int]:
    """file -> cluster number (Leiden over the module graph; folder nodes dropped)."""
    adj = _module_graph(index, files)
    communities = leiden(adj, resolution=resolution, seed=0)
    return {n: c for n, c in communities.items() if not n.endswith("/")}


def _cluster_name(paths: List[str], index: Index, taken: set) -> str:
    """The deepest folder holding most (60%) of its files, then (if another cluster has that
    name) its most connected file."""
    prefixes = Counter()
    for p in paths:
        parts = p.split("/")[:-1]
        for i in range(1, len(parts) + 1):
            prefixes["/".join(parts[:i])] += 1
    major = [d for d, n in prefixes.items() if n >= 0.6 * len(paths)]
    if major:
        base = max(major, key=lambda d: (d.count("/"), prefixes[d]))
        if prefixes[base] < len(paths):
            base += f" (+{len(paths) - prefixes[base]})"
    else:  # spread over folders: the two holding most of it
        parents = Counter(p.rsplit("/", 1)[0] if "/" in p else "(root)" for p in paths)
        base = " + ".join(d for d, _ in parents.most_common(2))
    if base not in taken:
        return base
    g = index.graph
    degree = Counter()
    for e in g.edges:
        if e.kind in EDGE_KIND_WEIGHT:
            degree[g.symbols[e.dst].path] += 1
    best = max(paths, key=lambda p: (degree.get(p, 0), p))
    name = f"{base} · {best.rsplit('/', 1)[-1].rsplit('.', 1)[0]}"
    while name in taken:
        name += "'"
    return name


def _assign_by_clusters(arch: Architecture, index: Index, files: List[str]):
    indexed = set(index.graph.files)
    clusters = cluster_files(index, indexed)
    groups: Dict[int, List[str]] = defaultdict(list)
    for f, c in clusters.items():
        groups[c].append(f)
    taken = set()
    by_cluster = {}
    for c, paths in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        name = _cluster_name(paths, index, taken)
        taken.add(name)
        cid = f"cluster:{name}"
        comp = _new_component(cid, name, "code", "Leiden cluster of the module graph")
        comp["path"] = name if re.fullmatch(r"[^()·]+", name) else ""
        arch.components[cid] = comp
        by_cluster[c] = cid
    for f in files:
        if f in clusters:
            cid = by_cluster[clusters[f]]
        else:  # not indexed (no parser or language server for it): the cluster of its folder
            folder = f.rsplit("/", 1)[0] if "/" in f else ""
            near = [g for g in indexed if g.startswith(folder + "/")] if folder else []
            cid = by_cluster[clusters[near[0]]] if near else None
            if cid is None:
                top = f.split("/", 1)[0] if "/" in f else "(root)"
                cid = f"folder:{top}"
                arch.components.setdefault(cid, _new_component(cid, top, "code", "folder", top if "/" in f else ""))
        arch.components[cid]["files"].append(f)
        arch.comp_of_file[f] = cid


def _read_text(path: str) -> str:
    try:
        if os.path.getsize(path) > MAX_SCAN_BYTES:
            return ""
        with open(path, "rb") as fh:
            raw = fh.read()
    except OSError:
        return ""
    if not path.endswith(".py") and looks_minified(raw):
        return ""
    return raw.decode("utf-8", "replace")


def _describe(index: Index, comp: dict):
    g = index.graph
    langs = Counter(LANGS.get(os.path.splitext(f)[1].lower(), "") for f in comp["files"])
    comp["languages"] = [l for l, _ in langs.most_common() if l and l != "protobuf"][:4]
    comp["fileCount"] = len(comp["files"])
    comp["symbols"] = sum(1 for f in comp["files"] if f in g.files for s in g.files[f].symbols if s.kind != "module")
    comp["indexedFiles"] = sum(1 for f in comp["files"] if f in g.files)
    base = index.abs(comp["path"]) if comp["path"] else None
    desc = ""
    if base and os.path.isdir(base):
        pkg = os.path.join(base, "package.json")
        if os.path.exists(pkg):
            try:
                with open(pkg, encoding="utf-8") as fh:
                    desc = json.load(fh).get("description") or ""
            except (OSError, ValueError):
                pass
        if not desc:
            for name in ("pyproject.toml", "setup.cfg", "pom.xml", "Cargo.toml"):
                text = _read_text(os.path.join(base, name))
                m = re.search(r"^\s*description\s*=\s*[\"']([^\"']+)", text, re.M) or \
                    re.search(r"<description>([^<]+)</description>", text)
                if m:
                    desc = m.group(1)
                    break
        if not desc:
            for name in ("README.md", "readme.md", "README.rst", "README.txt"):
                text = _read_text(os.path.join(base, name))
                para = [l.strip() for l in text.split("\n\n")
                        if l.strip() and not l.strip().startswith(("#", "!", "[", "<", "```", "|", "-", "*"))]
                if para:
                    desc = re.sub(r"\s+", " ", para[0])
                    break
    comp["description"] = desc[:300]


# --- connections -----------------------------------------------------------------------------
def _add(arch: Architecture, src: str, dst: str, kind: str, label: str, evidence: List[dict], **extra):
    if src == dst or src not in arch.components or dst not in arch.components:
        return
    arch.connections.append({"from": src, "to": dst, "kind": kind, "label": label,
                             "evidence": evidence[:MAX_EVIDENCE], **extra})


def _code_connections(index: Index, arch: Architecture):
    g = index.graph
    pairs: Dict[Tuple[str, str], list] = defaultdict(list)
    for e in g.edges:
        if e.kind not in EDGE_KIND_WEIGHT or not _plausible(g, e):
            continue
        a = arch.component_of(g.symbols[e.src].path)
        b = arch.component_of(g.symbols[e.dst].path)
        if a and b and a != b:
            pairs[(a, b)].append(e)
    for (a, b), edges in pairs.items():
        kinds = Counter(e.kind for e in edges)
        edges.sort(key=lambda e: (e.guessed, -EDGE_KIND_WEIGHT[e.kind], e.src))
        evidence = [{"path": g.symbols[e.src].path, "line": g.symbols[e.src].start,
                     "text": f"{g.symbols[e.src].qualname or g.symbols[e.src].path} --{e.kind}--> "
                             f"{g.symbols[e.dst].qualname or g.symbols[e.dst].path}",
                     "source": e.src, "target": e.dst} for e in edges[:MAX_EVIDENCE]]
        _add(arch, a, b, "code", f"{len(edges)} code references", evidence,
             count=len(edges), kinds=dict(kinds))


def _normalise_path(url: str) -> Optional[str]:
    """A URL as written in code -> its path with parameters as *: the base URL variable or
    scheme and host dropped, the query string too (API_BASE + /api/v1/items/{id}?x=1 gives
    /api/v1/items/*)."""
    url = url.strip("`'\"")
    url = re.sub(r"^https?://[^/]+", "", url)
    url = re.sub(r"^(?:\$\{[^}]*\}|\{[^}]*\})+", "", url)
    url = url.split("?", 1)[0].split("#", 1)[0]
    if not url.startswith("/"):
        return None
    segments = []
    for s in url.strip("/").split("/"):
        if not s:
            continue
        if re.fullmatch(r"\$\{[^}]*\}|\{[^}]*\}|:\w+|<[^>]*>|\*", s) or "${" in s or "{" in s:
            segments.append("*")
        else:
            segments.append(s)
    return "/" + "/".join(segments)


def _route_matches(route: str, client: str) -> bool:
    r, c = route.strip("/").split("/"), client.strip("/").split("/")
    if r == [""] or len(c) < len(r):
        return False
    c = c[len(c) - len(r):]  # a client may add a prefix (a proxy, a base path)
    return all(x == y or x == "*" or y == "*" for x, y in zip(r, c))


def _scan(index: Index, arch: Architecture, files: List[str]) -> dict:
    """Runs every detector over the source files (cached per file by mtime and size)."""
    found = {"routes": [], "clients": [], "grpc_servers": [], "grpc_clients": [], "processes": [],
             "infra": [], "protos": {}}
    cache = getattr(index, "_arch_scan", None)
    if cache is None:
        cache = index._arch_scan = {}
    for rel in files:
        full = index.abs(rel)
        try:
            st = os.stat(full)
        except OSError:
            continue
        key = (st.st_mtime, st.st_size)
        hit = cache.get(rel)
        if hit and hit[0] == key:
            result = hit[1]
        else:
            result = _scan_file(rel, _read_text(full))
            cache[rel] = (key, result)
        for k, v in result.items():
            if k == "protos":
                found["protos"].update(v)
            else:
                found[k] += v
    return found


def _scan_file(rel: str, text: str) -> dict:
    out = {"routes": [], "clients": [], "grpc_servers": [], "grpc_clients": [], "processes": [],
           "infra": [], "protos": {}}
    if not text:
        return out
    ext = os.path.splitext(rel)[1].lower()
    lang = LANGS.get(ext, "")
    test = _is_test(rel)

    def ev(pos: int, what: str = "") -> dict:
        line = _line_of(text, pos)
        return {"path": rel, "line": line, "text": what or _line_text(text, line)}

    def code_at(m) -> bool:
        """Not in a comment line (examples in comments and docs aren't connections)."""
        start = text.rfind("\n", 0, m.start()) + 1
        return not _COMMENT.match(text[start:m.start() + 1])

    if lang == "protobuf":
        for m in re.finditer(r"\bservice\s+(\w+)\s*\{(.*?)^\}", text, re.S | re.M):
            out["protos"][m.group(1)] = re.findall(r"\brpc\s+(\w+)\s*\(", m.group(2))
        return out
    if not test:
        # Server routes, with a class- or router-level prefix written before them
        prefix = ""
        for p in _PREFIXES:
            if lang in p["langs"]:
                m = p["regex"].search(text)
                if m:
                    prefix = m.group(1)
        first_class = re.search(r"\bclass\s+\w+", text)
        for r in _ROUTES:
            if lang not in r["langs"]:
                continue
            for m in r["regex"].finditer(text):
                if not code_at(m):
                    continue
                verb, path = m.group(1).upper(), ""
                if r["framework"] == "spring":
                    args = m.group(2) or ""
                    if verb == "REQUEST" and first_class and m.start() < first_class.start():
                        continue  # the class-level prefix itself
                    named = re.search(r"\b(?:value|path)\s*=\s*\{?\s*\"([^\"]*)\"", args)
                    positional = re.match(r"\(\s*\{?\s*\"([^\"]*)\"", args)
                    path = (named or positional).group(1) if (named or positional) else ""
                    if verb == "REQUEST":
                        method = re.search(r"RequestMethod\.(\w+)", args)
                        verb = method.group(1) if method else "ANY"
                elif r["framework"] == "flask/fastapi":
                    path = m.group(2)
                    if verb in ("ROUTE", "API_ROUTE"):
                        methods = re.findall(r"[\"'](GET|POST|PUT|DELETE|PATCH)[\"']", m.group(3) or "", re.I)
                        verb = methods[0].upper() if methods else "GET"
                elif r["framework"] == "go" and verb in ("HANDLEFUNC", "HANDLE"):
                    path, verb = m.group(2), "ANY"
                else:
                    path = m.group(2) or ""
                full = _normalise_path("/" + "/".join(x.strip("/") for x in (prefix, path) if x.strip("/")))
                if full:
                    out["routes"].append({"method": verb, "path": full, "evidence": ev(m.start())})
        # HTTP clients
        seen = set()
        for c in _CLIENTS:
            if lang not in c["langs"]:
                continue
            for m in c["regex"].finditer(text):
                if not code_at(m):
                    continue
                verb = m.group(1).upper() if m.lastindex and m.lastindex >= 2 else ""
                url = m.group(m.lastindex)
                if verb in ("URI", "EXCHANGE", ""):
                    verb = ""
                if verb.startswith("GETFOR"):
                    verb = "GET"
                elif verb.startswith("POSTFOR"):
                    verb = "POST"
                if not verb:  # fetch(url, {method: 'POST'})
                    call = text[m.end():m.end() + 300]
                    method = re.search(r"method\s*:\s*['\"`](\w+)", call.split(");", 1)[0])
                    verb = method.group(1).upper() if method else ("GET" if "fetch" in m.group(0) else "")
                path = _normalise_path(url)
                if path:
                    seen.add(m.start(m.lastindex))
                    out["clients"].append({"method": verb, "path": path, "evidence": ev(m.start())})
        if lang in _URLS["langs"]:
            for m in _URLS["regex"].finditer(text):
                if m.start() in seen or any(abs(m.start() - s) < 3 for s in seen) or not code_at(m):
                    continue
                path = _normalise_path(m.group(2))
                if path and path != "/":
                    out["clients"].append({"method": "", "path": path, "evidence": ev(m.start())})
        for kind, detectors in (("grpc_servers", _GRPC_SERVERS), ("grpc_clients", _GRPC_CLIENTS)):
            for d in detectors:
                if lang in d["langs"]:
                    for m in d["regex"].finditer(text):
                        if not code_at(m) or (kind == "grpc_clients" and re.match(
                                r"\s*(?:def|class)\b", text[text.rfind("\n", 0, m.start()) + 1:m.start()])):
                            continue
                        out[kind].append({"service": m.group(1), "evidence": ev(m.start())})
        for d in _PROCESSES:
            if lang in d["langs"]:
                for m in d["regex"].finditer(text):
                    if code_at(m):
                        out["processes"].append({"module": m.group(1), "evidence": ev(m.start())})
        for d in _INFRA:
            m = next((m for m in d["regex"].finditer(text) if code_at(m)), None)
            if m:
                out["infra"].append({"id": d["id"], "evidence": ev(m.start())})
    return out


def _infra_node(arch: Architecture, infra_id: str, name: str = "") -> str:
    d = next((d for d in _INFRA if d["id"] == infra_id), None)
    cid = f"infra:{infra_id}"
    if cid not in arch.components:
        comp = _new_component(cid, name or (d["name"] if d else infra_id), d["kind"] if d else "service",
                              "detected in client code" if not name else "docker-compose service")
        if d and name and name.lower() != d["name"].lower():
            comp["name"] = f"{d['name']} ({name})"
        arch.components[cid] = comp
    return cid


def _image_infra(image: str) -> Optional[str]:
    base = image.lower().split(":")[0]
    for d in _INFRA:
        if any(i in base for i in d.get("images", [])):
            return d["id"]
    return None


def _runtime_connections(index: Index, arch: Architecture, files: List[str], services: Dict[str, dict]):
    found = _scan(index, arch, files)
    comp = arch.component_of

    # docker-compose: services as components or infrastructure
    svc_comp: Dict[str, str] = {}
    for name, svc in services.items():
        if svc["context"] is not None and (svc["context"] or ".") in arch.components:
            svc_comp[name] = svc["context"] or "."
            continue
        if svc["context"] is not None:  # clusters or folders: the component with most of its files
            holders = Counter(c for f, c in arch.comp_of_file.items() if _inside(f, svc["context"]))
            if holders:
                svc_comp[name] = holders.most_common(1)[0][0]
                arch.components[svc_comp[name]]["services"].append(name)
                continue
        infra = _image_infra(svc["image"])
        if infra:
            svc_comp[name] = _infra_node(arch, infra, name)
            arch.components[svc_comp[name]]["services"].append(name)
        elif svc["image"]:
            cid = f"service:{name}"
            arch.components.setdefault(cid, _new_component(cid, name, "service", f"docker-compose image {svc['image']}"))
            arch.components[cid]["services"].append(name)
            svc_comp[name] = cid
    names = {}
    for name, svc in services.items():
        if name in svc_comp:
            for alias in [name] + svc["aliases"]:
                names[alias.lower()] = svc_comp[name]
    for name, svc in services.items():
        src = svc_comp.get(name)
        if not src:
            continue
        deps = svc["depends_on"] if isinstance(svc["depends_on"], list) else list(svc["depends_on"])
        for dep in deps:
            if str(dep) in svc_comp:
                _add(arch, src, svc_comp[str(dep)], "compose", "depends_on (compose)",
                     [{"path": svc["file"], "line": 0, "text": f"{name} depends_on {dep}"}])
        for key, value in svc["environment"].items():
            for host in re.findall(r"(?:^|//|@)([\w.\-]+)(?=[:/]|$)", value):
                target = names.get(host.lower())
                if target and target != src:
                    _add(arch, src, target, "compose", f"{key} (compose)",
                         [{"path": svc["file"], "line": 0, "text": f"{name}: {key}={value}"}])

    # HTTP: clients matched to routes
    for client in found["clients"]:
        src = comp(client["evidence"]["path"])
        best = None
        for route in found["routes"]:
            if client["method"] and route["method"] not in ("ANY", client["method"]):
                continue
            if _route_matches(route["path"], client["path"]):
                if best is None or len(route["path"]) > len(best["path"]):
                    best = route
        if best and src:
            dst = comp(best["evidence"]["path"])
            if dst and dst != src:
                verb = client["method"] or best["method"]
                _add(arch, src, dst, "http", f"HTTP {verb} {best['path']}",
                     [client["evidence"], best["evidence"]], endpoint=f"{verb} {best['path']}",
                     client=client["evidence"], server=best["evidence"])

    # gRPC: stubs matched to servicers by service name
    servers = defaultdict(list)
    for s in found["grpc_servers"]:
        servers[s["service"]].append(s)
    for c in found["grpc_clients"]:
        src = comp(c["evidence"]["path"])
        for s in servers.get(c["service"], []):
            dst = comp(s["evidence"]["path"])
            if src and dst and src != dst:
                rpcs = found["protos"].get(c["service"], [])
                label = f"gRPC {c['service']}" + (f" ({', '.join(rpcs[:4])}{', ...' if len(rpcs) > 4 else ''})" if rpcs else "")
                _add(arch, src, dst, "grpc", label, [c["evidence"], s["evidence"]],
                     endpoint=label, client=c["evidence"], server=s["evidence"])

    # Processes: "-m module" started by another component
    for p in found["processes"]:
        src = comp(p["evidence"]["path"])
        top = p["module"].split(".")[0]
        for c in arch.code_components():
            if c["id"] != src and any(f.split("/")[:-1] and top in f.split("/")[:-1] for f in c["files"]):
                _add(arch, src, c["id"], "process", f"starts `python -m {p['module']}`", [p["evidence"]])
                break

    # Infrastructure from client code
    for i in found["infra"]:
        src = comp(i["evidence"]["path"])
        if src and arch.components.get(src, {}).get("kind") == "code":
            _add(arch, src, _infra_node(arch, i["id"]), "datastore" if i["id"] != "llm" else "model API",
                 "uses", [i["evidence"]])


def _merge_connections(arch: Architecture):
    """One connection per (from, to, kind), its labels and evidence combined."""
    merged: Dict[Tuple[str, str, str], dict] = {}
    for c in arch.connections:
        key = (c["from"], c["to"], c["kind"])
        if key not in merged:
            merged[key] = dict(c, labels=[c["label"]], evidence=list(c["evidence"]),
                               endpoints=[c["endpoint"]] if c.get("endpoint") else [],
                               pairs=[{"client": c["client"], "server": c["server"]}] if c.get("client") else [])
            continue
        m = merged[key]
        if c["label"] not in m["labels"]:
            m["labels"].append(c["label"])
        if c.get("endpoint") and c["endpoint"] not in m["endpoints"]:
            m["endpoints"].append(c["endpoint"])
        if c.get("client"):
            m["pairs"].append({"client": c["client"], "server": c["server"]})
        for e in c["evidence"]:
            if len(m["evidence"]) < MAX_EVIDENCE * 2 and e not in m["evidence"]:
                m["evidence"].append(e)
    out = []
    for m in merged.values():
        for k in ("client", "server", "endpoint"):
            m.pop(k, None)
        labels = m.pop("labels")
        if m["kind"] in ("http", "grpc") and m["endpoints"]:
            m["label"] = ", ".join(m["endpoints"][:3]) + (f" (+{len(m['endpoints']) - 3} more)" if len(m["endpoints"]) > 3 else "")
        elif m["kind"] == "compose":
            m["label"] = ", ".join(labels[:2]) + (f" (+{len(labels) - 2})" if len(labels) > 2 else "")
        else:
            m["label"] = labels[0]
        out.append(m)
    kind_order = {"http": 0, "grpc": 0, "process": 1, "code": 2, "model API": 3, "datastore": 3, "compose": 4}
    out.sort(key=lambda c: (kind_order.get(c["kind"], 5), c["from"], c["to"]))
    arch.connections = out


# --- cache -----------------------------------------------------------------------------------
def get(index: Index, mode: str = "auto", storage: str = "") -> Architecture:
    """The architecture for the current code graph (rebuilt when the graph is, or after
    CACHE_SECONDS for files only the detectors read)."""
    cached = getattr(index, "_architecture", None)
    if cached and cached[0] is index.graph and cached[1] == mode and time.time() - cached[2].created < CACHE_SECONDS:
        return cached[2]
    arch = build(index, mode, load_summaries(storage, index) if storage else None)
    index._architecture = (index.graph, mode, arch)
    return arch


# --- views -----------------------------------------------------------------------------------
def view(index: Index, arch: Architecture, node: str = "") -> dict:
    """The diagram at one level: the system (node ""), a component (its Leiden clusters or
    modules), or a cluster ("<component>#<n>": its modules). Neighbouring components show as
    the ends of connections that leave it."""
    if not node:
        nodes = [_comp_node(c) for c in arch.components.values()]
        edges = [{"source": c["from"], "target": c["to"], "kind": c["kind"], "label": c["label"],
                  "count": c.get("count", len(c["evidence"])), "evidence": c["evidence"]} for c in arch.connections]
        return {"level": "system", "node": "", "title": "System architecture", "method": arch.method,
                "notes": arch.notes, "nodes": nodes, "edges": edges, "mermaid": mermaid(arch)}
    cid, _, part = node.partition("#")
    comp = arch.components.get(cid)
    if comp is None:
        raise ValueError(f"no component {cid!r}")
    files = comp["files"]
    clusters = _clusters_of(index, arch, cid)
    if part:
        files = [f for f in files if clusters.get(f) == int(part)]
        title = f"{comp['name']} › cluster {int(part) + 1}"
        groups = None
    else:
        title = comp["name"]
        groups = clusters if len([f for f in files if f in index.graph.files]) > CLUSTER_DRILL_MIN else None
    return _inner_view(index, arch, comp, files, groups, title, node)


def _clusters_of(index: Index, arch: Architecture, cid: str) -> Dict[str, int]:
    if cid not in arch._clusters:
        files = {f for f in arch.components[cid]["files"] if f in index.graph.files}
        arch._clusters[cid] = cluster_files(index, files) if files else {}
    return arch._clusters[cid]


def _comp_node(c: dict) -> dict:
    return {"id": c["id"], "label": c["name"], "kind": c["kind"], "source": c["source"],
            "files": c.get("fileCount", 0), "symbols": c.get("symbols", 0), "languages": c.get("languages", []),
            "description": c.get("summary") or c.get("description", ""), "path": (c["path"] + "/") if c.get("path") else "",
            "drill": c["kind"] == "code"}


def _inner_view(index: Index, arch: Architecture, comp: dict, files: List[str], groups: Optional[Dict[str, int]],
                title: str, node: str) -> dict:
    g = index.graph
    inside = set(files)

    def group_of(path: str) -> str:
        if path in inside:
            return f"{comp['id']}#{groups[path]}" if groups and path in groups else f"file:{path}"
        other = arch.component_of(path)
        return other or ""

    nodes: Dict[str, dict] = {}
    for f in files:
        gid = group_of(f)
        if gid.startswith("file:"):
            if f in g.files:
                nodes[gid] = {"id": gid, "label": f.rsplit("/", 1)[-1], "kind": "module", "path": f, "line": 1,
                              "files": 1, "drill": False}
            continue
        n = nodes.setdefault(gid, {"id": gid, "label": "", "kind": "cluster", "files": 0, "members": [], "drill": True})
        n["files"] += 1
        n["members"].append(f)
    degree = Counter()
    for e in g.edges:
        if e.kind in EDGE_KIND_WEIGHT and g.symbols[e.src].path != g.symbols[e.dst].path:
            degree[g.symbols[e.src].path] += 1
            degree[g.symbols[e.dst].path] += 1
    for n in nodes.values():
        if n["kind"] == "cluster":
            # Named after its most connected files: "main.py, retriever.py +10"
            members = sorted(n["members"], key=lambda m: (-degree.get(m, 0), m))
            stems = [m.rsplit("/", 1)[-1] for m in members]
            n["label"] = ", ".join(stems[:2]) + (f" +{len(stems) - 2}" if len(stems) > 2 else "")
            n["description"] = ", ".join(stems[:8]) + (f", +{len(stems) - 8} more" if len(stems) > 8 else "")
            n.pop("members")
    edges: Dict[Tuple[str, str, str], dict] = {}
    for e in g.edges:
        if e.kind not in EDGE_KIND_WEIGHT:
            continue
        a, b = g.symbols[e.src].path, g.symbols[e.dst].path
        if a not in inside and b not in inside:
            continue
        ga, gb = group_of(a), group_of(b)
        if not ga or not gb or ga == gb:
            continue
        if (a not in inside or b not in inside) and not _plausible(g, e):
            continue  # a guess or a cross-language match leaving the component isn't evidence
        for x in (ga, gb):
            if x not in nodes and x in arch.components:
                nodes[x] = dict(_comp_node(arch.components[x]), outside=True)
        key = (ga, gb, "code")
        edge = edges.setdefault(key, {"source": ga, "target": gb, "kind": "code", "count": 0, "kinds": Counter(), "evidence": []})
        edge["count"] += 1
        edge["kinds"][e.kind] += 1
        if len(edge["evidence"]) < MAX_EVIDENCE:
            edge["evidence"].append({"path": a, "line": g.symbols[e.src].start,
                                     "text": f"{g.symbols[e.src].qualname or a} --{e.kind}--> {g.symbols[e.dst].qualname or b}"})
    # Runtime connections that start or end inside: drawn from the file with the evidence
    for c in arch.connections:
        if c["kind"] == "code" or comp["id"] not in (c["from"], c["to"]):
            continue
        other = c["to"] if c["from"] == comp["id"] else c["from"]
        for ev in c["evidence"]:
            if ev["path"] in inside:
                gid = group_of(ev["path"])
                if gid not in nodes and gid.startswith("file:"):
                    nodes[gid] = {"id": gid, "label": ev["path"].rsplit("/", 1)[-1], "kind": "module",
                                  "path": ev["path"], "line": 1, "files": 1, "drill": False}
                if gid not in nodes:
                    continue
                if other not in nodes:
                    nodes[other] = dict(_comp_node(arch.components[other]), outside=True)
                src, dst = (gid, other) if c["from"] == comp["id"] else (other, gid)
                key = (src, dst, c["kind"])
                edges.setdefault(key, {"source": src, "target": dst, "kind": c["kind"], "label": c["label"],
                                       "count": 1, "evidence": [ev]})
    out_edges = []
    for e in edges.values():
        if isinstance(e.get("kinds"), Counter):
            e["label"] = f"{e['count']} ref" + ("s" if e["count"] != 1 else "")
            e["kinds"] = dict(e["kinds"])
        out_edges.append(e)
    return {"level": "cluster" if "#" in node else "component", "node": node, "title": title,
            "component": comp["id"], "method": arch.method, "notes": [],
            "nodes": list(nodes.values()), "edges": out_edges,
            "grouped": bool(groups) and "#" not in node}


def mermaid(arch: Architecture) -> str:
    """The system diagram as Mermaid (a flowchart), for documentation."""
    ids = {cid: f"c{i}" for i, cid in enumerate(arch.components)}
    shapes = {"code": ("[", "]"), "datastore": ("[(", ")]"), "queue": ("[[", "]]"), "service": ("([", "])")}
    lines = ["flowchart LR"]
    for cid, c in arch.components.items():
        a, b = shapes.get(c["kind"], ("[", "]"))
        label = c["name"].replace('"', "'")
        lines.append(f'  {ids[cid]}{a}"{label}"{b}')
    for c in arch.connections:
        label = c["label"].replace('"', "'").replace("|", "/")[:60]
        arrow = "-.->" if c["kind"] in ("compose", "code") else "-->"
        lines.append(f'  {ids[c["from"]]} {arrow}|"{label}"| {ids[c["to"]]}')
    return "\n".join(lines)


# --- text for the model ----------------------------------------------------------------------
def describe_component(c: dict) -> str:
    bits = [c["source"]]
    if c.get("languages"):
        bits.append("/".join(c["languages"]))
    if c.get("fileCount"):
        bits.append(f"{c['fileCount']} files")
    text = f"{c['name']} ({', '.join(bits)})"
    about = c.get("summary") or c.get("description")
    return f"{text}: {about}" if about else text


def connection_text(arch: Architecture, c: dict) -> str:
    a, b = arch.components[c["from"]]["name"], arch.components[c["to"]]["name"]
    where = ""
    if c["evidence"] and c["evidence"][0].get("line"):
        where = f" (e.g. `{c['evidence'][0]['path']}:{c['evidence'][0]['line']}`)"
    kinds = f" [{', '.join(f'{n} {k}' for k, n in Counter(c['kinds']).most_common())}]" if c.get("kinds") else ""
    return f"{a} --{c['kind']}: {c['label']}{kinds}--> {b}{where}"


def text_for(arch: Architecture, components: List[str] = (), limit: int = 30) -> str:
    """The architecture as the model reads it: every component, then the connections (those
    touching `components` first)."""
    focus = set(components)
    lines = [f"Components (found from {arch.method}):"]
    for c in arch.components.values():
        lines.append(f"- {describe_component(c)}")
    conns = sorted(arch.connections, key=lambda c: (not (focus & {c["from"], c["to"]}), c["kind"] == "compose"))
    lines.append("Connections (static analysis; http/grpc/process/datastore are detected heuristically):")
    lines += [f"- {connection_text(arch, c)}" for c in conns[:limit]]
    if len(conns) > limit:
        lines.append(f"- (+{len(conns) - limit} more)")
    return "\n".join(lines)


def is_architecture_question(query: str) -> bool:
    return bool(_ARCH_WORDS.search(query))


def answer(index: Index, arch: Architecture, query: str) -> dict:
    """A broad question about the system: the architecture as a graph answer (the same shape
    as structural.answer)."""
    code = arch.code_components()
    runtime = [c for c in arch.connections if c["kind"] != "code"]
    summary = (f"{len(code)} code component(s) ({arch.method}), "
               f"{len(arch.components) - len(code)} infrastructure/service node(s), "
               f"{len(arch.connections)} connection(s): "
               + "; ".join(f"{arch.components[c['from']]['name']} -> {arch.components[c['to']]['name']} ({c['kind']})"
                           for c in arch.connections if c["kind"] != "compose")[:900])
    facts = []
    for c in arch.connections:
        ev = c["evidence"][0] if c["evidence"] else {}
        facts.append({"source": c["from"], "target": c["to"], "kind": c["kind"], "guessed": c["kind"] not in ("code",),
                      "text": connection_text(arch, c), "path": ev.get("path", ""), "line": ev.get("line", 0), "why": ""})
    symbols = []
    g = index.graph
    for c in runtime:
        for ev in c["evidence"]:
            sid = g.symbol_at(ev["path"], ev["line"]) if ev.get("line") else None
            if sid and g.symbols[sid].kind not in ("module", "folder") and sid not in symbols:
                symbols.append(sid)
    return {"kind": "architecture", "targets": [], "summary": summary, "facts": facts, "symbols": symbols,
            "exact": False, "text": text_for(arch)}


# --- top-down retrieval ----------------------------------------------------------------------
def rank_components(index: Index, arch: Architecture, query: str, seeds: Dict[str, float]) -> Dict[str, float]:
    """How relevant each component is to a request: the components the question names, those
    holding its seed symbols, and similarity to each component's profile; then PPR over the
    component graph, so a component connected to those also scores."""
    g = index.graph
    start: Dict[str, float] = defaultdict(float)
    low = query.lower()
    for c in arch.components.values():
        names = {c["name"].lower(), c["name"].rsplit("/", 1)[-1].lower()} | {s.lower() for s in c["services"]}
        names |= {n.replace("_", " ") for n in names} | {n.replace("-", " ") for n in names}
        if any(len(n) >= 3 and re.search(r"(?<![\w-])" + re.escape(n) + r"(?![\w-])", low) for n in names):
            start[c["id"]] += 1.0
    total = sum(seeds.values()) or 1.0
    for sid, w in seeds.items():
        s = g.symbols.get(sid)
        cid = arch.component_of(s.path.rstrip("/")) if s and s.kind != "folder" else None
        if s is not None and s.kind == "folder":
            cid = next((c["id"] for c in arch.code_components() if c["path"] and
                        (s.path.rstrip("/") == c["path"] or s.path.startswith(c["path"] + "/"))), None)
        if cid:
            start[cid] += 0.8 * w / total
    for cid, sim in _profile_similarity(index, arch, query).items():
        start[cid] += 0.5 * sim
    if not start:
        return {}
    adj: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for c in arch.connections:
        w = {"code": 0.3, "compose": 0.2}.get(c["kind"], 0.6)
        adj[c["from"]][c["to"]] += w
        adj[c["to"]][c["from"]] += w * 0.7
    scores, _ = push_ppr(dict(start), lambda nodes: {n: dict(adj.get(n, {})) for n in nodes},
                         restart=0.3, epsilon=1e-5)
    best = max(scores.values(), default=0.0) or 1.0
    return {k: v / best for k, v in sorted(scores.items(), key=lambda kv: -kv[1])}


def _profile(index: Index, arch: Architecture, c: dict) -> str:
    g = index.graph
    names = Counter()
    for f in c["files"]:
        if f in g.files:
            for s in g.files[f].symbols:
                if s.kind in ("class", "function", "component", "interface"):
                    names[s.name] += len(g.into.get(s.id, [])) + 1
    labels = [x["label"] for x in arch.connections if c["id"] in (x["from"], x["to"]) and x["kind"] != "code"]
    return " ".join([c["name"], c.get("summary") or "", c.get("description") or "", " ".join(c.get("languages", [])),
                     " ".join(n for n, _ in names.most_common(30)), " ".join(labels)])


def _profile_similarity(index: Index, arch: Architecture, query: str) -> Dict[str, float]:
    if not query:
        return {}
    cache = getattr(arch, "_profiles", None)
    if cache is None:
        comps = [c for c in arch.components.values()]
        vectors = index.embedder.embed([_profile(index, arch, c) for c in comps]) if comps else []
        cache = arch._profiles = {c["id"]: v for c, v in zip(comps, vectors)}
    q = index.embedder.embed([query])[0]
    out = {}
    for cid, v in cache.items():
        dot = sum(a * b for a, b in zip(q, v))
        na = math.sqrt(sum(a * a for a in q)) or 1.0
        nb = math.sqrt(sum(b * b for b in v)) or 1.0
        sim = dot / (na * nb)
        if sim > 0.1:
            out[cid] = sim
    return out


def boundary_symbols(index: Index, arch: Architecture, components: List[str], limit: int = 6) -> List[str]:
    """Code at the boundaries between these components: the symbols holding the evidence of
    the runtime connections among them (the fetch call and the route handler, the stub and
    the servicer)."""
    g = index.graph
    chosen = set(components)
    out = []
    for c in arch.connections:
        if c["kind"] == "code" or c["from"] not in chosen or (c["to"] not in chosen and arch.components[c["to"]]["kind"] == "code"):
            continue
        for ev in c["evidence"]:
            sid = g.symbol_at(ev["path"], ev["line"]) if ev.get("line") and ev["path"] in g.files else None
            if sid and g.symbols[sid].kind not in ("module", "folder") and sid not in out:
                out.append(sid)
    return out[:limit]


def _window(index: Index, ev: dict, before: int, after: int) -> Tuple[int, List[str]]:
    try:
        lines = index.read(ev["path"]).splitlines()
    except OSError:
        return 0, []
    start = max(1, ev["line"] - before)
    return start, lines[start - 1:ev["line"] + after]


def _handler(index: Index, ev: dict, max_lines: int = 45) -> Tuple[int, List[str]]:
    """The handler at an endpoint: from its annotation to the end of its body (braces matched
    for C-like languages; a fixed window otherwise)."""
    try:
        lines = index.read(ev["path"]).splitlines()
    except OSError:
        return 0, []
    start = max(1, ev["line"])
    if os.path.splitext(ev["path"])[1].lower() in (".py", ".rb", ".ex", ".exs"):
        return start, lines[start - 1:start - 1 + SNIPPET_LINES]
    depth, opened = 0, False
    for i in range(start - 1, min(len(lines), start - 1 + max_lines)):
        code = re.sub(r"\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'|//.*$", "", lines[i])
        depth += code.count("{") - code.count("}")
        opened = opened or "{" in code
        if opened and depth <= 0:
            return start, lines[start - 1:i + 1]
    return start, lines[start - 1:start - 1 + max_lines] + ["    // ... (cut)"]


def boundary_snippets(index: Index, arch: Architecture, components: List[str], query: str, limit: int = 3) -> str:
    """Code around the HTTP/gRPC endpoints between these components, for files no parser or
    language server reads (a Java service without the Java language server): the model still
    sees the route handler a client calls. Endpoints that share words with the question first."""
    chosen = set(components)
    stems = {w[:5] for w in _QUERY_WORD.findall(query.lower()) if w not in _STOP}
    indexed = index.graph.files
    candidates = []
    for c in arch.connections:
        if c["kind"] not in ("http", "grpc") or not ({c["from"], c["to"]} & chosen):
            continue
        for pair in c.get("pairs", []):
            if pair["server"]["path"] in indexed and pair["client"]["path"] in indexed:
                continue  # both ends are symbols already: PPR and the boundary seeds reach them
            near = " ".join(" ".join(_window(index, pair[s], 6, 6)[1]) for s in ("client", "server")).lower()
            score = sum(1 for s in stems if s in near)
            candidates.append((score, c, pair))
    candidates.sort(key=lambda x: -x[0])
    blocks, seen = [], set()
    for score, c, pair in candidates:
        if len(blocks) >= limit:
            break
        server = pair["server"]
        if (server["path"], server["line"]) in seen:
            continue
        seen.add((server["path"], server["line"]))
        parts = [f"#### {c['kind'].upper()} {arch.components[c['from']]['name']} -> {arch.components[c['to']]['name']}"]
        for role, ev in (("caller", pair["client"]), ("handler", server)):
            start, lines = _window(index, ev, 3, 4) if role == "caller" else _handler(index, ev)
            if not lines:
                continue
            note = "" if ev["path"] in indexed else " (not indexed: no parser or language server read this file)"
            lang = LANGS.get(os.path.splitext(ev["path"])[1].lower(), "")
            parts.append(f"{role}: {ev['path']}:{ev['line']}{note}\n```{lang}\n" + "\n".join(lines) + "\n```")
        blocks.append("\n".join(parts))
    return "\n\n".join(blocks)


# --- impact across boundaries ----------------------------------------------------------------
def _encloses(g, outer: str, inner: str) -> bool:
    current = g.symbols.get(inner)
    while current is not None:
        if current.id == outer:
            return True
        current = g.symbols.get(current.parent) if current.parent else None
    return False


def cross_impact(index: Index, arch: Architecture, changed: List[str]) -> List[dict]:
    """Code in other components that talks to the changed code at runtime: changing a route
    handler or a gRPC servicer may break its clients, which no import connects to it."""
    g = index.graph
    out, seen = [], set()
    for c in arch.connections:
        for pair in c.get("pairs", []):
            server, client = pair["server"], pair["client"]
            if server["path"] not in g.files:
                continue
            sid = g.symbol_at(server["path"], server["line"])
            if not sid or not any(sid == x or _encloses(g, x, sid) or _encloses(g, sid, x) for x in changed):
                continue
            target = g.symbol_at(client["path"], client["line"]) if client["path"] in g.files else None
            key = target or f"{client['path']}:{client['line']}"
            if key in seen or key in changed:
                continue
            seen.add(key)
            s = g.symbols.get(target) if target else None
            out.append({"id": key, "path": client["path"], "line": client["line"],
                        "qualname": (s.qualname or s.name) if s else client["text"][:80],
                        "kind": s.kind if s else "call site", "score": 1.0, "direct": True,
                        "via": f"{c['kind'].upper()} {c['label']}",
                        "component": arch.components[c["from"]]["name"]})
    return out


# --- summaries -------------------------------------------------------------------------------
SUMMARY_SYSTEM = ("You describe one component of a software system for an architecture diagram. Reply with "
                  "two sentences at most: what it is and what it does in the system. No preamble.")


def _summary_input(index: Index, arch: Architecture, c: dict) -> str:
    g = index.graph
    top = Counter()
    docs = {}
    for f in c["files"]:
        if f in g.files:
            for s in g.files[f].symbols:
                if s.kind in ("class", "function", "component", "interface", "method"):
                    top[s.id] = len(g.into.get(s.id, []))
                    if s.doc:
                        docs[s.id] = s.doc.strip().splitlines()[0][:120]
    symbols = [f"- {g.symbols[sid].qualname} ({g.symbols[sid].kind})" + (f": {docs[sid]}" if sid in docs else "")
               for sid, _ in top.most_common(15)]
    conns = [f"- {connection_text(arch, x)}" for x in arch.connections if c["id"] in (x["from"], x["to"])][:12]
    return (f"Component: {c['name']} ({c['source']}; languages: {', '.join(c.get('languages', [])) or 'unknown'}; "
            f"{c.get('fileCount', 0)} files)\n"
            + (f"Its own description: {c['description']}\n" if c.get("description") else "")
            + "Files: " + ", ".join(f.rsplit("/", 1)[-1] for f in c["files"][:20]) + "\n"
            + ("Most used symbols:\n" + "\n".join(symbols) + "\n" if symbols else "")
            + ("Connections:\n" + "\n".join(conns) + "\n" if conns else ""))


def _summary_path(storage: str) -> str:
    return os.path.join(storage, SUMMARY_FILE)


def load_summaries(storage: str, index: Optional[Index] = None) -> Dict[str, str]:
    try:
        with open(_summary_path(storage), encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return {}
    return {cid: v["summary"] for cid, v in data.items() if isinstance(v, dict) and v.get("summary")}


def summarize(index: Index, arch: Architecture, llm, storage: str, progress, force: bool = False) -> Dict[str, str]:
    """A model-written summary per code component, cached by a hash of its input: only new or
    changed components cost a model call."""
    try:
        with open(_summary_path(storage), encoding="utf-8") as fh:
            cache = json.load(fh)
    except (OSError, ValueError):
        cache = {}
    done = 0
    comps = arch.code_components()
    for i, c in enumerate(comps):
        text = _summary_input(index, arch, c)
        h = hashlib.sha1(text.encode("utf-8")).hexdigest()
        if not force and cache.get(c["id"], {}).get("hash") == h:
            c["summary"] = cache[c["id"]]["summary"]
            continue
        progress(f"Summarising {c['name']} ({i + 1} of {len(comps)})...")
        reply = llm.invoke([{"role": "system", "content": SUMMARY_SYSTEM}, {"role": "user", "content": text}]).content or ""
        reply = re.sub(r"<think>.*?</think>", "", reply, flags=re.S).strip()
        if reply:
            c["summary"] = reply[:400]
            cache[c["id"]] = {"hash": h, "summary": c["summary"]}
            done += 1
    os.makedirs(storage, exist_ok=True)
    with open(_summary_path(storage), "w", encoding="utf-8") as fh:
        json.dump(cache, fh, indent=1)
    arch._profiles = None
    return {c["id"]: c.get("summary", "") for c in comps}
