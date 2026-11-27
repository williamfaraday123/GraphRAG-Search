"""
The workspace index: each file's parse result in SQLite (so a restart re-reads only files whose
size, modification time or parser changed), each symbol's embedding in the vector store
(LanceDB, else SQLite: vectors.py), and the in-memory code graph built from the parse results.

Incremental: update(paths) re-parses just those files (the file watcher sends them), then
rebuilds the graph from all parse results, which is fast because resolution needs no parsing,
and embeds only symbols whose text changed.

Language servers (when the extension enables them, `lsp`): symbols of files no parser reads
arrive from VS Code (add_external, parser "lsp"); their edges are asked for lazily and cached
here, keyed by symbol, until a file they involve changes. The same channel settles the
parser's guessed callers before an edit (settle_callers).
"""
import hashlib
import json
import logging
import os
import sqlite3
import time
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from . import parsing
from .embed import HashingEmbedder, symbol_text
from .graph import CodeGraph
from .parse_python import MAX_TEXT, FileResult, Symbol
from .parse_treesitter import strip_extension
from .vectors import SQLiteVectors, make_vector_store

# Asks the extension (VS Code) something: (method, params) -> result, or None if it can't answer
LanguageServers = Callable[[str, dict], Optional[dict]]
LSP_BATCH = 40  # symbols per lsp/neighbours request

logger = logging.getLogger(__name__)

SKIP_DIRS = {".git", ".hg", ".svn", "node_modules", "venv", ".venv", "env", ".env", "__pycache__",
             "build", "dist", "out", "coverage", "site-packages", ".tox", ".mypy_cache", ".pytest_cache",
             ".idea", ".vscode", ".graphrag", "wheel_cache", "torch_cache", ".eggs", ".next", ".nuxt",
             ".turbo", ".cache", "bower_components", "vendor"}
MAX_FILE_BYTES = 1_000_000
EMBED_BATCH = 64

Progress = Callable[[str], None]


def looks_minified(raw: bytes) -> bool:
    """Bundled or minified JavaScript (main.4f2a1c.js, vendor bundles): very long lines."""
    if len(raw) < 5000:
        return False
    head = raw[:200_000]
    lines = head.split(b"\n")
    return len(head) / max(len(lines), 1) > 250 or any(len(l) > 5000 for l in lines[:100])


class Index:
    def __init__(self, root: str, db_path: str, embedder=None, vector_store: str = "auto"):
        self.root = os.path.abspath(root)
        folder = os.path.dirname(os.path.abspath(db_path))
        os.makedirs(folder, exist_ok=True)
        self.db = sqlite3.connect(db_path, check_same_thread=False)
        self.db.execute("CREATE TABLE IF NOT EXISTS files (path TEXT PRIMARY KEY, mtime REAL, size INTEGER, "
                        "sha TEXT, result TEXT)")
        self.embedder = embedder or HashingEmbedder()
        self.vectors, self.vector_problem = make_vector_store(vector_store, folder, self.db, self.embedder)
        self.results: Dict[str, FileResult] = {}
        self.graph = CodeGraph([])
        self.lsp: Optional[LanguageServers] = None
        # symbol id -> [(source, target, kind, count)] from the language server, and whether
        # it has call hierarchy for that symbol
        self._lsp_edges: Dict[str, List[Tuple[str, str, str, int]]] = {}
        self._lsp_supported: Dict[str, bool] = {}

    # --- files -----------------------------------------------------------------------------
    def rel(self, path: str) -> str:
        full = path if os.path.isabs(path) else os.path.join(self.root, path)
        return os.path.relpath(full, self.root).replace(os.sep, "/")

    def abs(self, rel: str) -> str:
        return os.path.join(self.root, *rel.split("/"))

    def walk(self) -> List[str]:
        found = []
        for dirpath, dirnames, filenames in os.walk(self.root):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.endswith(".egg-info")]
            for name in filenames:
                rel = self.rel(os.path.join(dirpath, name))
                if parsing.supported(rel):
                    found.append(rel)
        return sorted(found)

    def _parse(self, rel: str) -> Tuple[Optional[FileResult], bool]:
        """(parse result, changed). None when the file is gone, too large or unsupported."""
        full = self.abs(rel)
        try:
            st = os.stat(full)
        except OSError:
            return None, True
        if st.st_size > MAX_FILE_BYTES or not parsing.supported(rel):
            return None, True
        parser = parsing.parser_name()
        row = self.db.execute("SELECT mtime, size, sha, result FROM files WHERE path = ?", (rel,)).fetchone()
        cached = FileResult.from_json(json.loads(row[3])) if row else None
        if cached and cached.parser == parser and row[0] == st.st_mtime and row[1] == st.st_size:
            return cached, False
        with open(full, "rb") as fh:
            raw = fh.read()
        if not rel.endswith(".py") and looks_minified(raw):
            return None, True  # a build output (bundle), not source: thousands of meaningless symbols
        sha = hashlib.sha1(raw).hexdigest()
        if cached and cached.parser == parser and row[2] == sha:
            self.db.execute("UPDATE files SET mtime = ?, size = ? WHERE path = ?", (st.st_mtime, st.st_size, rel))
            return cached, False
        result = parsing.parse_source(rel, raw.decode("utf-8", "replace"))
        if result is None:
            return None, True
        self.db.execute("INSERT OR REPLACE INTO files VALUES (?, ?, ?, ?, ?)",
                        (rel, st.st_mtime, st.st_size, sha, json.dumps(result.to_json())))
        return result, True

    # --- building --------------------------------------------------------------------------
    def build(self, progress: Progress = lambda m: None) -> dict:
        """Index the whole workspace (unchanged files come from the database)."""
        started = time.perf_counter()
        paths = self.walk()
        changed = 0
        results = {}
        for i, rel in enumerate(paths):
            result, was_changed = self._parse(rel)
            if result:
                results[rel] = result
                changed += was_changed
            if i % 200 == 199:
                progress(f"Parsed {i + 1} of {len(paths)} files")
        gone = set()
        for rel, mtime, size, stored in self.db.execute("SELECT path, mtime, size, result FROM files").fetchall():
            if rel in results:
                continue
            kept = FileResult.from_json(json.loads(stored))
            if kept.parser == "lsp" and self._unchanged(rel, mtime, size):
                results[rel] = kept  # language-server symbols: VS Code re-sends them when the file changes
            else:
                gone.add(rel)
        for rel in gone:
            self.db.execute("DELETE FROM files WHERE path = ?", (rel,))
        self.db.commit()
        self.results = results
        stats = self._rebuild(progress)
        stats.update(parsed=changed, removed=len(gone), seconds=round(time.perf_counter() - started, 2))
        return stats

    def _unchanged(self, rel: str, mtime: float, size: int) -> bool:
        try:
            st = os.stat(self.abs(rel))
        except OSError:
            return False
        return st.st_mtime == mtime and st.st_size == size

    def update(self, paths: Iterable[str], progress: Progress = lambda m: None) -> dict:
        """Re-index just these files (changed, created or deleted)."""
        paths = list(paths)
        self._forget_lsp([self.rel(p) for p in paths])
        for path in paths:
            rel = self.rel(path)
            if rel.startswith("..") or not parsing.supported(rel):
                continue
            result, _ = self._parse(rel)
            if result:
                self.results[rel] = result
            else:
                self.results.pop(rel, None)
                self.db.execute("DELETE FROM files WHERE path = ?", (rel,))
        self.db.commit()
        return self._rebuild(progress)

    # --- symbols from language servers ---------------------------------------------------------
    def add_external(self, files: List[dict], progress: Progress = lambda m: None) -> dict:
        """Symbols VS Code's language servers found in files no parser reads:
        [{path, lang, symbols: [{name, kind, qualname, parent, start, end, selLine, selCol}]}]."""
        changed = []
        for f in files:
            rel = self.rel(f["path"])
            if rel.startswith("..") or parsing.supported(rel):
                continue  # outside the workspace, or a parser reads it better
            full = self.abs(rel)
            try:
                st = os.stat(full)
                with open(full, "rb") as fh:
                    raw = fh.read()
            except OSError:
                continue
            lines = raw.decode("utf-8", "replace").splitlines(keepends=True)
            module_id = f"{rel}::"
            result = FileResult(path=rel, module=strip_extension(rel), lang=f.get("lang") or "", parser="lsp")
            result.symbols.append(Symbol(id=module_id, path=rel, kind="module", name=rel.rsplit("/", 1)[-1],
                                         qualname="", start=1, end=max(len(lines), 1), signature=rel, doc="",
                                         text="".join(lines[:40])[:MAX_TEXT]))
            for s in f.get("symbols", []):
                start, end = int(s["start"]), int(s["end"])
                first = lines[start - 1].strip() if 0 < start <= len(lines) else s["name"]
                result.symbols.append(Symbol(
                    id=f"{rel}::{s['qualname']}", path=rel, kind=s["kind"], name=s["name"], qualname=s["qualname"],
                    start=start, end=end, signature=first[:200], doc="", text="".join(lines[start - 1:end])[:MAX_TEXT],
                    parent=f"{rel}::{s['parent']}" if s.get("parent") else module_id,
                    sel_line=int(s.get("selLine") or 0), sel_col=int(s.get("selCol") or 0)))
            self.db.execute("INSERT OR REPLACE INTO files VALUES (?, ?, ?, ?, ?)",
                            (rel, st.st_mtime, st.st_size, hashlib.sha1(raw).hexdigest(), json.dumps(result.to_json())))
            self.results[rel] = result
            changed.append(rel)
        self.db.commit()
        self._forget_lsp(changed)
        return self._rebuild(progress)

    def remove_files(self, paths: Iterable[str], progress: Progress = lambda m: None) -> dict:
        rels = [self.rel(p) for p in paths]
        for rel in rels:
            self.results.pop(rel, None)
            self.db.execute("DELETE FROM files WHERE path = ?", (rel,))
        self.db.commit()
        self._forget_lsp(rels)
        return self._rebuild(progress)

    def _forget_lsp(self, rels: Iterable[str]):
        """Cached language-server edges that involve these files are stale."""
        paths = set(rels)
        if not paths:
            return
        in_paths = lambda sid: sid.split("::", 1)[0] in paths
        for sid in [s for s, edges in self._lsp_edges.items()
                    if in_paths(s) or any(in_paths(a) or in_paths(b) for a, b, _, _ in edges)]:
            self._lsp_edges.pop(sid, None)
            self._lsp_supported.pop(sid, None)

    def _node_ref(self, sid: str) -> dict:
        s = self.graph.symbols[sid]
        return {"id": sid, "path": s.path, "line": s.start, "end": s.end, "name": s.name, "kind": s.kind,
                "selLine": s.sel_line, "selCol": s.sel_col}

    def _locate(self, loc: dict) -> Optional[str]:
        """A location the language server gave ({path, line}) -> the symbol containing it."""
        path = loc.get("path")
        if not path:
            return None
        return self.graph.symbol_at(self.rel(path), int(loc.get("line") or 0))

    def lsp_edges(self, ids: List[str]) -> Dict[str, List[Tuple[str, str, str, int]]]:
        """Edges of these symbols from the language server (cached). Symbols it can't answer
        for now (no reply, timed out) get none and are asked again next time."""
        need = [i for i in ids if i not in self._lsp_edges and i in self.graph.symbols]
        for start in range(0, len(need), LSP_BATCH) if self.lsp else []:
            batch = need[start:start + LSP_BATCH]
            reply = self.lsp("lsp/neighbours", {"nodes": [self._node_ref(i) for i in batch]})
            if not reply:
                continue
            for sid, rel in (reply.get("results") or {}).items():
                if sid not in self.graph.symbols:
                    continue
                edges = []
                for direction, kind, incoming in (("incoming", "calls", True), ("outgoing", "calls", False),
                                                  ("subtypes", "extends", True), ("supertypes", "extends", False),
                                                  ("references", "uses", True)):
                    for loc in rel.get(direction) or []:
                        other = self._locate(loc)
                        if other and other != sid:
                            count = int(loc.get("count") or 1)
                            edges.append((other, sid, kind, count) if incoming else (sid, other, kind, count))
                self._lsp_edges[sid] = edges
                self._lsp_supported[sid] = bool(rel.get("supported"))
        return {i: self._lsp_edges.get(i, []) for i in ids}

    def settle_callers(self, ids: List[str]) -> Optional[Dict[str, int]]:
        """Before an edit: the language server's exact callers of these symbols replace the
        parser's guesses (graph.apply_exact_callers). None when no language server is in use."""
        if not self.lsp:
            return None
        targets = [i for i in ids if i in self.graph.symbols and self.graph.symbols[i].kind not in ("module",)]
        edges = self.lsp_edges(targets)
        total = {"checked": 0, "confirmed": 0, "dropped": 0, "added": 0}
        for t in targets:
            if not self._lsp_supported.get(t):
                continue  # this server has no call hierarchy here: keep the parser's view
            callers = {src for src, dst, kind, _ in edges[t] if dst == t and kind == "calls"}
            for k, v in self.graph.apply_exact_callers(t, callers).items():
                total[k] += v
            total["checked"] += 1
        return total

    def fence_language(self, rel: str) -> str:
        """Code fence language for a file (a language server's language id for its files)."""
        f = self.results.get(rel)
        if f is not None and f.parser == "lsp":
            return f.lang
        return parsing.code_language(rel)

    def _rebuild(self, progress: Progress) -> dict:
        self.graph = CodeGraph(self.results.values())
        if self.lsp:
            self.graph.external = self.lsp_edges
        embedded = self._embed(progress)
        stats = self.graph.stats()
        stats.update(embedded=embedded, embedder=self.embedder.name, vectorStore=self.vectors.name,
                     languages=sorted({f.lang for f in self.results.values()}), **parsing.status())
        return stats

    def _embed(self, progress: Progress) -> int:
        symbols = [s for s in self.graph.symbols.values() if s.kind not in ("module", "folder")]
        stored = self.vectors.hashes()
        todo = []
        for s in symbols:
            text = symbol_text(s)
            h = hashlib.sha1(text.encode("utf-8")).hexdigest()
            if stored.get(s.id) != h:
                todo.append((s, text, h))
        for i in range(0, len(todo), EMBED_BATCH):
            batch = todo[i:i + EMBED_BATCH]
            vectors = self.embedder.embed([t for _, t, _ in batch])
            self.vectors.upsert([(s.id, s.path, s.kind, h, list(map(float, v)))
                                 for (s, _, h), v in zip(batch, vectors)])
            progress(f"Embedded {min(i + EMBED_BATCH, len(todo))} of {len(todo)} changed symbols")
        live = {s.id for s in symbols}
        self.vectors.delete([sid for sid in stored if sid not in live])
        self.vectors.finish()
        self.db.commit()
        return len(todo)

    # --- search ----------------------------------------------------------------------------
    def vector_search(self, query: str, k: int = 20) -> Dict[str, float]:
        if not query:
            return {}
        hits = self.vectors.search(self.embedder.embed([query])[0], k)
        return {sid: score for sid, score in hits.items() if sid in self.graph.symbols}

    def read(self, rel: str) -> str:
        with open(self.abs(rel), encoding="utf-8", errors="replace") as fh:
            return fh.read()

    def close(self):
        self.vectors.close()
        self.db.close()


__all__ = ["Index", "SQLiteVectors"]
