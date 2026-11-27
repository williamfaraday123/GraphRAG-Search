"""
The sidecar's protocol: newline-delimited JSON over stdin/stdout.

  request       {"id": 1, "method": "ask", "params": {...}}
  response      {"id": 1, "result": {...}}   or   {"id": 1, "error": {"message": "..."}}
  notification  {"method": "event", "params": {"id": 1, "event": <graphrag_core.events dict>}}

The sidecar also asks the extension things, the other way round, while it handles a request:
  {"id": "s1", "method": "lsp/neighbours", "params": {"nodes": [...]}}  ->  {"id": "s1", "result": {...}}
(ids starting with "s"). That's how the code graph reaches VS Code's language servers.

Requests run one at a time, in order, on a worker thread (the local model answers one
request at a time anyway); the main thread keeps reading, so `update` calls queue up behind a
long `ask` and the extension's answers to the sidecar's own requests get through. stdout
carries only the protocol: logs go to stderr.
"""
import json
import logging
import os
import queue
import sys
import threading
import traceback
from typing import Any, Callable, Dict, Optional

from graphrag_core import events
from graphrag_core.llm import ChatClient

from . import parsing
from . import retrieve as R
from .embed import make_embedder
from .index import Index
from .workflows import Workflows

logger = logging.getLogger("graphrag_code")

CLIENT_TIMEOUT = 30.0  # seconds to wait for the extension to answer (language servers can be slow)


class Server:
    def __init__(self, write: Callable[[dict], None]):
        self.write = write
        self.index: Index = None  # type: ignore[assignment]
        self.flows: Workflows = None  # type: ignore[assignment]
        self.llm = None
        self.client_timeout = CLIENT_TIMEOUT
        self._asked = 0
        self._waiting: Dict[str, dict] = {}
        self._lock = threading.Lock()

    # --- asking the extension ------------------------------------------------------------------
    def ask_client(self, method: str, params: dict) -> Optional[dict]:
        """Sends a request to the extension and waits for its answer (None on an error or a
        timeout: the caller carries on without it)."""
        with self._lock:
            self._asked += 1
            rid = f"s{self._asked}"
            slot = {"event": threading.Event()}
            self._waiting[rid] = slot
        self.write({"id": rid, "method": method, "params": params})
        if not slot["event"].wait(self.client_timeout):
            with self._lock:
                self._waiting.pop(rid, None)
            logger.warning("%s: no answer from the extension within %.0f s", method, self.client_timeout)
            return None
        if "error" in slot:
            logger.warning("%s failed in the extension: %s", method, slot["error"])
            return None
        return slot.get("result")

    def answered(self, message: dict) -> None:
        """The extension's answer to one of ask_client's requests (called by the reading thread)."""
        with self._lock:
            slot = self._waiting.pop(str(message.get("id")), None)
        if slot is not None:
            slot.update({k: v for k, v in message.items() if k in ("result", "error")})
            slot["event"].set()

    # --- methods ---------------------------------------------------------------------------
    def initialize(self, p: dict, emit) -> dict:
        root = p["root"]
        storage = p.get("storagePath") or os.path.join(root, ".graphrag")
        parsing.set_mode(p.get("parser", "auto"))
        emit(events.status("LOADING", "Loading the embedding model..."))
        embedder = make_embedder(p.get("embeddingModel", "auto"))
        if self.index:
            self.index.close()
        self.index = Index(root, os.path.join(storage, "index.sqlite"), embedder,
                           vector_store=p.get("vectorStore", "auto"))
        # The extension answers lsp/neighbours from VS Code's language servers
        self.index.lsp = self.ask_client if p.get("languageServers") else None
        llm = p.get("llm") or {}
        self.llm = ChatClient(model=llm.get("model", "qwen3:4b-instruct"),
                              base_url=llm.get("baseUrl", "http://localhost:11434/v1"),
                              api_key=llm.get("apiKey") or "ollama",
                              temperature=llm.get("temperature", 0.1))
        # Graph-tool steps the model may take when a question's context has a gap (0: never)
        self.flows = Workflows(self.index, self.llm, explore_steps=int(p.get("exploreSteps", 4)))
        # What is in use, and why a fallback was used (the extension says so once)
        return {"root": self.index.root, "embedder": embedder.name, "vectorStore": self.index.vectors.name,
                "vectorStoreProblem": self.index.vector_problem, "languageServers": bool(self.index.lsp),
                # The extension sends language-server symbols for every other file
                "parsedExtensions": parsing.parsed_extensions(), **parsing.status()}

    def _ready(self):
        if not self.index:
            raise RuntimeError("not initialized")

    def index_workspace(self, p: dict, emit) -> dict:
        self._ready()
        emit(events.status("INDEXING", "Parsing Python files..."))
        return self.index.build(lambda m: emit(events.status("INDEXING", m)))

    def update(self, p: dict, emit) -> dict:
        self._ready()
        return self.index.update(p.get("paths", []), lambda m: emit(events.status("INDEXING", m)))

    def add_symbols(self, p: dict, emit) -> dict:
        """Symbols the language servers found in files no parser reads."""
        self._ready()
        return self.index.add_external(p.get("files", []), lambda m: emit(events.status("INDEXING", m)))

    def remove_files(self, p: dict, emit) -> dict:
        self._ready()
        return self.index.remove_files(p.get("paths", []), lambda m: emit(events.status("INDEXING", m)))

    def graph(self, p: dict, emit) -> dict:
        """The neighbourhood of the cursor (and/or a query) by PPR, without the model."""
        self._ready()
        result = R.retrieve(self.index, p.get("query", ""), p.get("focus"), p.get("openFiles", []))
        highlight = [result["focus"]] if result["focus"] else result["about"]
        return {"graph": R.graph_view(self.index, result["ppr"], highlight), "focus": result["focus"],
                "context": result["context"]}

    def impact(self, p: dict, emit) -> dict:
        self._ready()
        symbols = list(p.get("symbols") or [])
        focus = p.get("focus")
        if not symbols and focus:
            sid = self.index.graph.symbol_at(self.index.rel(focus["path"]), int(focus["line"]))
            symbols = [sid] if sid else []
        affected = R.impact(self.index, symbols)
        scores = {a["id"]: a["score"] for a in affected}
        scores.update({s: max(scores.values(), default=1.0) for s in symbols})
        return {"symbols": symbols, "impact": affected,
                "graph": R.graph_view(self.index, scores, symbols)}

    def ask(self, p: dict, emit) -> dict:
        self._ready()
        return self.flows.ask(p, emit)

    def edit(self, p: dict, emit) -> dict:
        self._ready()
        return self.flows.edit(p, emit)

    def stats(self, p: dict, emit) -> dict:
        self._ready()
        return self.index.graph.stats()

    METHODS = {"initialize": initialize, "index": index_workspace, "update": update, "graph": graph,
               "impact": impact, "ask": ask, "edit": edit, "stats": stats,
               "addSymbols": add_symbols, "removeFiles": remove_files}

    # --- dispatch --------------------------------------------------------------------------
    def handle(self, message: dict):
        rid = message.get("id")
        method = self.METHODS.get(message.get("method"))

        def emit(event: Dict[str, Any]):
            self.write({"method": "event", "params": {"id": rid, "event": event}})
        try:
            if method is None:
                raise ValueError(f"unknown method {message.get('method')!r}")
            result = method(self, message.get("params") or {}, emit)
            self.write({"id": rid, "result": result})
        except Exception as e:  # reported to the extension, which shows it
            logger.error("%s failed:\n%s", message.get("method"), traceback.format_exc())
            self.write({"id": rid, "error": {"message": f"{type(e).__name__}: {e}"}})


def main():
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(levelname)s %(name)s: %(message)s")
    stdin = open(sys.stdin.fileno(), encoding="utf-8", errors="replace", closefd=False)
    stdout = open(sys.stdout.fileno(), "w", encoding="utf-8", closefd=False)
    lock = threading.Lock()

    def write(obj: dict):
        line = json.dumps(obj, ensure_ascii=False)
        with lock:
            stdout.write(line + "\n")
            stdout.flush()

    server = Server(write)
    work: "queue.Queue[dict]" = queue.Queue()

    def worker():
        while True:
            message = work.get()
            if message is None:
                return
            server.handle(message)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            logger.warning("ignoring a line that isn't JSON: %r", line[:200])
            continue
        if message.get("method") == "shutdown":
            break
        if "method" not in message:  # the extension answering one of the sidecar's requests
            server.answered(message)
            continue
        work.put(message)
    work.put(None)
    thread.join(timeout=5)


if __name__ == "__main__":
    main()
