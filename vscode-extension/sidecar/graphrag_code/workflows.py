"""
The agent workflows, on graphrag_core.workflow:

  ask:  planner -> [graph] -> retriever -> analyst -> [explorer] -> answer
  edit: planner -> retriever -> analyst -> editor -> impact

planner    what the request is about: the symbol under the cursor, the symbols and folders it
           names, and whether it's a structural question (structural.py) or a broad question
           about the system (architecture.py)
graph      structural questions only ("who calls X", "does A use B"): the exact answer from the
           code graph; the model then explains it, with the code of the symbols involved.
           Broad questions ("how does a search request flow through the system?") get the
           architecture graph: components and their connections, with evidence
retriever  components first (top-down), then seeds -> local push PPR -> hybrid ranking ->
           context (retrieve.py)
analyst    coverage check: for an edit, every caller of the changed symbols is in context; for
           a question, whether the context has a gap (explore.py)
explorer   only when there's a gap: the model takes a few one-line steps with graph tools
           (find, callers, callees, read); what it reads joins the context
answer     the model answers from the context, streamed
editor     the model writes edit blocks (edits.py); a proposal that doesn't parse goes back
           to it once with the syntax errors
impact     PPR backwards along dependencies from the changed symbols: what may break

Verifying an applied edit (diagnostics, tests) happens in the extension, which then calls
`revise` with what failed: the editor runs again on the current files with that feedback.
"""
from typing import Any, Callable, Dict, List, Optional

from graphrag_core import events
from graphrag_core.workflow import Workflow, sequence

from . import architecture as A
from . import retrieve as R
from . import explore as X
from . import structural as S
from .edits import apply_blocks, parse_blocks
from .index import Index

Emit = Callable[[Dict[str, Any]], None]

ASK_SYSTEM = (
    "You answer questions about a codebase. You get code, and relationships the code graph found "
    "by static analysis (which symbol calls, imports, extends or uses which, across the whole "
    "repository, including code you aren't shown). Use only these. The relationships are reliable "
    "evidence about how code connects: use them for questions about which code uses which; edges "
    "marked (guessed) are likely but not certain. Never conclude that something doesn't happen "
    "because the code shown doesn't show it: say that the context doesn't show it and what's "
    "missing. The system architecture, when given, lists components and connections found from "
    "build files and patterns in the code: describe only connections it or the code shows, and "
    "don't add steps between components that neither shows. Refer to code as `path:line` (for "
    "example `app/models.py:42`) so the reader can open it. Be concise; use Markdown."
)


def relations_text(ret: dict) -> str:
    """The relationships section of a prompt (retrieve.relationships)."""
    parts = []
    if ret.get("folderLinks"):
        parts.append("Between the folders the question names (every edge in the graph):\n"
                     + "\n".join(f"- {l['text']}" for l in ret["folderLinks"]))
    if ret.get("relations"):
        parts.append("Relationships (code graph, static analysis):\n"
                     + "\n".join(f"- {r['text']}" for r in ret["relations"]))
    return "\n\n".join(parts)

EDIT_SYSTEM = """You change a codebase (Python, TypeScript or JavaScript). You get the code that matters (the symbols to change,
their callers and related code) and a change request. Reply with edit blocks only:

=== REPLACE <path> :: <Qualified.name>
<the complete new source of that symbol, including its decorators and docstring>
=== END

Other block types:
=== INSERT_AFTER <path> :: <Qualified.name>   (new code placed after that symbol)
=== APPEND <path>                             (new code at the end of the file)
=== CREATE <path>                             (a new file: its whole content)

Rules:
- Each piece of code you are given starts with a header "### <path> :: <Qualified.name>". Name
  the symbol to change exactly like that: "=== REPLACE pkg/models.py :: User.greet".
- A REPLACE block holds the whole symbol, not just the changed lines.
- If you change a function's signature or behaviour, also update every caller shown that needs it.
- No code fences, no explanations outside the blocks. One short line before the first block
  may say what you changed."""


def _llm_messages(system: str, user: str):
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _context_text(index: Index, items: List[dict]) -> str:
    return "\n\n".join(R.context_block(index, it) for it in items)


class Workflows:
    def __init__(self, index: Index, llm, explore_steps: int = X.DEFAULT_STEPS,
                 architecture: Optional[Callable[[], Any]] = None):
        self.index, self.llm, self.explore_steps = index, llm, explore_steps
        # () -> the current architecture.Architecture (cached by the server), or None
        self._architecture = architecture

    def arch(self):
        if self._architecture is None:
            return None
        try:
            return self._architecture()
        except Exception as e:  # the architecture is a help, never a reason for a request to fail
            import logging
            logging.getLogger(__name__).warning("architecture unavailable: %s", e)
            return None

    # --- shared steps ----------------------------------------------------------------------
    def planner(self, state, emit):
        emit(events.status("PLANNING", "Finding the code the request is about..."))
        g = self.index.graph
        named = g.seed_candidates(state["query"])
        focus = state.get("focus")
        focus_id = g.symbol_at(self.index.rel(focus["path"]), int(focus["line"])) if focus and focus.get("path") else None
        names = [g.symbols[s].qualname or g.symbols[s].name for s, w in named.items() if w >= 1.0]
        hints = [g.symbols[s].qualname or g.symbols[s].name for s, w in named.items() if w < 1.0]
        folders = list(g.named_folders(state["query"]))
        detail = []
        if focus_id:
            detail.append(f"cursor in {g.symbols[focus_id].qualname or g.symbols[focus_id].path}")
        if names:
            detail.append("names " + ", ".join(names[:5]))
        if folders:
            detail.append("folders " + ", ".join(f"{f}/" for f in folders[:5]))
        if hints:
            detail.append("weak hints " + ", ".join(hints[:5]))
        structural = S.classify(self.index, state["query"], focus_id) if state["mode"] == "ask" else None
        if (not structural and state["mode"] == "ask" and not names and not focus_id
                and A.is_architecture_question(state["query"]) and self.arch() is not None):
            # A broad question about the system: the architecture graph answers it first
            structural = {"kind": "architecture", "targets": []}
            detail.append("architecture question: answered from the component graph")
        elif structural:
            detail.append(f"structural question ({structural['kind']}: "
                          + ", ".join(t.label for t in structural["targets"]) + "): answered from the graph")
        return {"structural": structural, "_detail": "; ".join(detail) or "no symbol named: semantic search only"}

    def graph_query(self, state, emit):
        emit(events.status("GRAPH", "Answering from the code graph..."))
        if state["structural"]["kind"] == "architecture":
            result = A.answer(self.index, self.arch(), state["query"])
            return {"graph_answer": result, "targets": result["symbols"][:4], "_detail": result["summary"][:300]}
        result = S.answer(self.index, state["structural"], self.arch())
        # The code of the symbols the answer involves is read with it
        return {"graph_answer": result, "targets": result["symbols"][:4], "_detail": result["summary"]}

    def retriever(self, state, emit):
        emit(events.status("RETRIEVING", "Ranking symbols by similarity and graph proximity (PPR)..."))
        result = R.retrieve(self.index, state["query"], state.get("focus"), state.get("openFiles", []),
                            mode=state["mode"], targets=state.get("targets", []), arch=self.arch())
        comps = ", ".join(c["name"] for c in result["components"][:3])
        return {"retrieval": result, "context": result["context"],
                "_detail": (f"components first: {comps}; " if comps else "")
                           + f"{len(result['context'])} symbols from {result['fetched']} explored, "
                             f"{len(result['relations'])} relationship(s)"}

    def analyst(self, state, emit):
        emit(events.status("ANALYZING", "Checking the context covers what the request touches..."))
        ret = state["retrieval"]
        added = [c for c in state["context"] if "coverage check" in c.get("reason", "")]
        if state["mode"] != "edit":
            links = ret.get("folderLinks") or []
            gap = X.gap(self.index, state) if self.explore_steps > 0 else None
            detail = f"gap: {gap}; exploring the graph" if gap else "context covers the symbols the question is about"
            if links:
                detail += "; " + "; ".join(l["text"] for l in links)
            return {"gap": gap, "_detail": detail}
        detail = f"added {len(added)} caller(s)" if added else "all callers already in context"
        settled = ret.get("settled")
        if settled and settled["checked"]:
            detail += (f"; language server checked the callers of {settled['checked']} symbol(s): "
                       f"{settled['confirmed']} guess(es) confirmed, {settled['dropped']} dropped, "
                       f"{settled['added']} caller(s) added")
        if ret["uncoveredCallers"]:
            detail += f"; {len(ret['uncoveredCallers'])} more caller(s) didn't fit the context budget"
        return {"_detail": detail}

    def explorer(self, state, emit):
        emit(events.status("EXPLORING", "The search found too little: exploring the code graph..."))
        found = X.Explorer(self.index, self.llm, self.explore_steps).run(
            state["query"], state["context"], state["gap"], emit)
        ret = dict(state["retrieval"])
        seen = {(r["source"], r["target"], r["kind"]) for r in ret["relations"]}
        ret["relations"] = ret["relations"] + [r for r in found["relations"]
                                               if (r["source"], r["target"], r["kind"]) not in seen][:R.MAX_RELATIONS]
        return {"context": state["context"] + found["context"], "retrieval": ret, "explored": found["steps"],
                "_detail": f"{len(found['steps'])} step(s), read {len(found['context'])} more symbol(s)"}

    # --- ask -------------------------------------------------------------------------------
    def answer(self, state, emit):
        emit(events.status("ANSWERING", "Writing the answer..."))
        links = relations_text(state["retrieval"])
        graph = state.get("graph_answer")
        architecture = state["retrieval"].get("architecture") or ""
        if graph and graph["kind"] == "architecture":
            graph_text = (f"The system's architecture, from build files, docker-compose and static analysis:\n"
                          f"{graph['text']}")
            architecture = ""  # already given
        else:
            graph_text = S.answer_text(graph) if graph else ""
        boundary = state["retrieval"].get("boundaryCode") or ""
        user = ((f"{graph_text}\n\nAnswer from this graph query result first; the code "
                 "and relationships below are for detail.\n\n" if graph else "")
                + (f"System architecture (components and how they connect):\n{architecture}\n\n" if architecture else "")
                + (f"Code at the boundaries between components (endpoints and their callers):\n\n{boundary}\n\n"
                   if boundary else "")
                + f"Code:\n\n{_context_text(self.index, state['context'])}\n\n"
                + (f"{links}\n\n" if links else "")
                + f"Question: {state['query']}")
        parts = []
        for piece in self.llm.stream(_llm_messages(ASK_SYSTEM, user)):
            parts.append(piece)
            emit(events.chunk(piece))
        return {"answer": "".join(parts), "_detail": "answer written"}

    def ask(self, params: dict, emit: Emit) -> dict:
        state = {"query": params["query"], "focus": params.get("focus"),
                 "openFiles": params.get("openFiles", []), "mode": "ask"}
        order = sequence("planner", "graph", "retriever", "analyst", "explorer", "answer")

        def route(current, st):
            nxt = order(current, st)
            if nxt == "graph" and not st.get("structural"):
                return "retriever"
            if nxt == "explorer" and not st.get("gap"):  # the one-shot context is enough
                return "answer"
            return nxt
        wf = Workflow({"planner": self.planner, "graph": self.graph_query, "retriever": self.retriever,
                       "analyst": self.analyst, "explorer": self.explorer, "answer": self.answer},
                      "planner", route)
        state = wf.run(state, emit)
        ret = state["retrieval"]
        return {"answer": state["answer"], "context": state["context"],
                "relations": ret["relations"], "folderLinks": ret["folderLinks"],
                "components": ret.get("components", []),
                "graphAnswer": state.get("graph_answer"), "explored": state.get("explored", []),
                "graph": R.graph_view(self.index, ret["ppr"], [c["id"] for c in state["context"]])}

    # --- edit ------------------------------------------------------------------------------
    def editor(self, state, emit):
        emit(events.status("EDITING", "Writing the change..."))
        feedback = state.get("feedback")
        user = f"Code:\n\n{_context_text(self.index, state['context'])}\n\nChange request: {state['query']}"
        if feedback:
            user += ("\n\nThe previous attempt was applied, but checking it found these problems. "
                     f"The code above is the current version. Fix them:\n{feedback}")
        messages = _llm_messages(EDIT_SYSTEM, user)
        proposal, reply, attempts = None, "", 0
        for attempts in (1, 2):
            reply = "".join(self._stream(messages, emit))
            proposal = apply_blocks(parse_blocks(reply), self._read, self._symbols_of)
            if proposal.files and not proposal.syntax_errors:
                break
            problem = ("\n".join(proposal.syntax_errors) if proposal.syntax_errors else
                       "No edit block could be applied: " + ("; ".join(proposal.skipped) or "none found"))
            if attempts == 2:
                break
            emit(events.agent_step("editor", f"retrying: {problem[:200]}"))
            valid = "\n".join(f"  {c['path']} :: {c['qualname']}" for c in state["context"])
            messages = messages + [{"role": "assistant", "content": reply},
                                   {"role": "user", "content": f"That didn't work: {problem}\n"
                                                               f"The symbols you can REPLACE are:\n{valid}\n"
                                                               "Reply again with corrected edit blocks only."}]
        # The one line of prose the model may write before its first block
        before_blocks = reply.split("===", 1)[0]
        summary = next((l.strip() for l in before_blocks.splitlines() if l.strip() and not l.strip().startswith("```")), "")
        return {"proposal": proposal, "reply": reply, "summary": summary[:300],
                "_detail": f"{len(proposal.files)} file(s) changed" + (f", {attempts} attempts" if attempts > 1 else "")}

    def _stream(self, messages, emit):
        for piece in self.llm.stream(messages):
            emit(events.chunk(piece))
            yield piece

    def _symbols_of(self, path: str, text: str):
        """Symbols of a file only a language server reads, valid for its text on disk."""
        f = self.index.results.get(path)
        if f is None or f.parser != "lsp" or text != self._read(path):
            return None
        return f.symbols

    def _read(self, path: str) -> Optional[str]:
        try:
            return self.index.read(path)
        except OSError:
            return None

    def impact_step(self, state, emit):
        changed = state["proposal"].changed_symbols
        affected = R.impact(self.index, changed)
        arch = self.arch()
        across = A.cross_impact(self.index, arch, changed) if arch is not None and changed else []
        affected = across + affected
        direct = sum(1 for a in affected if a["direct"])
        return {"impact": affected, "_detail": f"{direct} direct dependent(s), {len(affected) - direct} further"
                                               + (f"; {len(across)} in other components over HTTP/gRPC" if across else "")}

    def edit(self, params: dict, emit: Emit) -> dict:
        state = {"query": params["instruction"], "focus": params.get("focus"),
                 "openFiles": params.get("openFiles", []), "mode": "edit",
                 "targets": params.get("targets", []), "feedback": params.get("feedback")}
        wf = Workflow({"planner": self.planner, "retriever": self.retriever, "analyst": self.analyst,
                       "editor": self.editor, "impact": self.impact_step}, "planner",
                      sequence("planner", "retriever", "analyst", "editor", "impact"))
        state = wf.run(state, emit)
        p = state["proposal"]
        return {
            "summary": state["summary"],
            "files": [{"path": path, **f} for path, f in p.files.items()],
            "changedSymbols": p.changed_symbols,
            "skipped": p.skipped,
            "syntaxErrors": p.syntax_errors,
            "impact": state["impact"],
            "context": state["context"],
            "graph": R.graph_view(self.index, state["retrieval"]["ppr"],
                                  [c["id"] for c in state["context"]] + p.changed_symbols),
        }
