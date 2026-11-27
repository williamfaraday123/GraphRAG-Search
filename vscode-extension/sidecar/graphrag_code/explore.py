"""
Graph tools for the model, used only when the one-shot context has a gap (most questions never
get here: one PPR search is cheaper and more reliable with a small model).

A gap (gap()):
  - the question names nothing (no symbol, folder or cursor) and even the best match is weak;
  - a symbol the question is about didn't fit in the context.

Then the model gets up to `steps` turns with tools over the prebuilt graph, each one a graph
hop rather than a text search plus a file read. It replies with exactly one line:
  find <words>        symbols whose names or code match
  callers <symbol>    who calls or uses it
  callees <symbol>    what it calls or uses
  read <symbol>       its code (it joins the context the answer is written from)
  done
A symbol is "path :: Qualified.name" as the tools print it, or just a name. A strict one-line
format rather than an API's tool calling: small local models follow it more reliably.
"""
import re
from typing import Callable, Dict, List, Optional

from graphrag_core import events

from . import retrieve as R
from .index import Index

DEFAULT_STEPS = 4
LOW_SIMILARITY = 0.2        # best vector match below this, with nothing named: a gap
LIST_LIMIT = 8
READ_CHARS = 4000

SYSTEM = """You explore a codebase's code graph to find what a question needs before it is answered.
The search so far found too little. Reply with exactly ONE line, one of:
find <words>
callers <symbol>
callees <symbol>
read <symbol>
done
A <symbol> is written as the tools print it ("path :: Qualified.name") or as a plain name.
Read the code that answers the question; reply "done" as soon as you have it. No other text."""

_COMMAND = re.compile(r"^\s*`?(find|callers|callees|read|done)\b`?\s*:?\s*(.*?)`?\s*$", re.I)


def gap(index: Index, state: dict) -> Optional[str]:
    """Why the context isn't enough, or None."""
    ret = state["retrieval"]
    ids = {c["id"] for c in state["context"]}
    missing = [s for s in ret["about"] if s not in ids]
    if missing:
        return f"{len(missing)} symbol(s) the question is about didn't fit in the context"
    if state.get("structural") or ret["about"] or ret.get("folders") or ret.get("focus"):
        return None
    best = max(index.vector_search(state["query"], k=1).values(), default=0.0)
    if best < LOW_SIMILARITY:
        return f"the question names no symbol, folder or cursor, and the best match is weak ({best:.2f})"
    return None


class Explorer:
    def __init__(self, index: Index, llm, steps: int = DEFAULT_STEPS):
        self.index, self.llm, self.steps = index, llm, steps

    # --- tools ---------------------------------------------------------------------------------
    def _label(self, sid: str) -> str:
        s = self.index.graph.symbols[sid]
        return f"{s.path} :: {s.qualname} ({s.kind})" if s.qualname else f"{s.path} ({s.kind})"

    def resolve(self, text: str) -> Optional[str]:
        g = self.index.graph
        text = text.strip().strip("`\"'")
        if "::" in text:
            path, _, qual = text.partition("::")
            sid = f"{path.strip()}::{qual.strip().split(' (')[0]}"
            if sid in g.symbols:
                return sid
            text = qual.strip().split(" (")[0]
        hits = [s for s in g.seed_candidates(text) if g.symbols[s].kind not in ("module", "folder")]
        return hits[0] if hits else None

    def find(self, words: str) -> str:
        g = self.index.graph
        named = [s for s in self.index.graph.seed_candidates(words) if g.symbols[s].kind not in ("module", "folder")]
        found = list(dict.fromkeys(named + [s for s in self.index.vector_search(words, k=LIST_LIMIT)
                                             if g.symbols[s].kind not in ("module", "folder")]))[:LIST_LIMIT]
        return "\n".join(self._label(s) for s in found) or "nothing found"

    def neighbours(self, sid: str, incoming: bool) -> tuple:
        g = self.index.graph
        g.callers(sid)  # loads language-server edges where needed
        edges = [e for e in (g.into.get(sid, []) if incoming else g.out.get(sid, []))
                 if e.kind not in ("contains", "contains_dir")][:LIST_LIMIT * 2]
        lines = [f"{self._label(e.src if incoming else e.dst)} [{e.kind}{', guessed' if e.guessed else ''}]"
                 for e in edges]
        return "\n".join(lines) or ("nothing calls or uses it" if incoming else "it calls or uses nothing indexed"), edges

    # --- the loop -----------------------------------------------------------------------------
    def run(self, query: str, context: List[dict], reason: str, emit: Callable[[dict], None]) -> Dict[str, list]:
        """Returns {"context": items to add, "relations": edges seen, "steps": what was done}."""
        g = self.index.graph
        have = {c["id"] for c in context}
        added, relations, log = [], [], []
        known = "\n".join(f"- {c['path']} :: {c['qualname']}" for c in context) or "- (nothing)"
        messages = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": f"Question: {query}\n\nWhy more is needed: {reason}.\n\n"
                                                f"Code already found:\n{known}\n\nYour first command:"}]
        bad = 0
        for _ in range(self.steps):
            reply = self.llm.invoke(messages).content or ""
            line = next((l for l in reply.splitlines() if l.strip()), "")
            m = _COMMAND.match(line)
            if not m:
                bad += 1
                if bad >= 2:
                    break
                messages += [{"role": "assistant", "content": reply},
                             {"role": "user", "content": "Reply with one command line only: find, callers, callees, read or done."}]
                continue
            command, arg = m.group(1).lower(), m.group(2).strip().strip("`\"'").strip()
            if command == "done":
                log.append("done")
                break
            if command == "find":
                result = self.find(arg)
            else:
                sid = self.resolve(arg)
                if sid is None:
                    result = f"no symbol {arg!r}; use find first"
                elif command == "read":
                    result = R.context_block(self.index, {"id": sid})[:READ_CHARS]
                    if sid not in have:
                        have.add(sid)
                        s = g.symbols[sid]
                        added.append({"id": sid, "score": 0.0, "semantic": 0.0, "ppr": 0.0, "named": 0.0,
                                      "path": s.path, "start": s.start, "end": s.end, "kind": s.kind,
                                      "qualname": s.qualname, "signature": s.signature,
                                      "reason": "read while exploring the graph"})
                else:
                    result, edges = self.neighbours(sid, incoming=command == "callers")
                    relations += [R.relation(g, e) for e in edges]
            step = f"{command} {arg}".strip()
            log.append(step)
            emit(events.agent_step("explorer", f"{step}: {result.splitlines()[0][:120] if result else ''}"))
            messages += [{"role": "assistant", "content": line},
                         {"role": "user", "content": f"Result of {step}:\n{result}\n\nNext command:"}]
        return {"context": added, "relations": relations, "steps": log}
