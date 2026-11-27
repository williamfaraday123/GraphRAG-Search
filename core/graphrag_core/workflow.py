"""
A small agent workflow: named steps that read and update a shared state, run in the order a
router picks, with an agent_step event after each (what it did, how long it took).

The text search builds its workflow with LangGraph; the extension uses this one (no
dependencies) for planner -> retriever -> analyst -> answer / editor.
"""
import time
from typing import Any, Callable, Dict, Optional

from . import events

Emit = Callable[[Dict[str, Any]], None]
# A step returns the state keys it changes; "_detail" (optional) describes what it did
Step = Callable[[Dict[str, Any], Emit], Dict[str, Any]]
# Picks the next step's name from the state after `current`; None ends the run
Router = Callable[[str, Dict[str, Any]], Optional[str]]

MAX_STEPS = 20


class Workflow:
    def __init__(self, steps: Dict[str, Step], first: str, router: Router, max_steps: int = MAX_STEPS):
        if first not in steps:
            raise ValueError(f"unknown first step {first!r}")
        self.steps, self.first, self.router, self.max_steps = steps, first, router, max_steps

    def run(self, state: Dict[str, Any], emit: Emit) -> Dict[str, Any]:
        current: Optional[str] = self.first
        for _ in range(self.max_steps):
            if current is None:
                return state
            if current not in self.steps:
                raise ValueError(f"router chose unknown step {current!r}")
            started = time.perf_counter()
            update = self.steps[current](state, emit) or {}
            detail = update.pop("_detail", "") if isinstance(update, dict) else ""
            state.update(update)
            elapsed = time.perf_counter() - started
            emit(events.agent_step(current, f"{detail or 'done'} ({elapsed:.1f} s)"))
            current = self.router(current, state)
        raise RuntimeError(f"workflow did not finish within {self.max_steps} steps")


def sequence(*names: str) -> Router:
    """A router that runs the steps in this order, once each."""
    order = list(names)

    def route(current: str, state: Dict[str, Any]) -> Optional[str]:
        i = order.index(current)
        return order[i + 1] if i + 1 < len(order) else None
    return route
