"""
Coverage check without a model call: the context is enough when every node it must contain
(the entities a question names; the callers of a function about to change) is in it, or is
mentioned by something in it.
"""
from typing import Callable, Dict, Hashable, Iterable, List, Optional


def uncovered(required: Iterable[Hashable], context: Iterable[Hashable],
              covers: Optional[Callable[[Hashable, Hashable], bool]] = None) -> List[Hashable]:
    """The required nodes, in order, that nothing in `context` covers. A node covers itself;
    `covers(context_node, required_node)` can say more (e.g. a chunk mentioning an entity)."""
    have = list(context)
    have_set = set(have)
    missing = []
    for node in required:
        if node in have_set:
            continue
        if covers and any(covers(c, node) for c in have):
            continue
        if node not in missing:
            missing.append(node)
    return missing


def required_by(targets: Iterable[Hashable],
                dependents: Callable[[Hashable], Iterable[Hashable]],
                limit: int = 50) -> Dict[Hashable, List[Hashable]]:
    """For each target, what must be read with it (e.g. its callers), capped at `limit` in all.
    Returns {required node: [targets that need it]}."""
    need: Dict[Hashable, List[Hashable]] = {}
    for t in targets:
        for d in dependents(t):
            if d == t:
                continue
            if d not in need and len(need) >= limit:
                return need
            need.setdefault(d, [])
            if t not in need[d]:
                need[d].append(t)
    return need
