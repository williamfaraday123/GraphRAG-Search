"""
Hybrid ranking: several scores per item (semantic similarity, PPR, ...), each scaled to 0-1 by
its best value, then mixed with fixed weights. A signal no item has (all zero) drops out and
the other weights are rescaled, so e.g. a search without name matches still ranks 0-1.
"""
from typing import Dict, List, Mapping


def normalise(values: Mapping[str, float]) -> Dict[str, float]:
    best = max(values.values(), default=0.0)
    return {k: (v / best if best > 0 else 0.0) for k, v in values.items()}


def hybrid_rank(items: List[str], signals: Mapping[str, Mapping[str, float]],
                weights: Mapping[str, float]) -> List[dict]:
    """items: ids; signals: {signal name: {id: raw score}}; weights: {signal name: weight}.
    Returns [{"id", "score", <signal>: normalised score, ...}] best first."""
    scaled = {name: normalise({i: signals.get(name, {}).get(i, 0.0) for i in items})
              for name in weights}
    present = {name: w for name, w in weights.items() if any(scaled[name].values())}
    total = sum(present.values()) or 1.0
    ranked = []
    for i in items:
        row = {"id": i}
        for name in weights:
            row[name] = scaled[name].get(i, 0.0)
        row["score"] = sum(w * scaled[name][i] for name, w in present.items()) / total
        ranked.append(row)
    ranked.sort(key=lambda r: (-r["score"], r["id"]))
    return ranked
