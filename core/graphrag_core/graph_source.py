"""
What a graph must offer to be searched with the shared PPR and coverage check.

The text search's graph lives in Neo4j (chunks and entities); the extension's lives in memory
(symbols and code relationships). Both answer the same two questions.
"""
from typing import Dict, Hashable, List, Mapping, Protocol


class GraphSource(Protocol):
    def neighbours(self, nodes: List[Hashable]) -> Mapping[Hashable, Mapping[Hashable, float]]:
        """{node: {neighbour: edge weight}} for the given nodes (see ppr.push_ppr)."""
        ...

    def seed_candidates(self, query: str, limit: int) -> Dict[Hashable, float]:
        """Nodes the query names directly, with a start weight each."""
        ...
