from neo4j import GraphDatabase
from typing import List, Dict, Set
import logging

logger = logging.getLogger(__name__)

class Neo4jService:
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def get_subgraph_adjacency(self, node_ids: List[str]) -> List[Dict]:
        """
        Retrieves the adjacency matrix (edge list) for a specific set of node IDs.
        This creates an 'induced subgraph' containing only the provided nodes 
        and the edges connecting them.
        
        Logic:
        MATCH (n)-[r]-(m)
        WHERE n.id IN $ids AND m.id IN $ids
        RETURN n.id, r.type, m.id
        """
        if not node_ids:
            return []

        cypher = """
        MATCH (n)-[r]-(m)
        WHERE n.chunk_id IN $node_ids AND m.chunk_id IN $node_ids
        RETURN n.chunk_id AS source, type(r) AS relation, m.chunk_id AS target, r.weight AS weight
        """
        
        # Neo4j has a limit on parameter list size usually, but for 100 items it's fine.
        # If > 1000, we might need to batch this.
        try:
            with self.driver.session() as session:
                result = session.run(cypher, node_ids=node_ids)
                edges = []
                for record in result:
                    edges.append({
                        "source": record["source"],
                        "target": record["target"],
                        "relation": record["relation"],
                        "weight": record["weight"] if record["weight"] else 1.0
                    })
                
                logger.info(f"Retrieved subgraph with {len(edges)} edges for {len(node_ids)} nodes.")
                return edges
        except Exception as e:
            logger.error(f"Neo4j subgraph extraction failed: {e}")
            return []

    # Optional: Helper to get node details if needed for context later
    def get_node_details(self, node_ids: List[str]) -> List[Dict]:
        if not node_ids:
            return []
        
        cypher = """
        MATCH (n)
        WHERE n.chunk_id IN $node_ids
        RETURN n.chunk_id AS id, n.content AS content, n.summary AS summary
        """
        
        with self.driver.session() as session:
            result = session.run(cypher, node_ids=node_ids)
            return [{"id": r["id"], "content": r["content"], "summary": r.get("summary", "")} for r in result]