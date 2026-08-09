# Store the Nodes and Edges to Neo4j

from neo4j import GraphDatabase
from typing import List, Dict
import logging
from config import Config

logger = logging.getLogger(__name__)

class GraphBuilder:
    def __init__(self):
        self.driver = GraphDatabase.driver(Config.NEO4J_URI, auth=(Config.NEO4J_USER, Config.NEO4J_PASSWORD))

    def close(self):
        self.driver.close()

    def ingest_nodes(self, nodes: List[Dict], batch_size: int = 1000):
        """
        Creates nodes in Neo4j with label 'Chunk'.
        Uses MERGE to avoid duplicates if re-running.
        Writes in batches so a single huge document can't hold the whole
        node list in one transaction/network payload.
        """
        cypher = """
        UNWIND $nodes AS node
        MERGE (c:Chunk {chunk_id: node.chunk_id})
        SET c.content = node.content,
            c.source = node.source,
            c.chunk_index = node.chunk_index,
            c.total_chunks = node.total_chunks
        """
        with self.driver.session() as session:
            for i in range(0, len(nodes), batch_size):
                batch = nodes[i:i + batch_size]
                session.run(cypher, nodes=batch)
                logger.info(f"Ingested nodes batch {i // batch_size + 1}: {min(i + batch_size, len(nodes))}/{len(nodes)}")
        logger.info(f"Ingested {len(nodes)} nodes into Neo4j.")

    def ingest_edges(self, edges: List[Dict], batch_size: int = 1000):
        """
        Creates relationships between existing nodes.
        Writes in batches for the same reason as ingest_nodes.
        """
        cypher = """
        UNWIND $edges AS edge
        MATCH (a:Chunk {chunk_id: edge.source})
        MATCH (b:Chunk {chunk_id: edge.target})
        MERGE (a)-[r:RELATION {type: edge.relation_type}]->(b)
        SET r.weight = edge.weight,
            r.reason = edge.reason
        """
        with self.driver.session() as session:
            for i in range(0, len(edges), batch_size):
                batch = edges[i:i + batch_size]
                session.run(cypher, edges=batch)
                logger.info(f"Ingested edges batch {i // batch_size + 1}: {min(i + batch_size, len(edges))}/{len(edges)}")
        logger.info(f"Ingested {len(edges)} edges into Neo4j.")

    def create_vector_index(self):
        """
        Creates a Vector Index on Chunk.content for later retrieval by the Query Processor.
        Requires Neo4j with GenAI plugin or manual index creation depending on version.
        Here we assume standard setup or prepare for Milvus sync.
        
        NOTE: In this architecture, Milvus holds the vectors. 
        This function ensures Neo4j is ready to link to Milvus IDs.
        """
        logger.info("Neo4j schema ready. Ensure Milvus index is created separately.")