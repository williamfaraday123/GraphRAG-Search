# Use an LLM to look at adjacent or related chunks and decide if a logical relationship exists.

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from typing import List, Dict, Tuple
import json
import logging
from config import Config

logger = logging.getLogger(__name__)

class EdgeGenerator:
    def __init__(self):
        self.llm = ChatOpenAI(model=Config.EDGE_MODEL, temperature=0)
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """
            You are a Knowledge Graph Architect. 
            Analyze the following two text chunks from the same document.
            Determine if there is a strong SEMANTIC relationship between them.
            
            Possible Relationships:
            - "PREMISE_OF": Chunk A provides the logical basis for Chunk B.
            - "ELABORATES_ON": Chunk B adds detail to Chunk A.
            - "CONTRASTS_WITH": Chunk B contradicts or offers an alternative to Chunk A.
            - "SEQUENCE": Chunk B chronologically follows Chunk A.
            - "NONE": No significant logical link.
            
            Output ONLY a valid JSON object: {"relation": "TYPE", "confidence": 0.0-1.0, "reason": "brief explanation"}
            """),
            ("human", """
            Chunk A (ID: {id_a}):
            {content_a}
            
            Chunk B (ID: {id_b}):
            {content_b}
            """)
        ])

    def analyze_pair(self, node_a: Dict, node_b: Dict) -> Dict:
        """
        Asks LLM to define the edge between two nodes.
        Returns edge dict or None if no relation.
        """
        try:
            chain = self.prompt | self.llm
            response = chain.invoke({
                "id_a": node_a['chunk_id'],
                "content_a": node_a['content'][:500], # Truncate to save tokens
                "id_b": node_b['chunk_id'],
                "content_b": node_b['content'][:500]
            })
            
            content = response.content.strip()
            # Clean markdown code blocks if present
            if content.startswith("```json"):
                content = content[7:-3]
            elif content.startswith("```"):
                content = content[3:-3]
                
            result = json.loads(content)
            
            if result.get("relation") != "NONE" and result.get("confidence", 0) > 0.7:
                return {
                    "source": node_a['chunk_id'],
                    "target": node_b['chunk_id'],
                    "relation_type": result["relation"],
                    "weight": result["confidence"],
                    "reason": result["reason"]
                }
            return None
            
        except Exception as e:
            logger.warning(f"Failed to analyze pair {node_a['chunk_id']} - {node_b['chunk_id']}: {e}")
            return None

    def generate_edges_for_document(self, nodes: List[Dict]) -> List[Dict]:
        """
        Strategy: 
        1. Connect sequential chunks (i -> i+1) automatically (Structure).
        2. Use LLM to validate/strengthen sequential links or find non-sequential links.
        """
        edges = []
        
        # 1. Add Sequential Edges (Fast, no LLM needed for basic flow)
        for i in range(len(nodes) - 1):
            edges.append({
                "source": nodes[i]['chunk_id'],
                "target": nodes[i+1]['chunk_id'],
                "relation_type": "NEXT_CHUNK",
                "weight": 1.0,
                "reason": "Sequential document order"
            })
            
        # 2. LLM Enhancement (Sampled to save cost, e.g., every 3rd pair or sliding window)
        # For production, you might run this on all pairs if budget allows, or use embedding similarity to pick pairs
        logger.info(f"Running LLM edge enhancement on {len(nodes)} nodes...")
        
        # Example: Check non-adjacent chunks within a window of 5
        for i in range(len(nodes)):
            for j in range(i+2, min(i+6, len(nodes))):
                edge = self.analyze_pair(nodes[i], nodes[j])
                if edge:
                    edges.append(edge)
                    
        return edges