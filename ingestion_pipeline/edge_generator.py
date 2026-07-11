# Use an LLM to look at adjacent or related chunks and decide if a logical relationship exists.

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from typing import List, Dict, Tuple
import json
import logging
import re
from config import Config

logger = logging.getLogger(__name__)

class EdgeGenerator:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=Config.EDGE_MODEL,
            base_url=Config.EDGE_MODEL_BASE_URL,
            api_key=Config.LLM_API_KEY
        )
        
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
            
            Output ONLY a valid JSON object: {{"relation": "TYPE", "confidence": 0.0-1.0, "reason": "brief explanation"}}
            """),
            ("human", """
            Chunk A (ID: {id_a}):
            {content_a}
            
            Chunk B (ID: {id_b}):
            {content_b}
            """)
        ])

    def _parse_llm_output(self, raw_text: str, id_a: str, id_b: str) -> dict:
        """
        Robustly extract and parse a JSON object from raw LLM output.
        Handles thinking tokens, surrounding text, smart quotes,
        unescaped braces in strings, and markdown-style output.
        """
        text = raw_text.strip()
        
        # --- Helper: clean a JSON candidate string ---
        def clean_json(s):
            # Replace smart quotes with ASCII equivalents
            s = s.replace('\u2018', "'").replace('\u2019', "'")
            s = s.replace('\u201c', '"').replace('\u201d', '"')
            # Single quotes -> double quotes
            s = s.replace("'", '"')
            # Remove trailing commas before } and ]
            s = re.sub(r',\s*\}', '}', s)
            s = re.sub(r',\s*\]', ']', s)
            # Quote unquoted keys (only after { or ,)
            s = re.sub(r'([\{,])\s*(\w+)\s*:', r'\1"\2":', s)
            # Insert missing commas between properties: "value" "key": -> "value", "key":
            s = re.sub(r'("(?:[^"\\]|\\.)*")\s+("(?:[^"\\]|\\.)*"\s*:)', r'\1,\2', s)
            return s
        
        # --- Helper: try to json.loads with cleaning ---
        def try_parse(s):
            s_clean = clean_json(s)
            try:
                return json.loads(s_clean)
            except json.JSONDecodeError:
                pass
            # Try ast.literal_eval as last resort (handles True/False/None)
            try:
                import ast
                py = s_clean.replace("true", "True").replace("false", "False").replace("null", "None")
                return ast.literal_eval(py)
            except Exception:
                return None
        
        # Strategy 1: Extract content from markdown code blocks first
        code_block = re.search(r'```(?:json)?\s*\n?(\{.*?\})\s*\n?```', text, re.DOTALL)
        if code_block:
            result = try_parse(code_block.group(1))
            if result:
                return result
        
        # Strategy 2: Find first { to last } (greedy, full JSON object)
        first_brace = text.find('{')
        last_brace = text.rfind('}')
        if first_brace != -1 and last_brace > first_brace:
            # Try the full span first
            candidate = text[first_brace:last_brace + 1]
            result = try_parse(candidate)
            if result:
                return result
            
            # Strategy 3: If full span fails (e.g. unescaped } in string),
            # try each } position from right to left to find the real closing brace
            candidate = text[first_brace:]  # everything from first {
            for pos in range(len(candidate) - 1, -1, -1):
                if candidate[pos] == '}':
                    result = try_parse(candidate[:pos + 1])
                    if result:
                        return result
        
        # Strategy 4: Look for markdown/YAML-style output like **Relation**: "TYPE"
        relation_map = {
            "PREMISE_OF": "PREMISE_OF", "ELABORATES_ON": "ELABORATES_ON",
            "CONTRASTS_WITH": "CONTRASTS_WITH", "SEQUENCE": "SEQUENCE", "NONE": "NONE"
        }
        for rel_name in relation_map:
            # Pattern: Relation: "ELABORATES_ON" or **Relation**: ELABORATES_ON
            if re.search(r'[Rr]elation[:\s]*"?'+ rel_name + r'"?' , text):
                # Extract confidence
                conf_match = re.search(r'[Cc]onfidence[:\s]*([0-9.]+)', text)
                conf = float(conf_match.group(1)) if conf_match else 0.5
                # Extract reason
                reason_match = re.search(r'[Rr]eason[:\s]*(.+?)(?:\n|$)', text)
                reason = reason_match.group(1).strip().rstrip('.') if reason_match else ""
                return {"relation": rel_name, "confidence": conf, "reason": reason}
        
        # All strategies exhausted
        logger.warning(f"Could not parse LLM output. Raw text (first 300 chars): {text[:300]}")
        return None
    
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
            
            result = self._parse_llm_output(response.content, node_a['chunk_id'], node_b['chunk_id'])
            
            if result and result.get("relation") != "NONE" and result.get("confidence", 0) > 0.5:
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
            
        # 2. LLM Enhancement — sampled to save cost and time
        # Running the LLM on every pair with thousands of nodes would be expensive.
        sample_step = max(1, len(nodes) // 20)  # Analyze ~5% of nodes
        max_llm_pairs = 20                     # Hard cap on API calls
        pairs_analyzed = 0
        logger.info(f"Running LLM edge enhancement (sampling every {sample_step}th node, max {max_llm_pairs} pairs)...")
        
        for i in range(0, len(nodes), sample_step):
            for j in range(i+2, min(i+6, len(nodes))):
                if pairs_analyzed >= max_llm_pairs:
                    break
                edge = self.analyze_pair(nodes[i], nodes[j])
                if edge:
                    edges.append(edge)
                pairs_analyzed += 1
            if pairs_analyzed >= max_llm_pairs:
                break
                    
        return edges