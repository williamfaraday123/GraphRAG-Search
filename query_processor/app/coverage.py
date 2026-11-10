"""
Graph coverage check: decides whether another retrieval round is needed without a model call.

The context is enough when every entity the question names is mentioned by a chunk in it.
For an entity that isn't, the follow-up search is the entity plus its strongest fact (the
neighbour it shares the most facts with), e.g. "Europe provided artists and technical
innovations", so vector search and the graph both start from it next round. Entities that
no chunk in the selected files mentions are ignored: another search can't reach them.
"""
import logging

logger = logging.getLogger(__name__)


class CoverageCheck:
    def __init__(self, neo4j_driver):
        self.driver = neo4j_driver

    def uncovered(self, corpus_id, file_ids, keys, chunk_ids):
        """Entities in `keys` that no chunk in `chunk_ids` mentions but some chunk in the
        selected files does. Returns [{'key', 'name'}] in the order given."""
        if not keys:
            return []
        with self.driver.session() as session:
            rows = session.run(
                """
                UNWIND $keys AS k
                MATCH (e:Entity {corpus_id: $corpus_id, key: k})
                RETURN k AS key, e.name AS name,
                       [(e)<-[:MENTIONS]-(c:Chunk) WHERE c.chunk_id IN $chunk_ids | 1] <> [] AS covered,
                       [(e)<-[:MENTIONS]-(c:Chunk) WHERE $all OR c.file_id IN $file_ids | 1] <> [] AS reachable
                """,
                keys=list(keys), corpus_id=corpus_id, chunk_ids=list(chunk_ids),
                file_ids=file_ids or [], all=not file_ids,
            ).data()
        by_key = {r["key"]: r for r in rows}
        return [{"key": k, "name": by_key[k]["name"]} for k in keys
                if k in by_key and by_key[k]["reachable"] and not by_key[k]["covered"]]

    def follow_up_query(self, corpus_id, file_ids, entity):
        """Search text for an uncovered entity: its strongest fact, else just its name."""
        with self.driver.session() as session:
            row = session.run(
                """
                MATCH (e:Entity {corpus_id: $corpus_id, key: $key})-[f:FACT]-(o:Entity)
                WHERE ($all OR f.file_id IN $file_ids) AND o <> e
                WITH o, collect(f) AS facts
                ORDER BY size(facts) DESC, o.key LIMIT 1
                WITH o, facts[0] AS f
                RETURN startNode(f).name AS source, f.relation AS relation, endNode(f).name AS target
                """,
                corpus_id=corpus_id, key=entity["key"], file_ids=file_ids or [], all=not file_ids,
            ).single()
        if row and row["relation"]:
            return f"{row['source']} {row['relation']} {row['target']}"
        return entity["name"]
