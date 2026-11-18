# Wiki snapshot

The wiki corpus is built once, offline, and shipped as a snapshot, because extracting its
knowledge graph takes hours. This folder holds:

| Path | What it is |
|---|---|
| `raw_data/` | The WikiText files (`wiki.train.txt`, `wiki.valid.txt`, `wiki.test.txt`), the build's input |
| `wiki_snapshot.py` | Exports the built wiki's Milvus collection and article texts; restores them on a new install |
| `neo4j-load.sh` | Loads the Neo4j dump into an empty `neo4j_data` volume before Neo4j first starts |
| `Dockerfile` | Image `graphrag-wiki-restore`: `wiki_snapshot.py` plus the exported vectors and texts |
| `Dockerfile.neo4j` | Image `graphrag-wiki-neo4j`: the Neo4j image plus the dump and `neo4j-load.sh` |
| `snapshot/` | Created by an export, not committed: `manifest.json`, `milvus.jsonl.gz`, `minio.jsonl.gz`, `neo4j.dump` |

The build itself is `query_processor/app/build_wiki.py`: it indexes each article the same way
as an uploaded file, then extracts its knowledge graph. See "Build the wiki snapshot" in the
root [README](../README.md) for the commands.
