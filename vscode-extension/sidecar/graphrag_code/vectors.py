"""
Where symbol embeddings live and how the nearest ones are found.

LanceDBVectors (default when lancedb is installed and loads): an embedded vector database in
the extension's storage folder, one table per embedding model (so changing the model never
mixes vector sizes). Search runs in LanceDB; past INDEX_AT rows it builds an approximate
(IVF-PQ) index so search stays fast on large codebases.

SQLiteVectors (fallback): the vectors in the index's SQLite file, searched in memory (numpy if
available, else plain Python).

Both keep a content hash per symbol, so re-indexing embeds only symbols whose text changed.
"""
import hashlib
import logging
import os
import struct
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

try:
    import numpy as _np
except Exception:  # pragma: no cover - depends on the environment
    _np = None

try:
    import lancedb
    import pyarrow as pa
    LANCE_ERROR = ""
except Exception as e:  # not installed, or a compiled module was blocked
    lancedb = None
    LANCE_ERROR = f"{type(e).__name__}: {e}"

INDEX_AT = 20_000  # rows before LanceDB gets an approximate index
DELETE_BATCH = 400

# (symbol id, path, kind, content hash, vector)
Row = Tuple[str, str, str, str, List[float]]


class SQLiteVectors:
    name = "sqlite"

    def __init__(self, db, embedder_name: str):
        self.db, self.embedder = db, embedder_name
        db.execute("CREATE TABLE IF NOT EXISTS embeddings (symbol_id TEXT PRIMARY KEY, embedder TEXT, hash TEXT, vec BLOB)")
        self._ids: List[str] = []
        self._matrix = None

    def hashes(self) -> Dict[str, str]:
        return {r[0]: r[1] for r in self.db.execute(
            "SELECT symbol_id, hash FROM embeddings WHERE embedder = ?", (self.embedder,))}

    def upsert(self, rows: Sequence[Row]) -> None:
        for sid, _, _, h, vec in rows:
            self.db.execute("INSERT OR REPLACE INTO embeddings VALUES (?, ?, ?, ?)",
                            (sid, self.embedder, h, struct.pack(f"{len(vec)}f", *vec)))

    def delete(self, ids: Iterable[str]) -> None:
        for sid in ids:
            self.db.execute("DELETE FROM embeddings WHERE symbol_id = ?", (sid,))

    def finish(self) -> None:
        self.db.execute("DELETE FROM embeddings WHERE embedder != ?", (self.embedder,))
        self.db.commit()
        rows = list(self.db.execute("SELECT symbol_id, vec FROM embeddings"))
        self._ids = [r[0] for r in rows]
        vectors = [list(struct.unpack(f"{len(r[1]) // 4}f", r[1])) for r in rows]
        self._matrix = _np.asarray(vectors, dtype="float32") if (_np is not None and vectors) else vectors

    def search(self, query: List[float], k: int) -> Dict[str, float]:
        if not self._ids:
            return {}
        if _np is not None:
            sims = self._matrix @ _np.asarray(query, dtype="float32")
            order = _np.argsort(-sims)[:k]
            return {self._ids[i]: float(sims[i]) for i in order if sims[i] > 0}
        scored = sorted(((sum(a * b for a, b in zip(row, query)), i) for i, row in enumerate(self._matrix)), reverse=True)
        return {self._ids[i]: s for s, i in scored[:k] if s > 0}

    def close(self) -> None:
        pass


class LanceDBVectors:
    name = "lancedb"

    def __init__(self, folder: str, embedder_name: str, dim: int):
        os.makedirs(folder, exist_ok=True)
        self.db = lancedb.connect(folder)
        self.dim = dim
        # One table per model: a different model means different vector sizes
        self.table_name = "symbols_" + hashlib.sha1(f"{embedder_name}:{dim}".encode()).hexdigest()[:12]
        for old in self._table_names():
            if old.startswith("symbols_") and old != self.table_name:
                self.db.drop_table(old)
        schema = pa.schema([
            pa.field("id", pa.string()), pa.field("path", pa.string()), pa.field("kind", pa.string()),
            pa.field("hash", pa.string()), pa.field("vector", pa.list_(pa.float32(), dim)),
        ])
        if self.table_name in self._table_names():
            self.table = self.db.open_table(self.table_name)
        else:
            self.table = self.db.create_table(self.table_name, schema=schema)

    def _table_names(self) -> List[str]:
        try:
            listed = self.db.list_tables()
            names = getattr(listed, "tables", listed)  # newer lancedb returns a response object
        except AttributeError:
            names = self.db.table_names()
        return list(names)

    def hashes(self) -> Dict[str, str]:
        data = self.table.to_arrow().select(["id", "hash"]).to_pydict()
        return dict(zip(data["id"], data["hash"]))

    def delete(self, ids: Iterable[str]) -> None:
        ids = list(ids)
        for i in range(0, len(ids), DELETE_BATCH):
            quoted = ", ".join("'" + s.replace("'", "''") + "'" for s in ids[i:i + DELETE_BATCH])
            self.table.delete(f"id IN ({quoted})")

    def upsert(self, rows: Sequence[Row]) -> None:
        if not rows:
            return
        self.delete([r[0] for r in rows])
        self.table.add([{"id": sid, "path": path, "kind": kind, "hash": h, "vector": vec}
                        for sid, path, kind, h, vec in rows])

    def finish(self) -> None:
        try:
            rows = self.table.count_rows()
            if rows >= INDEX_AT and not self.table.list_indices():
                self.table.create_index(metric="cosine", vector_column_name="vector")
            if hasattr(self.table, "optimize"):
                self.table.optimize()  # merges the small fragments that incremental updates leave
        except Exception as e:  # an index is an optimisation: search works without it
            logger.warning("LanceDB maintenance skipped: %s", e)

    def search(self, query: List[float], k: int) -> Dict[str, float]:
        if not self.table.count_rows():
            return {}
        q = self.table.search(query, vector_column_name="vector")
        q = q.distance_type("cosine") if hasattr(q, "distance_type") else q.metric("cosine")
        hits = q.limit(k).select(["id", "_distance"]).to_list()
        # cosine distance = 1 - similarity
        return {h["id"]: 1.0 - float(h["_distance"]) for h in hits if 1.0 - float(h["_distance"]) > 0}

    def close(self) -> None:
        pass


def make_vector_store(spec: str, folder: str, sqlite_db, embedder) -> Tuple[object, str]:
    """('auto' | 'lancedb' | 'sqlite') -> (store, why a fallback was used or '')."""
    if spec in ("auto", "lancedb", "") and lancedb is not None:
        try:
            return LanceDBVectors(os.path.join(folder, "lancedb"), embedder.name, embedder.dim), ""
        except Exception as e:
            logger.warning("LanceDB unavailable (%s); using SQLite", e)
            return SQLiteVectors(sqlite_db, embedder.name), f"{type(e).__name__}: {e}"
    reason = "" if spec == "sqlite" else (LANCE_ERROR or "lancedb is not installed")
    return SQLiteVectors(sqlite_db, embedder.name), reason
