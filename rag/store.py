"""Persistent vector store: SQLite for text/metadata, .npy for the matrix.

Why not a vector database? With a corpus this size the whole matrix is a few
megabytes, and an exact brute-force dot product over unit vectors is both
faster and more accurate than any approximate index. numpy does ~1e5 x 1536
in single-digit milliseconds. Reach for FAISS / pgvector / Qdrant when the
matrix stops fitting in RAM or you need concurrent writers -- not before.

Each (model, dimension) pair gets its own index directory, so you can build
several and compare them without rebuilding.
"""
import json
import sqlite3

import numpy as np

from .config import EMBED_DIM, EMBED_MODEL, INDEX_DIR

SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    idx      INTEGER PRIMARY KEY,   -- row number in vectors.npy
    chunk_id TEXT UNIQUE NOT NULL,
    doc_id   TEXT NOT NULL,
    title    TEXT NOT NULL,
    url      TEXT NOT NULL,
    section  TEXT NOT NULL,
    text     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_doc ON chunks(doc_id);
"""


def index_dir(model: str = EMBED_MODEL, dim: int = EMBED_DIM):
    return INDEX_DIR / f"{model}-{dim}d"


class Store:
    def __init__(self, model: str = EMBED_MODEL, dim: int = EMBED_DIM):
        self.model, self.dim = model, dim
        self.dir = index_dir(model, dim)
        self.db_path = self.dir / "chunks.db"
        self.vec_path = self.dir / "vectors.npy"
        self.meta_path = self.dir / "meta.json"
        # Survives re-ingests; see embed.py for why it is content-addressed.
        self.vector_cache = self.dir / "vector_cache.npz"

    # --- write ---------------------------------------------------------
    def write(self, chunks: list[dict], vectors: np.ndarray) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunk/vector count mismatch")
        if vectors.shape[1] != self.dim:
            raise ValueError(
                f"vectors are {vectors.shape[1]}d, store expects {self.dim}d")

        self.dir.mkdir(parents=True, exist_ok=True)
        if self.db_path.exists():
            self.db_path.unlink()

        con = sqlite3.connect(self.db_path)
        con.executescript(SCHEMA)
        con.executemany(
            "INSERT INTO chunks (idx, chunk_id, doc_id, title, url, section,"
            " text) VALUES (?,?,?,?,?,?,?)",
            [(i, c["chunk_id"], c["doc_id"], c["title"], c["url"],
              c["section"], c["text"]) for i, c in enumerate(chunks)],
        )
        con.commit()
        con.close()

        np.save(self.vec_path, vectors.astype(np.float32))
        self.meta_path.write_text(json.dumps({
            "embed_model": self.model,
            "dim": self.dim,
            "chunks": len(chunks),
            "docs": len({c["doc_id"] for c in chunks}),
        }, indent=2), encoding="utf-8")

    # --- read ----------------------------------------------------------
    def exists(self) -> bool:
        return self.vec_path.exists() and self.db_path.exists()

    def load(self) -> tuple[list[dict], np.ndarray]:
        if not self.exists():
            raise SystemExit(
                f"No index at {self.dir}. Run: python cli.py ingest")
        meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
        if meta["embed_model"] != self.model or meta["dim"] != self.dim:
            raise SystemExit(
                f"Index was built with {meta['embed_model']} @ {meta['dim']}d "
                f"but config asks for {self.model} @ {self.dim}d. Re-ingest.")

        con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row
        rows = con.execute(
            "SELECT * FROM chunks ORDER BY idx").fetchall()
        con.close()

        vectors = np.load(self.vec_path)
        if len(rows) != len(vectors):
            raise SystemExit("Index is corrupt: row/vector count mismatch.")
        return [dict(r) for r in rows], vectors
