"""Chroma as a side-by-side comparison, not a replacement.

`store.py` already established -- with reasoning, and with a measured
"exact search takes well under a millisecond at this size" -- that a real
vector database is the wrong tool for this corpus. This module exists to make
that claim checkable rather than merely asserted: `uv run compare-store
"question"` builds an actual Chroma collection from the exact same vectors
the native store already has, and shows what each one returns for the same
query, side by side, with latency.

No re-embedding happens here. The comparison is about two ways of *storing
and querying* one set of vectors, not two ways of producing them -- so both
sides see identical numbers, and any difference in results is purely a
consequence of exact search vs. Chroma's approximate nearest-neighbor index.

Needs the optional `chroma` extra (`uv sync --extra chroma`); the core
pipeline never imports this module.
"""
import time

import numpy as np

from .embed import embed_query
from .store import Store


class ChromaNotInstalled(RuntimeError):
    pass


def _collection(store: Store):
    try:
        import chromadb
    except ImportError:
        raise ChromaNotInstalled(
            "chromadb is not installed. It is an optional extra, kept out "
            "of the default install on purpose (see pyproject.toml):\n"
            "  uv sync --extra chroma"
        ) from None

    client = chromadb.PersistentClient(path=str(store.dir / "chroma"))
    name = f"{store.model.replace('.', '_')}-{store.dim}d"
    return client.get_or_create_collection(
        name, metadata={"hnsw:space": "cosine"})


def build(store: Store) -> tuple[object, list[dict], np.ndarray]:
    """Populate (or reuse) a Chroma collection from the store's own vectors."""
    chunks, vectors = store.load()
    collection = _collection(store)

    if collection.count() != len(chunks):
        collection.upsert(
            ids=[c["chunk_id"] for c in chunks],
            embeddings=vectors.tolist(),
            documents=[c["text"] for c in chunks],
            metadatas=[{"title": c["title"], "doc_id": c["doc_id"],
                       "section": c["section"], "url": c["url"]}
                      for c in chunks],
        )
    return collection, chunks, vectors


def compare(question: str, top_k: int = 6) -> dict:
    """Native exact cosine vs. Chroma's approximate search, same vectors."""
    store = Store()
    collection, chunks, vectors = build(store)
    qv = embed_query(question)

    t0 = time.time()
    cosine = vectors @ qv
    native_ids = np.argsort(-cosine)[:top_k]
    native_ms = (time.time() - t0) * 1000
    native = [{"title": chunks[i]["title"], "section": chunks[i]["section"],
              "score": float(cosine[i])} for i in native_ids]

    t0 = time.time()
    r = collection.query(query_embeddings=[qv.tolist()], n_results=top_k)
    chroma_ms = (time.time() - t0) * 1000
    # Chroma's "cosine" space returns a distance, 1 - cosine_similarity, so
    # this converts back to the same similarity scale the native score uses.
    chroma = [{"title": m["title"], "section": m["section"], "score": 1 - d}
             for m, d in zip(r["metadatas"][0], r["distances"][0])]

    return {"native": native, "native_ms": native_ms,
            "chroma": chroma, "chroma_ms": chroma_ms,
            "n_vectors": len(chunks), "dir": str(store.dir / "chroma")}
