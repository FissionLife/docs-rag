"""The two operations that make up a RAG system.

  ingest()  documents -> chunks -> vectors -> disk        (offline, once)
  Rag.ask() question  -> retrieve -> gate -> generate     (online, per query)
"""
from .chunk import chunk_all
from .config import EMBED_DIM, EMBED_MODEL, GEN_MODEL, TOP_K
from .corpus import load
from .embed import embed_documents
from .generate import REFUSAL_MESSAGE, SYSTEM_OVERVIEW, answer
from .overview import build_context, is_corpus_question
from .retrieve import Retriever
from .store import Store


def ingest() -> Store:
    docs = load()
    print(f"corpus: {len(docs)} documents")

    chunks = chunk_all(docs)
    sizes = sorted(len(c["text"]) for c in chunks)
    print(f"chunks: {len(chunks)} "
          f"(median {sizes[len(sizes) // 2]} chars, max {sizes[-1]})")

    store = Store()
    print(f"embedding with {EMBED_MODEL} @ {EMBED_DIM}d ...")
    # Persistent and content-addressed: kept between runs so that re-ingesting
    # after adding one document only pays for that document's chunks.
    vectors = embed_documents([c["embed_text"] for c in chunks],
                              progress=True, cache=store.vector_cache)

    store.write(chunks, vectors)
    mb = vectors.nbytes / 1024 / 1024
    print(f"stored {vectors.shape[0]} x {vectors.shape[1]} vectors "
          f"({mb:.1f} MB) in {store.dir}")
    return store


class Rag:
    """Loads the index once; answers many questions."""

    def __init__(self):
        chunks, vectors = Store().load()

        # `remove` and `add` change the corpus but not the index. Serving a
        # document the user has deleted is worse than being noisy about it,
        # and a duplicated document silently costs context slots.
        from .corpus import documents
        indexed = {c["doc_id"] for c in chunks}
        current = {d["doc_id"] for d in documents()}
        if indexed != current:
            stale, missing = indexed - current, current - indexed
            if stale:
                print(f"  ! index still holds {len(stale)} removed "
                      f"document(s): {', '.join(sorted(stale))}")
            if missing:
                print(f"  ! {len(missing)} document(s) added but not indexed: "
                      f"{', '.join(sorted(missing))}")
            print("  ! run `uv run ingest` to resync\n")

        self.retriever = Retriever(chunks, vectors)
        self.chunks = chunks
        self.n = len(chunks)
        self._overview: list[dict] | None = None

    def overview(self) -> list[dict]:
        """Corpus-level context, built once and reused."""
        if self._overview is None:
            self._overview = build_context(self.chunks)
        return self._overview

    def ask(self, question: str, top_k: int = TOP_K) -> dict:
        # Route first. A question about the collection ("summarise this",
        # "what are the titles") has no answer in any single chunk, so
        # similarity search would return unrelated paragraphs and the system
        # would correctly -- but uselessly -- refuse. See overview.py.
        if is_corpus_question(question):
            hits = self.overview()
            result = answer(question, hits, system=SYSTEM_OVERVIEW)
            result["hits"] = hits
            result["gated"] = False
            result["route"] = "corpus-overview"
            result["best_cosine"] = None
            return result

        hits = self.retriever.search(question, top_k)

        if not Retriever.is_relevant(hits):
            # Nothing in the corpus is close enough. Refuse without paying
            # for a generation call -- and without giving the model an
            # opportunity to answer from its own knowledge.
            best = max((h["cosine"] for h in hits), default=0.0)
            return {"answer": REFUSAL_MESSAGE, "refused": True,
                    "citations": [], "sources": [], "hits": hits,
                    "gated": True, "route": "retrieval",
                    "best_cosine": round(best, 3)}

        result = answer(question, hits)
        result["hits"] = hits
        result["gated"] = False
        result["route"] = "retrieval"
        result["best_cosine"] = round(max(h["cosine"] for h in hits), 3)
        return result

    def __repr__(self) -> str:
        return (f"<Rag chunks={self.n} embed={EMBED_MODEL}@{EMBED_DIM}d "
                f"gen={GEN_MODEL}>")
