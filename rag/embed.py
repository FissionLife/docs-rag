"""Gemini embeddings, with the two supported models' quirks handled.

gemini-embedding-001
    Text only, 2048 input tokens. Accepts `task_type`, which lets documents
    and queries be embedded *asymmetrically* -- RETRIEVAL_DOCUMENT for the
    corpus, RETRIEVAL_QUERY for the question. That is a real retrieval-quality
    win and the reason this is the default. Truncated (non-3072) vectors come
    back un-normalized, so we normalize ourselves.

gemini-embedding-2
    Multimodal, 8192 input tokens, newest GA model. Has no `task_type`
    parameter -- the documented substitute is an instruction prefix. It also
    auto-renormalizes truncated vectors. One trap: passing a list of strings
    returns ONE aggregated vector for the whole list; to get one vector per
    input each must be wrapped in its own types.Content.
"""
import hashlib
from pathlib import Path

import numpy as np
from google import genai
from google.genai import types

from .backoff import RateLimiter, with_retry
from .config import (API_KEY, EMBED_BATCH, EMBED_DIM, EMBED_ITEMS_PER_MIN,
                     EMBED_MAX_RETRIES, EMBED_MODEL)

_TASKLESS = {"gemini-embedding-2"}  # models without a task_type parameter

# Used in place of task_type for models that don't support the parameter.
_PREFIX = {
    "RETRIEVAL_DOCUMENT": "Passage to be retrieved later: ",
    "RETRIEVAL_QUERY": "Search query: ",
}

_client = None
_limiter = RateLimiter(EMBED_ITEMS_PER_MIN)


def client() -> genai.Client:
    global _client
    if _client is None:
        if not API_KEY:
            raise SystemExit(
                "GEMINI_API_KEY is not set. Copy .env.example to .env and "
                "put your key in it (https://aistudio.google.com/apikey)."
            )
        _client = genai.Client(api_key=API_KEY)
    return _client


def normalize(m: np.ndarray) -> np.ndarray:
    """L2-normalize rows so a dot product is exactly cosine similarity."""
    n = np.linalg.norm(m, axis=-1, keepdims=True)
    return m / np.maximum(n, 1e-12)


def _call(texts: list[str], task: str) -> list[list[float]]:
    if EMBED_MODEL in _TASKLESS:
        payload = [
            types.Content(parts=[types.Part.from_text(
                text=_PREFIX.get(task, "") + t)])
            for t in texts
        ]
        cfg = types.EmbedContentConfig(output_dimensionality=EMBED_DIM)
    else:
        payload = texts
        cfg = types.EmbedContentConfig(
            output_dimensionality=EMBED_DIM, task_type=task)

    def once():
        _limiter.reserve(len(texts))
        r = client().models.embed_content(
            model=EMBED_MODEL, contents=payload, config=cfg)
        got = [e.values for e in r.embeddings]
        # gemini-embedding-2 silently aggregates a list of plain strings into
        # a single vector. Never trust the count -- check it.
        if len(got) != len(texts):
            raise RuntimeError(
                f"expected {len(texts)} embeddings, got {len(got)}")
        return got

    return with_retry(once, max_attempts=EMBED_MAX_RETRIES)


def _key(text: str) -> str:
    """Content address for one embedding.

    The model and dimension are part of the key because vectors from
    different models are not comparable, so they must never collide.
    """
    return hashlib.sha1(
        f"{EMBED_MODEL}|{EMBED_DIM}|{text}".encode("utf-8")).hexdigest()


def _read_cache(path: Path | None) -> dict[str, np.ndarray]:
    if path is None or not path.exists():
        return {}
    try:
        d = np.load(path, allow_pickle=False)
        return dict(zip(d["keys"].tolist(), d["vectors"]))
    except Exception:
        return {}          # a corrupt cache costs re-embedding, not a crash


def _write_cache(path: Path | None, store: dict[str, np.ndarray]) -> None:
    if path is None or not store:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path,
             keys=np.array(list(store), dtype=object).astype("U40"),
             vectors=np.asarray(list(store.values()), dtype=np.float32))


def embed(texts: list[str], task: str, progress: bool = False,
          cache: Path | None = None) -> np.ndarray:
    """Embed texts into an (n, EMBED_DIM) float32 unit-norm matrix.

    The cache is **content-addressed**, not positional: each text is keyed by
    a hash of (model, dimension, its own content). That is what makes ingest
    incremental. Add one document to a corpus of fifty and only the new
    document's chunks are sent to the API -- everything else is a dictionary
    lookup. Edit a document and only its changed chunks are re-embedded.

    This matters more than it sounds: the free tier allows 1000 embedding
    calls per day, so a pipeline that re-embeds the whole corpus on every run
    can only be run once or twice a day before it stops working.
    """
    store = _read_cache(cache)
    before = len(store)

    # Deduplicate: identical text embeds once no matter how often it appears.
    todo = list(dict.fromkeys(t for t in texts if _key(t) not in store))
    if progress:
        reused = len(texts) - len(todo)
        print(f"  {len(todo)} to embed, {reused} reused from cache")

    for i in range(0, len(todo), EMBED_BATCH):
        batch = todo[i:i + EMBED_BATCH]
        for text, vec in zip(batch, _call(batch, task)):
            store[_key(text)] = np.asarray(vec, dtype=np.float32)
        _write_cache(cache, store)      # after every batch, so a crash keeps it
        if progress:
            print(f"  embedded {min(i + EMBED_BATCH, len(todo))}/{len(todo)}")

    if progress and len(store) > before:
        print(f"  cache now holds {len(store)} vectors")
    return normalize(np.asarray([store[_key(t)] for t in texts],
                                dtype=np.float32))


def embed_documents(texts: list[str], progress: bool = False,
                    cache: Path | None = None) -> np.ndarray:
    return embed(texts, "RETRIEVAL_DOCUMENT", progress, cache)


def embed_query(text: str) -> np.ndarray:
    return embed([text], "RETRIEVAL_QUERY")[0]
