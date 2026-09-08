"""Hybrid retrieval: dense vectors + BM25, fused with RRF, diversified by MMR.

Dense search understands meaning but is weak on rare literal tokens -- gene
names, drug names, numbers. BM25 is the opposite: exact-token matching with
no notion of synonyms. Running both and fusing catches what either alone
misses, which matters a lot on a corpus full of terms like "MPZ", "Nav1.6"
or "Charcot-Marie-Tooth".

Fusion is Reciprocal Rank Fusion: score = sum over retrievers of
1/(k + rank). It combines *ranks*, not scores, so it needs no calibration
between two systems whose numbers aren't comparable.

MMR then trims near-duplicates, because the top 6 dense hits are often six
overlapping windows of the same paragraph -- which wastes context and hides
the second fact an answer might need.
"""
import math
import re
from collections import Counter

import numpy as np

from .config import (CANDIDATES, DEDUP_COSINE, MIN_COSINE, MMR_LAMBDA,
                     RRF_K, TOP_K)
from .embed import embed_query

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = set(
    "a an and are as at be by for from has have in is it its of on or that "
    "the to was were will with which this these those but not can may".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower())
            if len(t) > 1 and t not in _STOP]


class BM25:
    """Okapi BM25. ~40 lines is the whole algorithm; no dependency needed."""

    def __init__(self, corpus: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(t)) for t in corpus]
        self.lens = np.array([sum(d.values()) for d in self.docs],
                             dtype=np.float32)
        self.avg_len = float(self.lens.mean()) if len(self.lens) else 0.0

        df = Counter()
        for d in self.docs:
            df.update(d.keys())
        n = len(self.docs)
        # Robertson/Sparck-Jones idf with the +1 that keeps it non-negative.
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5))
                    for t, c in df.items()}
        # term -> [(doc_index, term_frequency)], so scoring only touches
        # documents that actually contain a query term.
        self.postings: dict[str, list[tuple[int, int]]] = {}
        for i, d in enumerate(self.docs):
            for t, f in d.items():
                self.postings.setdefault(t, []).append((i, f))

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(len(self.docs), dtype=np.float32)
        for t in tokenize(query):
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i, f in self.postings[t]:
                denom = f + self.k1 * (
                    1 - self.b + self.b * self.lens[i] / self.avg_len)
                out[i] += idf * f * (self.k1 + 1) / denom
        return out


def _dedup(vectors: np.ndarray, pool: list[int],
          fused: dict[int, float], threshold: float = DEDUP_COSINE) -> list[int]:
    """Collapse near-identical chunks before ranking sees them.

    MMR discourages picking two similar chunks into the *same* top-k, but it
    still lets both compete for a slot -- a near-duplicate can win purely on
    relevance and only get diversity-penalized against a genuinely different
    second choice. A true near-duplicate carries no second fact, so the fix is
    to remove it before MMR runs, not to out-argue it during selection.

    O(pool^2) cosine comparisons, worst case ~30^2 = 900 dot products -- the
    pool is already capped at CANDIDATES, so this costs nothing measurable.
    """
    ordered = sorted(pool, key=lambda i: -fused[i])
    kept: list[int] = []
    for i in ordered:
        if any(float(vectors[i] @ vectors[j]) >= threshold for j in kept):
            continue
        kept.append(i)
    return kept


def _mmr(query_vec: np.ndarray, vectors: np.ndarray,
         candidates: list[int], k: int, lam: float) -> list[int]:
    """Maximal Marginal Relevance: pick relevant items that aren't redundant."""
    if not candidates:
        return []
    cand = list(candidates)
    rel = vectors[cand] @ query_vec
    selected = [cand.pop(int(np.argmax(rel)))]
    while cand and len(selected) < k:
        sub = vectors[cand]
        relevance = sub @ query_vec
        redundancy = (sub @ vectors[selected].T).max(axis=1)
        pick = int(np.argmax(lam * relevance - (1 - lam) * redundancy))
        selected.append(cand.pop(pick))
    return selected


class Retriever:
    def __init__(self, chunks: list[dict], vectors: np.ndarray):
        self.chunks = chunks
        self.vectors = vectors
        # BM25 indexes title+section+text so a breadcrumb term is matchable.
        self.bm25 = BM25([f"{c['title']} {c['section']} {c['text']}"
                          for c in chunks])

    def search(self, query: str, top_k: int = TOP_K) -> list[dict]:
        qv = embed_query(query)
        cosine = self.vectors @ qv                       # unit vectors -> cosine

        n = len(self.chunks)
        take = min(CANDIDATES, n)
        dense_ids = np.argsort(-cosine)[:take]
        bm25_all = self.bm25.scores(query)
        bm25_ids = [i for i in np.argsort(-bm25_all)[:take] if bm25_all[i] > 0]

        fused: dict[int, float] = {}
        for rank, i in enumerate(dense_ids):
            fused[int(i)] = fused.get(int(i), 0.0) + 1.0 / (RRF_K + rank + 1)
        for rank, i in enumerate(bm25_ids):
            fused[int(i)] = fused.get(int(i), 0.0) + 1.0 / (RRF_K + rank + 1)

        pool = sorted(fused, key=lambda i: -fused[i])[:take]
        pool = _dedup(self.vectors, pool, fused)
        chosen = _mmr(qv, self.vectors, pool, top_k, MMR_LAMBDA)
        # Present in fused-score order; MMR decided membership, not ordering.
        chosen.sort(key=lambda i: -fused[i])

        return [{
            **self.chunks[i],
            "cosine": float(cosine[i]),
            "bm25": float(bm25_all[i]),
            "rrf": fused[i],
        } for i in chosen]

    @staticmethod
    def is_relevant(hits: list[dict]) -> bool:
        """Gate: if nothing clears MIN_COSINE, the corpus has no answer."""
        return bool(hits) and max(h["cosine"] for h in hits) >= MIN_COSINE
