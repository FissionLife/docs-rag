"""RAPTOR-lite: a small hierarchical index on top of the flat chunks.

Real RAPTOR clusters chunk embeddings, asks an LLM to summarise each cluster,
embeds the summaries, and recurses to build a multi-level tree. This is the
same shape one level deep, with one deliberate simplification: cluster
summaries are built by **template, not by an LLM call**.

That mirrors the corpus-wide overview in `overview.py` -- zero generation
cost, deterministic, and immune to a mid-demo API failure or quota wall --
at the cost of a summary that *lists* what a cluster contains rather than
synthesising a new sentence about it. Swap `_summarise_cluster` for a
`generate()` call if you want abstraction over extraction and can afford
roughly one API call per cluster (~20-30 on this corpus).

Retrieval needs **no separate code path** for this. Level-1 summary chunks
are appended to the same flat list of chunks that `ingest()` embeds and
stores, tagged only by a `_cluster_NNN` doc_id -- so `retrieve.py`'s ordinary
hybrid search already treats them as retrievable documents, competing in the
same ranking as leaf chunks. This is RAPTOR's "collapsed tree" retrieval
strategy: every node at every level is a candidate, rather than a level-by-
level tree walk that would need new retrieval code entirely.
"""
import numpy as np

from .config import CLUSTER_TARGET_SIZE

CLUSTER_PREFIX = "_cluster_"


def is_summary(doc_id: str) -> bool:
    return doc_id.startswith(CLUSTER_PREFIX)


def _kmeans(vectors: np.ndarray, k: int, iters: int = 25,
           seed: int = 0) -> np.ndarray:
    """Minimal k-means over unit vectors. Returns a (n,) int array of labels.

    Hand-rolled rather than adding scikit-learn: a few hundred rows converges
    in well under a second, and this keeps the core dependency list unchanged
    for anyone not opting into the hierarchical demo. On unit vectors, cosine
    similarity is a plain dot product, so "nearest centroid" is just argmax.
    """
    rng = np.random.default_rng(seed)
    n = len(vectors)
    k = max(1, min(k, n))
    centroids = vectors[rng.choice(n, size=k, replace=False)].copy()
    labels = np.full(n, -1)

    for i in range(iters):
        sims = vectors @ centroids.T
        new_labels = sims.argmax(axis=1)
        if i > 0 and np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for c in range(k):
            members = vectors[labels == c]
            if len(members) == 0:
                continue          # an empty cluster just keeps its centroid
            centroid = members.mean(axis=0)
            norm = np.linalg.norm(centroid)
            centroids[c] = centroid / norm if norm > 1e-12 else centroid

    return labels


def _summarise_cluster(members: list[dict]) -> str:
    """Extractive, template-built summary: what this cluster actually holds.

    One line per distinct document, using the longest member chunk's first
    sentence as a lead. Grounded entirely in retrieved text, same principle
    as the corpus-wide manifest in overview.py, just scoped to one cluster.

    "Longest chunk" is a deliberate, corpus-agnostic choice, not "the first
    one encountered": a document's first chunk is sometimes a short metadata
    or breadcrumb header rather than real prose (this repo's own AWS pages
    open with a `> tagline **Category:** ... **Source:** ...` front-matter
    block before the actual "Overview" text), and the naive first-chunk
    version was verified to grab a garbled run-on from that header instead of
    a real sentence. The longer chunk for a given title is reliably the one
    that is actually about the subject.
    """
    best: dict[str, str] = {}
    for m in members:
        prev = best.get(m["title"])
        if prev is None or len(m["text"]) > len(prev):
            best[m["title"]] = m["text"]

    lines = [f"This group covers {len(best)} related items:"]
    for title in sorted(best):
        lead = best[title].split(". ", 1)[0].strip().rstrip(".")
        lines.append(f"- {title}: {lead}.")
    return "\n".join(lines)


def build_summary_chunks(leaf_chunks: list[dict],
                         leaf_vectors: np.ndarray) -> list[dict]:
    """One level-1 summary chunk per cluster of at least two leaf chunks.

    Chunk shape matches chunk.py's exactly, plus a doc_id that self-identifies
    as a summary node (see is_summary()) rather than a schema change -- so
    store.py, retrieve.py and generate.py need no changes at all to carry
    these alongside ordinary leaf chunks.
    """
    n = len(leaf_chunks)
    if n < 4:
        return []

    k = max(2, round(n / CLUSTER_TARGET_SIZE))
    labels = _kmeans(leaf_vectors, k)

    summaries = []
    for c in sorted(set(labels.tolist())):
        idx = [i for i in range(n) if labels[i] == c]
        members = [leaf_chunks[i] for i in idx]
        titles = sorted({m["title"] for m in members})
        if len(titles) < 2:
            continue           # nothing to summarise above a single document

        text = _summarise_cluster(members)
        title = (f"Cluster summary ({len(titles)} related items)")
        summaries.append({
            "chunk_id": f"{CLUSTER_PREFIX}{c:03d}#0",
            "doc_id": f"{CLUSTER_PREFIX}{c:03d}",
            "title": title,
            "url": "",
            "section": f"{len(members)} leaf chunks, {len(titles)} documents",
            "text": text,
            "embed_text": f"{title}\n\n{text}",
        })
    return summaries
