"""Corpus-level questions, and the router that spots them.

Similarity search answers questions *about a fact*. It cannot answer questions
*about the corpus* -- "summarise this", "what topics are covered", "what are
the titles" -- because no single chunk contains that answer. Embedding such a
query returns six unrelated paragraphs at indistinguishable scores (measurably:
cosine 0.57-0.59, BM25 exactly 0.00), and refusing on them is correct
behaviour, not a bug.

The fix is a second route, not a looser prompt. When a question is about the
collection rather than about a fact in it, we assemble a different context:

  block [1]      a manifest -- every document title and its section headings,
                 built from stored metadata, so counts and titles are exact
  blocks [2..N]  the lead chunk of each document

Wikipedia lead sections are human-written summaries of their articles, so the
per-document summaries cost nothing to produce and are verbatim corpus text.
Grounding is unchanged: the manifest is data we stored, the leads are real
chunks, and the same citation verification applies to both.

This is the cheap, deterministic form of a hierarchical index (RAPTOR, guide
§8.7), which builds the same detail-to-abstraction tree with an LLM summarising
each cluster at ingest time.
"""
import re

# Deliberately narrow. A router that fires too eagerly sends ordinary factual
# questions down a path that cannot answer them, which is a worse failure than
# not firing at all -- so every branch requires corpus-level vocabulary, and
# the bare "summarise" forms must be the whole question.
_META = re.compile(r"""
      ^\s*(summari[sz]e|summar(y|ies)|overview|tl;?dr|recap)\s*[.!?]?\s*$
    | \bsummari[sz]e\b[^?]{0,30}\b(corpus|collection|context|documents?|
          articles?|sources?|everything|it|this|these|them|all)\b
    | \bwhat\s+(is|are)\s+(this|these|it|the)\b[^?]{0,30}\b(about|corpus|
          collection|documents?|articles?|topics?|subjects?|context|sources?)\b
    | \bwhat\s+(topics?|subjects?|documents?|articles?|sources?|titles?)\b
          [^?]{0,20}\b(are|is|do|does|have|exist|cover|available|included?)\b
    | ^\s*what\s+(is|are)\s+the\s+titles?\s*[.!?]?\s*$
    | \bwhat\s+can\s+(you|i)\s+(answer|ask|tell|find|search)\b
    | \blist\s+((all|the|every)\s+){0,2}
          (documents?|articles?|topics?|titles?|sources?)\b
    | \btable\s+of\s+contents\b
    | \bhow\s+many\s+(documents?|articles?|chunks?|sources?|topics?)\b
    | \b(corpus|knowledge\s*base|index|collection)\b[^?]{0,25}
          \b(contain|cover|about|include|have|consist)\b
    | \b(give|show|provide)\s+(me\s+)?(an?\s+)?
          (overview|summary|recap|rundown|synopsis)\b
    | \bwhat(?:'s|\s+is|\s+are)\s+(in|inside)\s+
          (here|this|it|the\s+(corpus|collection|index|dataset))\b
    | \btell\s+me\s+about\s+(this|the|your)\s+
          (collection|corpus|dataset|documents?|index|knowledge\s*base)\b
    """, re.I | re.X)


def is_corpus_question(q: str) -> bool:
    return bool(_META.search(q))


def _manifest(chunks: list[dict]) -> str:
    """Exact titles, counts and section headings, straight from metadata."""
    docs: dict[str, dict] = {}
    for c in chunks:
        d = docs.setdefault(c["doc_id"], {
            "title": c["title"], "url": c["url"], "sections": [], "n": 0})
        d["n"] += 1
        s = c["section"].split(" > ")[0]
        if s and s not in d["sections"]:
            d["sections"].append(s)

    lines = [f"This collection contains {len(docs)} documents "
             f"({len(chunks)} indexed passages in total). "
             f"The documents and their sections are:", ""]
    for d in sorted(docs.values(), key=lambda x: x["title"]):
        secs = ", ".join(d["sections"][:12]) or "(no subsections)"
        lines.append(f"- {d['title']} ({d['n']} passages) - sections: {secs}")
    return "\n".join(lines)


def build_context(chunks: list[dict]) -> list[dict]:
    """Hit-shaped blocks: the manifest, then each document's lead chunk.

    Returned dicts carry the same keys as retrieval hits so that generation,
    citation checking and the CLI need no special cases.
    """
    manifest = {
        "chunk_id": "corpus#manifest",
        "doc_id": "corpus",
        "title": "Corpus manifest",
        "url": "",
        "section": "",
        "text": _manifest(chunks),
        "cosine": 1.0, "bm25": 0.0, "rrf": 0.0,
    }

    # The first chunk of each document is its lead section -- on Wikipedia,
    # a summary of the whole article written by a human.
    leads, seen = [], set()
    for c in chunks:
        if c["doc_id"] in seen or c["section"]:
            continue
        seen.add(c["doc_id"])
        leads.append({**c, "cosine": 1.0, "bm25": 0.0, "rrf": 0.0})
    leads.sort(key=lambda c: c["title"])

    return [manifest] + leads
