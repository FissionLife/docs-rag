"""Section-aware sentence-packing chunker.

Three ideas do most of the work here:

1. Respect structure. Documents arrive carrying headings -- Markdown "##"
   from .md files and extracted web articles, "== ==" from Wikipedia, and
   synthesised "## Page N" markers from PDFs -- and a chunk never straddles
   two sections.
2. Pack whole sentences. Splitting mid-sentence destroys the meaning the
   embedding is supposed to capture.
3. Prepend a breadcrumb. The text we embed starts with
   "Article > Section > Subsection", which gives an otherwise anonymous
   paragraph the context it needs to be retrievable. The breadcrumb is
   part of the embedded text but is shown to the reader as a citation
   header rather than as body text.
"""
import re

from .config import CHUNK_CHARS, CHUNK_OVERLAP, MIN_CHUNK_CHARS

# Two heading dialects reach us: Markdown ATX ("## Section", from .md files,
# extracted web articles and the PDF loader's page markers) and MediaWiki
# ("== Section =="). A document uses one or the other; supporting both means
# no loader has to rewrite its text just to be chunkable.
_ATX = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$")
_WIKI = re.compile(r"^\s*(={2,})\s*(.+?)\s*\1\s*$")
_FENCE = re.compile(r"^\s{0,3}(```|~~~)")


def _heading(line: str) -> tuple[int, str] | None:
    """Return (level, text) for a heading line, else None."""
    m = _ATX.match(line)
    if m:
        return len(m.group(1)), m.group(2).strip()
    m = _WIKI.match(line)
    if m:
        return len(m.group(1)), m.group(2).strip()
    return None


# Split after . ! ? (optionally followed by a closing quote/bracket) when the
# next token starts a new sentence. Written as two fixed-width lookbehinds
# because Python's re rejects variable-width ones.
_SENT = re.compile(
    r"""(?:(?<=[.!?])|(?<=[.!?]["')\]]))\s+(?=["'(\[]?[A-Z0-9])""")
_ABBREV = re.compile(r"\b(?:e\.g|i\.e|cf|vs|etc|approx|Dr|Fig|no|al)\.$", re.I)


def split_sentences(text: str) -> list[str]:
    parts, buf = [], ""
    for piece in _SENT.split(text):
        cand = (buf + " " + piece).strip() if buf else piece
        # Re-join if we split on an abbreviation or a lone initial ("J. Smith").
        if _ABBREV.search(buf) or re.search(r"\b[A-Z]\.$", buf):
            buf = cand
            continue
        if buf:
            parts.append(buf)
        buf = piece
    if buf:
        parts.append(buf)
    return [p for p in (s.strip() for s in parts) if p]


def _sections(text: str) -> list[tuple[list[str], str]]:
    """Yield (heading_path, body) pairs. heading_path is [] for the lead.

    Nesting is tracked with a stack rather than by indexing into the path by
    level. Levels are not dense -- a document may jump from "#" to "###", or
    use "##" for every section with no "#" at all (the PDF page markers do
    exactly that) -- and index-based nesting silently turns consecutive
    same-level headings into ancestors of each other.
    """
    out, buf = [], []
    stack: list[tuple[int, str]] = []       # (level, heading text)
    in_fence = False

    def flush():
        body = "\n".join(buf).strip()
        if body:
            out.append(([name for _, name in stack], body))
        buf.clear()

    for line in text.split("\n"):
        # Never read headings inside a fenced code block: "# comment" is code,
        # and a Python comment is not a section of the document.
        if _FENCE.match(line):
            in_fence = not in_fence
            buf.append(line)
            continue

        h = None if in_fence else _heading(line)
        if h:
            flush()
            level, name = h
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, name))
        else:
            buf.append(line)
    flush()
    return out


def _pack(sentences: list[str]) -> list[str]:
    """Greedily fill chunks to CHUNK_CHARS, then back up for the overlap."""
    chunks, cur, size = [], [], 0
    for s in sentences:
        if cur and size + len(s) + 1 > CHUNK_CHARS:
            chunks.append(" ".join(cur))
            # Carry the trailing sentences that fit in the overlap budget.
            tail, tail_len = [], 0
            for prev in reversed(cur):
                if tail_len + len(prev) > CHUNK_OVERLAP:
                    break
                tail.insert(0, prev)
                tail_len += len(prev) + 1
            cur, size = tail, tail_len
        cur.append(s)
        size += len(s) + 1
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def chunk_doc(doc: dict) -> list[dict]:
    out = []
    dropped: list[tuple[list[str], str]] = []
    title_norm = doc["title"].strip().lower()
    for path, body in _sections(doc["text"]):
        # A Markdown file usually opens with "# <its own title>". Keeping it
        # would render every breadcrumb as "Runbook > Runbook > Rotation".
        if path and path[0].strip().lower() == title_norm:
            path = path[1:]
        section = " > ".join(path)
        breadcrumb = " > ".join([doc["title"]] + path)
        for text in _pack(split_sentences(re.sub(r"\s+", " ", body))):
            if len(text) < MIN_CHUNK_CHARS:
                # MIN_CHUNK_CHARS exists to discard trailing scraps, not to
                # discard documents. A short note or stub page is still worth
                # indexing, so remember it in case nothing else survives.
                dropped.append((path, text))
                continue
            out.append({
                "chunk_id": f"{doc['doc_id']}#{len(out)}",
                "doc_id": doc["doc_id"],
                "title": doc["title"],
                "url": doc["url"],
                "section": section,
                "text": text,
                "embed_text": f"{breadcrumb}\n\n{text}",
            })

    if not out and dropped:
        # Every chunk was below the floor, i.e. the whole document is short.
        # Index it anyway rather than letting it disappear without a word.
        path, text = max(dropped, key=lambda d: len(d[1]))
        section = " > ".join(path)
        breadcrumb = " > ".join([doc["title"]] + path)
        out.append({
            "chunk_id": f"{doc['doc_id']}#0",
            "doc_id": doc["doc_id"],
            "title": doc["title"],
            "url": doc["url"],
            "section": section,
            "text": text,
            "embed_text": f"{breadcrumb}\n\n{text}",
        })
    return out


def chunk_all(docs: list[dict]) -> list[dict]:
    return [c for d in docs for c in chunk_doc(d)]
