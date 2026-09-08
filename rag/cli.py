"""Command line entry points.

Each function here is exposed as `uv run <name>` via [project.scripts] in
pyproject.toml. The workflow is three steps, and only the first two touch
your documents:

    uv run add <path|url>   1. PARSE a source and put it in the corpus
    uv run ingest           2. CHUNK it, EMBED it, and SAVE the index
    uv run dev              3. ASK questions about it

Everything else is either inspection (`list`, `inspect`, `status`) or
maintenance (`clear`, `eval`). Full descriptions in `main()` below, or run
`uv run help`.
"""
import shutil
import sys

from .config import CORPUS_DIR, EMBED_DIM, EMBED_MODEL, GEN_MODEL, INDEX_DIR


def _args(usage: str) -> list[str]:
    rest = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not rest:
        raise SystemExit(usage)
    return rest


def show_chunks(r: dict) -> None:
    """Print every retrieved chunk, marking which ones the answer cited.

    The uncited ones are the interesting half. They show what the retriever
    thought was relevant and the generator then ignored -- which is how you
    tell "retrieval missed it" from "retrieval found it and the model didn't
    use it". A refusal with six on-topic chunks is a very different bug from a
    refusal with six irrelevant ones.
    """
    hits = r.get("hits") or []
    if not hits:
        print("  (no chunks retrieved)\n")
        return

    cited = set(r.get("citations") or [])
    overview = r.get("route") == "corpus-overview"
    print(f"  ── retrieved context ({len(hits)} blocks"
          + (", overview route" if overview else "") + ") "
          + "─" * 28)

    for i, h in enumerate(hits, 1):
        # The overview route supplies every document's lead section, so
        # printing all of them buries the answer. Show what was used.
        if overview and i not in cited:
            continue
        mark = "CITED " if i in cited else "unused"
        head = h["title"] + (f" > {h['section']}" if h["section"] else "")
        scores = ("" if overview else
                  f"  cos {h['cosine']:.3f}  bm25 {h['bm25']:.2f}"
                  f"  rrf {h['rrf']:.4f}")
        print(f"\n  [{i}] {mark}  {head}{scores}")
        print(f"      {h['chunk_id']}")
        for line in h["text"].splitlines() or [""]:
            print(f"      {line}")

    if overview:
        skipped = len(hits) - len(cited)
        if skipped > 0:
            print(f"\n  ({skipped} further overview blocks not cited, hidden)")
    print()


def show(q: str, r: dict, chunks: bool = False) -> None:
    print(f"\nQ: {q}")
    print(f"\n{r['answer']}\n")
    if r["refused"]:
        why = ("no chunk cleared the relevance gate"
               if r.get("gated") else r.get("note", "model declined"))
        cos = r.get("best_cosine")
        print(f"  [refused: {why}"
              + (f"; best cosine {cos}" if cos is not None else "") + "]")
    else:
        overview = r.get("route") == "corpus-overview"
        if overview:
            print("  (answered from the corpus overview, not similarity search)")
        print("  sources:")
        for s in r["sources"]:
            loc = s["title"] + (f" > {s['section']}" if s["section"] else "")
            # Overview blocks carry no real similarity score; showing the
            # placeholder 1.0 would imply a measurement that never happened.
            score = "" if overview else f"  (cos {s['cosine']})"
            print(f"    [{s['n']}] {loc}{score}")
            if s["url"]:
                print(f"        {s['url']}")
        if r.get("note"):
            print(f"  note: {r['note']}")
    if chunks:
        print()
        show_chunks(r)


# ---------------------------------------------------------------------------
# 1. Building the corpus
# ---------------------------------------------------------------------------

def add() -> None:
    """Parse one or more sources into the corpus.

        uv run add https://en.wikipedia.org/wiki/CRISPR
        uv run add https://medium.com/@someone/an-article
        uv run add ./paper.pdf ./notes.md ./transcript.txt
        uv run add ./research/            (a folder, searched recursively)
        uv run add ./paper.pdf --replace  (overwrite if already present)

    Accepts .pdf .md .txt .html .rst .csv and any http(s) URL. Parsing happens
    now and the extracted text is stored, so `ingest` never re-reads a PDF.
    Nothing is embedded at this stage -- no API calls, no cost.
    """
    from .corpus import add as run
    sources = _args('usage: uv run add <file|folder|url> [more...] [--replace]')
    docs = run(sources, replace="--replace" in sys.argv)
    if docs:
        chars = sum(len(d["text"]) for d in docs)
        print(f"\n{len(docs)} document(s) added, {chars:,} characters.")
        print("Next:  uv run ingest")
    else:
        print("\nNothing added.")


def ingest() -> None:
    """Chunk the corpus, embed the chunks, and write the searchable index.

        uv run ingest

    Safe and cheap to re-run: embeddings are cached by content, so only new or
    edited chunks cost an API call. Run it after every `add` or `remove`.
    """
    from .pipeline import ingest as run
    run()


def clear() -> None:
    """Delete the corpus and/or the index.

        uv run clear --corpus   remove the documents you added
        uv run clear --index    remove embeddings and the vector cache
        uv run clear --all      both

    Requires one of those flags: this deletes data, so it should not be
    possible to do by accident.
    """
    want_corpus = "--corpus" in sys.argv or "--all" in sys.argv
    want_index = "--index" in sys.argv or "--all" in sys.argv
    if not (want_corpus or want_index):
        raise SystemExit(
            "Refusing to delete anything without an explicit target.\n"
            "  uv run clear --corpus   the documents you added\n"
            "  uv run clear --index    embeddings and vector cache\n"
            "  uv run clear --all      both")

    for label, path, on in (("corpus", CORPUS_DIR, want_corpus),
                            ("index", INDEX_DIR, want_index)):
        if not on:
            continue
        if path.exists():
            n = len(list(path.rglob("*")))
            shutil.rmtree(path)
            print(f"removed {label}: {path} ({n} files)")
        else:
            print(f"{label} was already empty")


def remove() -> None:
    """Remove one document from the corpus by its id (see `uv run list`).

        uv run remove crispr-a1b2c3

    The index still contains it until you re-run `uv run ingest`.
    """
    from .corpus import remove as run
    ids = _args("usage: uv run remove <doc_id>   (get ids from `uv run list`)")
    for doc_id in ids:
        print(("removed " if run(doc_id) else "not found ") + doc_id)
    print("\nRe-run `uv run ingest` to update the index.")


# ---------------------------------------------------------------------------
# 2. Looking at what you have
# ---------------------------------------------------------------------------

def list_docs() -> None:
    """List every document in the corpus, with its id, type and size.

        uv run list
    """
    from .corpus import documents
    docs = documents()
    if not docs:
        print("Corpus is empty. Add something with `uv run add <file|url>`.")
        return
    print(f"{len(docs)} document(s) in {CORPUS_DIR}:\n")
    for d in sorted(docs, key=lambda x: x["title"].lower()):
        print(f"  {d['title']}")
        print(f"    id={d['doc_id']}  kind={d['kind']}  "
              f"{len(d['text']):,} chars")
        print(f"    {d['url']}")
    print(f"\ntotal {sum(len(d['text']) for d in docs):,} characters")


def status() -> None:
    """Show whether the corpus and index are in sync, and how they are configured.

        uv run status
    """
    import json

    from .corpus import documents
    from .store import Store

    docs = documents()
    st = Store()
    print(f"embedding model  {EMBED_MODEL} @ {EMBED_DIM}d")
    print(f"generation model {GEN_MODEL}")
    print(f"corpus           {len(docs)} document(s), "
          f"{sum(len(d['text']) for d in docs):,} chars")
    if st.exists():
        meta = json.loads(st.meta_path.read_text(encoding="utf-8"))
        print(f"index            {meta['chunks']} chunks from "
              f"{meta['docs']} document(s)")
        print(f"                 {st.dir}")
        if meta["docs"] != len(docs):
            print("  ** corpus and index disagree -- run `uv run ingest` **")
    else:
        print("index            none yet -- run `uv run ingest`")
    if st.vector_cache.exists():
        mb = st.vector_cache.stat().st_size / 1024 / 1024
        print(f"vector cache     {mb:.1f} MB (re-ingest is nearly free)")


def inspect() -> None:
    """Show what retrieval returns for a question, with no generation step.

        uv run inspect "how does X work"

    The most useful debugging view in the system: it separates "the right
    passage was never retrieved" from "the model had it and answered badly".
    Costs one embedding call, no generation.
    """
    from .pipeline import Rag
    q = " ".join(_args('usage: uv run inspect "your question"'))
    for h in Rag().retriever.search(q):
        head = h["title"] + (f" > {h['section']}" if h["section"] else "")
        print(f"\ncos {h['cosine']:.3f}  bm25 {h['bm25']:6.2f}  "
              f"rrf {h['rrf']:.4f}  {head}  ({h['chunk_id']})")
        print("  " + h["text"][:280] + "...")


# ---------------------------------------------------------------------------
# 3. Asking questions
# ---------------------------------------------------------------------------

def ask() -> None:
    """Answer one question from the indexed documents, with citations.

        uv run ask "what does the report say about revenue?"
        uv run ask "..." --chunks    also print the retrieved passages

    Refuses rather than guessing when the documents do not contain the answer.
    `--chunks` shows every retrieved passage and marks which were cited, which
    is the fastest way to see why an answer came out the way it did.
    """
    from .pipeline import Rag
    q = " ".join(_args('usage: uv run ask "your question" [--chunks]'))
    show(q, Rag().ask(q), chunks="--chunks" in sys.argv)


def dev() -> None:
    """Interactive question loop. The main way to use the system.

        uv run dev
        uv run dev --chunks    start with passage printing on

    Loads the index once and answers repeatedly. Blank line or Ctrl-C exits.

    At the prompt:
        \\chunks   toggle passage printing without restarting
        \\last     reprint the passages behind the previous answer
    """
    from .pipeline import Rag
    rag = Rag()
    chunks = "--chunks" in sys.argv
    last = None

    print(f"{rag}\nAsk a question. Blank line or Ctrl-C to quit.")
    print(f"\\chunks toggles passage printing (now "
          f"{'on' if chunks else 'off'}); \\last reprints the previous ones.")

    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            break

        if q.lower() in ("\\chunks", "/chunks", ".chunks"):
            chunks = not chunks
            print(f"  passage printing {'on' if chunks else 'off'}")
            continue
        if q.lower() in ("\\last", "/last", ".last"):
            if last is None:
                print("  nothing asked yet")
            else:
                show_chunks(last)
            continue

        last = rag.ask(q)
        show(q, last, chunks=chunks)


def evaluate() -> None:
    """Score the pipeline against eval/questions.json.

        uv run eval                  reuse cached answers where possible
        uv run eval --fresh          re-ask everything
        uv run eval --faithfulness   also judge whether citations are honest

    Reports retrieval recall, MRR, nDCG, context precision, answer accuracy
    and refusal rate always; --faithfulness adds one extra judge call per
    answered question (cached, like everything else) to score whether each
    citation actually supports the claim it is attached to -- citation
    verification alone only checks the citation points at a real block. The
    shipped question set describes a specific corpus -- write your own to
    match your documents, or the numbers mean nothing.
    """
    from .evaluate import main as run
    run()


def compare_store() -> None:
    """Compare the native store against a real vector database (Chroma).

        uv run compare-store "your question"

    Not part of the pipeline -- this is the demonstration this project's own
    design decision invites: `rag/store.py` argues that exact search beats an
    approximate index at this corpus's size, and this command makes that
    checkable rather than merely asserted. It loads the SAME vectors into an
    actual Chroma collection (no re-embedding) and shows both stores' answers
    to the same query side by side, with latency.

    Needs the optional `chroma` extra: `uv sync --extra chroma`. The default
    install and every other command never touch it.
    """
    from .chroma_store import ChromaNotInstalled, compare
    q = " ".join(_args('usage: uv run compare-store "your question"'))
    try:
        r = compare(q)
    except ChromaNotInstalled as e:
        raise SystemExit(str(e))

    print(f"\n{r['n_vectors']} vectors, compared identically in both stores\n")
    print(f"NATIVE (SQLite + NumPy, exact cosine)   {r['native_ms']:.2f} ms")
    for h in r["native"]:
        loc = h["title"] + (f" > {h['section']}" if h["section"] else "")
        print(f"    {h['score']:.3f}  {loc}")

    print(f"\nCHROMA ({r['dir']}, approximate HNSW)   {r['chroma_ms']:.2f} ms")
    for h in r["chroma"]:
        loc = h["title"] + (f" > {h['section']}" if h["section"] else "")
        print(f"    {h['score']:.3f}  {loc}")

    native_top = {h["title"] for h in r["native"]}
    chroma_top = {h["title"] for h in r["chroma"]}
    if native_top == chroma_top:
        print("\nSame top results from both stores. At this corpus size, "
              "Chroma's approximate index costs a new dependency and a "
              "second copy of the data to reach the answer exact search "
              "already gives for free.")
    else:
        only_native = native_top - chroma_top
        print(f"\nResults differ: {len(only_native)} result(s) only the "
              f"native store found -- {', '.join(sorted(only_native))}. "
              "This is what 'approximate' means: HNSW trades a small, "
              "usually-invisible recall loss for speed at a scale this "
              "corpus has not reached.")


def main() -> None:
    print(f"""docs-rag   embed={EMBED_MODEL}@{EMBED_DIM}d   gen={GEN_MODEL}

A question-answering system over documents you choose. It answers only from
those documents, cites what it used, and refuses when they don't contain the
answer.

BUILD THE CORPUS
  uv run add <file|folder|url>   Parse a source and store it. Accepts .pdf,
                                 .md, .txt, .html, .rst, .csv, folders, and
                                 http(s) URLs (Wikipedia, blogs, Medium...).
                                 No API calls, nothing embedded yet.
                                 --replace overwrites an existing document.
  uv run ingest                  Chunk -> embed -> save the index. Re-running
                                 is cheap: only new or changed chunks are
                                 embedded. Run after every add/remove.
  uv run remove <doc_id>         Drop one document (then re-run ingest).
  uv run clear --corpus|--index|--all
                                 Delete data. Requires an explicit target.

SEE WHAT YOU HAVE
  uv run list                    Every document: id, type, size, source.
  uv run status                  Config, corpus/index sizes, whether they
                                 are in sync, vector-cache size.
  uv run inspect "question"      What retrieval returns, with scores and no
                                 generation. Use this first when an answer
                                 looks wrong -- it tells you whether the
                                 problem is retrieval or the model.

ASK
  uv run dev                     Interactive loop. Start here.
                                 \\chunks toggles passage printing, \\last
                                 reprints the previous answer's passages.
  uv run ask "question"          A single answer with citations.
  ... --chunks                   On dev or ask: also print every retrieved
                                 passage, marked CITED or unused. The fastest
                                 way to see why an answer came out as it did.
  uv run eval                    Score accuracy and grounding against
                                 eval/questions.json (--fresh to re-ask,
                                 --faithfulness to also judge citations).

DEMO / COMPARISON  (not part of the pipeline)
  RAG_CHUNK_MODE=hierarchical    Switch to a 2-level chunk hierarchy built
  (or =flat to switch back)      by rag/hierarchy.py. Each mode has its own
                                 pre-built index, so switching is instant
                                 once both exist (build both with `ingest`).
  uv run compare-store "..."     Same vectors, native store vs. a real
                                 Chroma collection, side by side. Needs
                                 `uv sync --extra chroma`.

TYPICAL SESSION
  uv run add https://en.wikipedia.org/wiki/Retrieval-augmented_generation
  uv run add ./papers/           # a folder of PDFs
  uv run ingest
  uv run dev""")


if __name__ == "__main__":
    main()
