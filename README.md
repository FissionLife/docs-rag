# docs-rag

A small, complete RAG pipeline over **documents you choose** — PDFs, Markdown,
plain text, HTML, Wikipedia pages, blog and Medium articles — built so that
**the model can only answer from those documents**, and so that the claim is
verified mechanically rather than merely requested in a prompt.

Learning the concepts: **[docs/RAG_GUIDE.md](docs/RAG_GUIDE.md)**.

Three dependencies: `google-genai`, `numpy`, `python-dotenv`. No LangChain, no
vector database, no framework. Managed with [uv](https://docs.astral.sh/uv/).

## Setup

```bash
cp .env.example .env      # then put your Gemini key in it
```

Get a key at <https://aistudio.google.com/apikey>. There is no install step —
`uv run` creates the environment from `uv.lock` on first use.

```bash
uv run add ./paper.pdf ./notes/ https://en.wikipedia.org/wiki/CRISPR
uv run ingest        # chunk -> embed -> store
uv run dev           # ask questions
```

`uv run help` prints all of this at any time.

## Commands

### Build the corpus

| Command | Does |
|---|---|
| `uv run add <file\|folder\|url>` | Parse a source and store it. **No API calls.** `--replace` overwrites. |
| `uv run ingest` | Chunk → embed → save the index. Cheap to re-run (see below). |
| `uv run remove <doc_id>` | Drop one document, then re-ingest. |
| `uv run clear --corpus\|--index\|--all` | Delete data. Requires an explicit target. |

**`add` accepts:** `.pdf` `.md` `.txt` `.html` `.rst` `.csv`, any folder
(searched recursively), and any `http(s)` URL. PDFs are split by page, so a
citation reads `report.pdf > Page 12`.

Some URLs are worth special-casing, because scraping the rendered page is
strictly worse than fetching the source:

| URL | Handled by | Why |
|---|---|---|
| `en.wikipedia.org/wiki/X` | MediaWiki API | Clean plaintext with real section markers |
| `github.com/O/R/blob/REF/path` | `raw.githubusercontent.com` | The blob page is an app, not a document — scraping it loses **every heading** |
| `github.com/O/R` | its README on `main`/`master` | The obvious thing to paste |
| anything else | trafilatura | Strips nav, related-post rails, cookie banners |

The GitHub case is worth spelling out: scraping
`github.com/.../README.md` yielded 12,469 characters and **0 sections**, because
GitHub renders headings as `<h2>` inside its own layout and boilerplate removal
cannot tell the file's headings from the page's chrome. Fetching the raw file
yielded 17,009 characters and **17 sections** — and section breadcrumbs are most
of what makes a chunk retrievable (see [the guide, §3](docs/RAG_GUIDE.md#3-chunking)).

### See what you have

| Command | Does |
|---|---|
| `uv run list` | Every document: id, type, size, source. |
| `uv run status` | Config, corpus/index sizes, whether they are in sync. |
| `uv run inspect "..."` | What retrieval returns, with scores, **no generation**. |

`inspect` is the one to reach for when an answer looks wrong: it separates
"the passage was never retrieved" from "the model had it and answered badly".
`--chunks` answers the same question *alongside* the answer rather than instead
of it — and the **unused** passages are the informative half, because they show
what retrieval found and the generator then ignored. In `dev`, `\chunks`
toggles it mid-session and `\last` reprints the previous answer's passages.

### Ask

| Command | Does |
|---|---|
| `uv run dev` | Interactive loop. Start here. |
| `uv run ask "..."` | A single answer with citations. |
| `... --chunks` | On either: also print every retrieved passage, marked **CITED** or **unused** |
| `uv run eval` | Score accuracy and grounding (`--fresh` to re-ask). |

These are console entry points declared in `[project.scripts]`; uv has no
npm-style script table, so that is the idiomatic equivalent.

## Re-running ingest is nearly free

Embeddings are cached **by content**, keyed on a hash of
`(model, dimension, chunk text)`. So:

- re-running `ingest` with nothing changed embeds **0** chunks (~1 second);
- adding one document embeds only **that document's** chunks;
- editing a document re-embeds only the chunks that actually changed.

This is not just a speed optimisation. The free tier allows **1000 embedding
calls per day**, so a pipeline that re-embeds everything on every run can only
be run once or twice a day before it stops working.

## How it works

```
INDEXING   corpus.py ─▶ chunk.py ─▶ embed.py ─▶ store.py
           Wikipedia   section-    Gemini      SQLite +
           API         aware       embeddings  vectors.npy

QUERYING   overview.py ─▶ embed.py ─▶ retrieve.py ─▶ generate.py
           route the      query       dense + BM25,   strict prompt,
           question       vector      RRF, MMR        citation check
              │
              └─ corpus-level ("summarise", "what topics") bypasses search
                 and answers from a manifest + document leads instead
```

| File | Responsibility |
|---|---|
| `rag/config.py` | Every tunable value, in one place |
| `rag/loaders.py` | Parse PDF / Markdown / text / HTML / URLs into documents |
| `rag/corpus.py` | The document collection: add, list, remove |
| `rag/chunk.py` | Section-aware, sentence-packing chunker with breadcrumbs |
| `rag/embed.py` | Gemini embeddings; handles both supported models' quirks |
| `rag/store.py` | Persistent store: SQLite metadata + numpy matrix |
| `rag/retrieve.py` | Hybrid dense/BM25 retrieval, RRF fusion, MMR diversity |
| `rag/overview.py` | Query routing; corpus-level context from metadata + leads |
| `rag/generate.py` | Grounded answering and citation verification |
| `rag/pipeline.py` | `ingest()` and `Rag.ask()` — the system in 60 lines |
| `rag/backoff.py` | Retry on transient errors; proactive quota rate limiting |
| `rag/cli.py` | The `uv run` entry points |
| `rag/evaluate.py` | Retrieval recall, answer accuracy, refusal rate |

## Measured results

`uv run eval`, 29 questions, 777 chunks, `gemini-embedding-001` @ 1536d +
`gemini-3.5-flash-lite`:

| Metric | Result |
|---|---|
| Retrieval recall@6 | **20/20 (100%)** |
| Answer accuracy (exact facts present) | **20/20 (100%)** |
| Refusal on unanswerable | **9/9 (100%)** |
| — out-of-domain | 4/4, all stopped by the relevance gate |
| — adjacent-absent | 5/5, all stopped by the prompt + citation check |
| False refusals | **0** |
| Elapsed | 125s |

The split in that table is the interesting part. The four out-of-domain
questions scored cosine 0.465–0.523 and never reached the model. The five
*adjacent-absent* ones — questions whose topic is in the corpus but whose
answer is not, like "in what year was Theodor Schwann born?" — scored
0.621–0.735, comfortably **above** the 0.55 gate. The gate let all five
through; layers 2–4 refused them. A single-layer system would have leaked
every one.

Answerable questions scored 0.72–0.78, so the threshold sits in a real gap
rather than on top of the distribution.

### Two kinds of question

Similarity search answers questions *about a fact*. It cannot answer questions
*about the collection* — "summarise this", "what topics are covered", "what
are the titles" — because no single chunk contains that answer. Embedding such
a query returns unrelated paragraphs at indistinguishable scores (measured:
cosine 0.57–0.59, **BM25 exactly 0.00**), and refusing on them is correct, not
a bug.

So `rag/overview.py` routes them to a different context: a manifest built from
stored metadata (exact titles, counts, section headings) plus the lead section
of every document — which on Wikipedia is a human-written summary, so it costs
nothing to produce. Grounding is unchanged: same citation verification, same
refusal sentinel, and the manifest is data we stored rather than data the model
recalled. The router is deliberately narrow — *"summarise the role of Schwann
cells in remyelination"* still goes to normal retrieval.

## How "only from context" is enforced

Four independent layers, because a prompt alone is not an enforcement
mechanism:

1. **Retrieval gate** — if no chunk clears `MIN_COSINE`, refuse without ever
   calling the generator. A model that isn't invoked cannot hallucinate.
2. **Prompt contract** — outside knowledge forbidden, a `[n]` citation required
   on every factual sentence, and an exact sentinel to emit when the context is
   insufficient.
3. **`temperature = 0`** — no sampling creativity during extraction.
4. **Citation verification** — every `[n]` is parsed and checked against what
   was actually supplied. An answer that cites nothing is rejected and
   converted into a refusal; citations of blocks that were never shown are
   stripped.

Layer 4 is what makes the guarantee mechanical. See
[the guide, §7](docs/RAG_GUIDE.md#7-grounded-generation) for what it still does
*not* guarantee.

## Configuration

All defaults live in `rag/config.py`; the three most useful are overridable
from `.env`:

```bash
RAG_EMBED_MODEL=gemini-embedding-001   # or gemini-embedding-2
RAG_EMBED_DIM=1536                     # 128-3072, MRL-truncatable
RAG_GEN_MODEL=gemini-3.6-flash
RAG_EMBED_RPM=95                       # free tier allows 100 texts/minute
```

Each `(model, dimension)` pair gets its own index directory under
`data/index/`, so you can build several and compare them without re-ingesting.

### Choosing the generation model

| Model | Grounding behaviour | Daily quota |
|---|---|---|
| `gemini-3.6-flash` **(default)** | Follows the full rule set, including partial answers | Tighter |
| `gemini-3.5-flash` | Same | 20/day |
| `gemini-3.5-flash-lite` | **Over-refuses** — see below | Generous |

Flash-Lite was the default until it was caught refusing *"what is the best X"*
with the subject plainly in the context. It obeys the hard rules (only use the
context, cite, refuse when empty) but not the subtler ones (answer the part
that is supported, say what is not). **Over-refusal is the worst failure mode
here**, because the system looks correctly grounded while being useless — and
unlike a hallucination, nothing in the output tells you it went wrong. Use
Flash-Lite only when the daily quota on flash actually bites.

## Free-tier quotas

Both limits are **per day** and **per model**, and both were hit while building
this:

| | Limit | Notes |
|---|---|---|
| Embeddings | 1000/day, 100/minute | Counted in *individual texts*, not requests |
| Generation | 20/day for `gemini-3.5-flash` | Flash-Lite is far more generous |

A per-day cap cannot be waited out, yet the API still replies "please retry in
38s". The code detects the `PerDay` quota id and fails immediately with the fix
rather than sleeping through useless retries. **Each model has its own separate
allowance**, so switching `RAG_EMBED_MODEL` or `RAG_GEN_MODEL` in `.env` is the
quickest unblock.

## The shipped eval set

`eval/questions.json` was written for a specific corpus (Wikipedia
neuroscience). It is kept as a **worked example of how to write one** — the
answerable/unanswerable split, and the `out-of-domain` vs `adjacent-absent`
distinction, are the parts worth copying. Its numbers mean nothing against your
documents; write your own questions against your own corpus.
