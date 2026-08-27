# docs-rag

A retrieval-augmented question-answering system over **documents you choose** —
PDFs, Markdown, plain text, HTML, Wikipedia pages, GitHub files, blog articles.
It answers **only** from those documents, cites what it used, and refuses when
they do not contain the answer — and that claim is verified mechanically rather
than merely requested in a prompt.

Currently indexed: **244 AWS services**, built from the *Overview of Amazon Web
Services* whitepaper and enriched from the AWS product catalogue.

Learning the concepts: **[docs/RAG_GUIDE.md](docs/RAG_GUIDE.md)** — 13 sections
from chunking to evaluation, including a case study of building this corpus.

Five dependencies. No LangChain, no vector database, no framework. Managed with
[uv](https://docs.astral.sh/uv/).

## Setup

```bash
cp .env.example .env      # then put your Gemini key in it
```

Get a key at <https://aistudio.google.com/apikey>. There is no install step —
`uv run` builds the environment from `uv.lock` on first use.

```bash
uv run add ./paper.pdf ./notes/ https://en.wikipedia.org/wiki/CRISPR
uv run ingest        # chunk -> embed -> store
uv run dev           # ask questions
```

`uv run help` prints every command at any time.

## Commands

### Build the corpus

| Command | Does |
|---|---|
| `uv run add <file\|folder\|url>` | Parse a source and store it. **No API calls.** `--replace` overwrites. |
| `uv run ingest` | Chunk → embed → save the index. Cheap to re-run (see below). |
| `uv run remove <doc_id>` | Drop one document, then re-ingest. |
| `uv run clear --corpus\|--index\|--all` | Delete data. Requires an explicit target. |

**`add` accepts** `.pdf` `.md` `.txt` `.html` `.rst` `.csv`, any folder
(searched recursively), and any `http(s)` URL. PDFs are split by page, so a
citation reads `report.pdf > Page 12`.

Some URLs are special-cased, because fetching the source beats scraping the
rendering:

| URL | Handled by | Why |
|---|---|---|
| `en.wikipedia.org/wiki/X` | MediaWiki API | Clean plaintext with real section markers |
| `github.com/O/R/blob/REF/path` | `raw.githubusercontent.com` | The blob page is an app — scraping it loses **every heading** |
| `github.com/O/R` | its README on `main`/`master` | The obvious thing to paste |
| anything else | trafilatura | Strips nav, related-post rails, cookie banners |

Measured on one README: scraping gave 12,469 chars and **0 sections**; fetching
the raw file gave 17,009 chars and **17 sections**. Section breadcrumbs are most
of what makes a chunk retrievable.

### See what you have

| Command | Does |
|---|---|
| `uv run list` | Every document: id, type, size, source |
| `uv run status` | Config, corpus/index sizes, whether they are in sync |
| `uv run inspect "..."` | What retrieval returns, with scores, **no generation** |

### Ask

| Command | Does |
|---|---|
| `uv run dev` | Interactive loop. Start here. |
| `uv run ask "..."` | A single answer with citations |
| `... --chunks` | On either: also print every retrieved passage, marked **CITED** or **unused** |
| `uv run eval` | Score accuracy and grounding (`--fresh` to re-ask) |

`inspect` and `--chunks` are the debugging pair: they separate *"the passage was
never retrieved"* from *"the model had it and answered badly"*. The **unused**
passages are the informative half. In `dev`, `\chunks` toggles printing
mid-session and `\last` reprints the previous answer's passages.

## The AWS corpus

Rebuild it from the whitepaper with [tools/aws_build.py](tools/aws_build.py):

```bash
# 1. PDF -> Markdown (one-off tool, isolated so it stays out of the project deps)
uv tool run --python 3.13 --from "markitdown[pdf]" \
    markitdown aws-overview.pdf -o data/aws/aws-overview.md

# 2. recover structure, enrich from the AWS portal, write one page per service
uv run python tools/aws_build.py parse      # -> 244 services, 23 categories
uv run python tools/aws_build.py catalog    # -> 358 products from the AWS API
uv run python tools/aws_build.py pages      # -> data/aws/services/*.md

# 3. index it
uv run add data/aws/services/
uv run ingest
```

The interesting part is that **markitdown produces zero headings** — PDF is a
layout format, so a heading is just bold text. The document's own table of
contents is used as a segmentation schema instead, and services the TOC omits
(Blockchain, Game tech, Serverless) are recovered from the body, which restates
each service name in its opening sentence.

Enrichment uses AWS's own product-directory **JSON API** — one request for all
358 products with canonical URL, category, pricing page, launch date and
free-tier status — rather than scraping 350 HTML pages. 224 of 244 matched;
the rest keep their whitepaper description.

The full write-up is [the guide, §12](docs/RAG_GUIDE.md#12-case-study-turning-a-162-page-pdf-into-a-corpus).

## Re-running ingest is nearly free

Embeddings are cached **by content**, keyed on a hash of
`(model, dimension, chunk text)`:

- re-running `ingest` unchanged embeds **0** chunks (~1 second);
- adding one document embeds only **that document's** chunks;
- editing a paragraph re-embeds only the chunks that changed;
- changing chunk size or embedding model correctly invalidates everything.

Not just a speed win. The free tier allows **1,000 embedding calls per day**, so
a pipeline that re-embeds everything can be run twice a day before it stops
working.

## How it works

```
INDEXING   loaders.py ─▶ chunk.py ─▶ embed.py ─▶ store.py
           PDF/MD/HTML   section-    Gemini      SQLite +
           /URL parsing  aware       embeddings  vectors.npy

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
| `rag/corpus.py` | The document collection: add, list, remove, dedupe |
| `rag/chunk.py` | Section-aware, sentence-packing chunker with breadcrumbs |
| `rag/embed.py` | Gemini embeddings + the content-addressed cache |
| `rag/store.py` | Persistent store: SQLite metadata + numpy matrix |
| `rag/retrieve.py` | Hybrid dense/BM25 retrieval, RRF fusion, MMR diversity |
| `rag/overview.py` | Query routing; corpus-level context from metadata + leads |
| `rag/generate.py` | Grounded answering and citation verification |
| `rag/pipeline.py` | `ingest()` and `Rag.ask()` — the system in ~100 lines |
| `rag/backoff.py` | Retry on transient errors; proactive quota rate limiting |
| `rag/cli.py` | The `uv run` entry points |
| `rag/evaluate.py` | Retrieval recall, answer accuracy, refusal rate |
| `tools/aws_build.py` | Build the AWS corpus from the whitepaper |

## Measured results

`uv run eval` — 31 questions over 244 documents / 559 chunks, with
`gemini-embedding-2` @ 1536d and `gemini-3.5-flash-lite`:

| Metric | Result |
|---|---|
| Retrieval recall@6 | **22/22 (100%)** |
| Answer accuracy (exact facts present) | **22/22 (100%)** |
| Refusal on unanswerable | **9/9 (100%)** |
| — out-of-domain | 4/4, stopped by the relevance gate |
| — adjacent-absent | 5/5, stopped by the prompt + citation check |
| False refusals | **0** |
| Elapsed | 257s |

`gemini-3.7-flash` reached 18/18 on the same set before hitting its daily quota,
so this is not a Flash-Lite-specific result.

**The split is the interesting part.** *Out-of-domain* questions ("capital of
Peru") never reach the model — the relevance gate stops them. *Adjacent-absent*
questions are the honest test: the topic **is** in the corpus but the fact is
not, so retrieval returns confident, on-topic, useless chunks and the gate lets
them through. "What is the maximum execution timeout for a Lambda function?"
retrieves the Lambda page, which never states a timeout. Only the prompt
contract and the citation check stand between that and a leak.

> A score is only meaningful next to the corpus it was measured on. These
> questions were written against **this** AWS corpus. Change the documents and
> rewrite `eval/questions.json`, or the numbers describe nothing — `uv run eval`
> prints a warning when scores drop below 70% for exactly this reason.

## How "only from context" is enforced

Four independent layers, because a prompt is not an enforcement mechanism:

1. **Retrieval gate** — if no chunk clears `MIN_COSINE`, refuse without ever
   calling the generator. A model that isn't invoked cannot hallucinate.
2. **Prompt contract** — outside knowledge forbidden, a `[n]` citation required
   on every factual sentence, and an exact sentinel for insufficient context.
3. **`temperature = 0`** — no sampling creativity during extraction.
4. **Citation verification** — every `[n]` is parsed and checked against what
   was actually supplied. An answer citing nothing is rejected and converted
   into a refusal; citations of blocks never shown are stripped.

Layer 4 is what makes the guarantee mechanical. See
[the guide, §7](docs/RAG_GUIDE.md#7-grounded-generation) for what it still does
*not* guarantee.

## Configuration

Defaults live in `rag/config.py`; the useful ones are overridable from `.env`:

```bash
RAG_EMBED_MODEL=gemini-embedding-2     # or gemini-embedding-001
RAG_EMBED_DIM=1536                     # 128-3072, MRL-truncatable
RAG_GEN_MODEL=gemini-3.6-flash
RAG_EMBED_RPM=95                       # free tier allows 100 texts/minute
RAG_AWS_PAGE_URL=aws                   # AWS citations: `aws` or `local`
```

Each `(model, dimension)` pair gets its own index directory under `data/index/`,
so you can build several and compare without re-ingesting.

### Choosing the generation model

| Model | Grounding behaviour | Daily quota |
|---|---|---|
| `gemini-3.6-flash` **(default)** | Follows the full rule set | ~20/day |
| `gemini-3.7-flash` | Same | ~20/day |
| `gemini-3.5-flash-lite` | **Can over-refuse** — see below | Generous |

Flash-Lite was the default until it was caught refusing *"what is the best X"*
with the subject plainly in the context. It obeys the hard rules (only use the
context, cite, refuse when empty) but not always the subtler ones (answer the
part that *is* supported, say what is not). **Over-refusal is the worst failure
mode here** — the system looks correctly grounded while being useless, and
nothing in the output tells you it went wrong.

## Free-tier quotas

Both limits are **per day** and **per model**, and both were hit building this:

| | Limit | Notes |
|---|---|---|
| Embeddings | 1,000/day, 100/minute | Counted in *individual texts*, not requests |
| Generation | ~20/day per flash model | Flash-Lite is far more generous |

A per-day cap cannot be waited out, yet the API still replies "please retry in
38s". The code detects the `PerDay` quota id and fails immediately with the fix
rather than sleeping through useless retries. **Each model has its own
allowance**, so switching `RAG_EMBED_MODEL` or `RAG_GEN_MODEL` is the quickest
unblock.
