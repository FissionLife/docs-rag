# Retrieval-Augmented Generation: a working guide

This document teaches RAG using the code in this repository as the worked
example. Every concept points at the file that implements it, so you can read
the idea and then read the twenty lines that do it.

**Contents**

1. [The problem RAG solves](#1-the-problem-rag-solves)
2. [The two halves of every RAG system](#2-the-two-halves-of-every-rag-system)
3. [Chunking](#3-chunking)
4. [Embeddings](#4-embeddings)
5. [Storing vectors](#5-storing-vectors)
6. [Retrieval](#6-retrieval)
7. [Grounded generation](#7-grounded-generation)
8. [The taxonomy: kinds of RAG system](#8-the-taxonomy-kinds-of-rag-system)
9. [Evaluation](#9-evaluation)
10. [Debugging a RAG system](#10-debugging-a-rag-system)
11. [Cost, latency and scale](#11-cost-latency-and-scale)
12. [Case study: turning a 162-page PDF into a corpus](#12-case-study-turning-a-162-page-pdf-into-a-corpus)
13. [Exercises](#13-exercises)

---

## 1. The problem RAG solves

A language model knows what was in its training data, frozen at a cutoff date,
blended into its weights with no source attribution. That gives you four
problems at once:

| Problem           | What it looks like                                             |
| ----------------- | -------------------------------------------------------------- |
| **Staleness**     | It cannot know your Q3 numbers or last week's incident report. |
| **Privacy**       | Your internal documents were never in anyone's training set.   |
| **Attribution**   | It cannot tell you _where_ an answer came from.                |
| **Hallucination** | When it doesn't know, fluency is unchanged — it invents.       |

RAG addresses all four with one move: **don't ask the model to recall, give it
the text and ask it to read.** The model stops being a knowledge store and
becomes a reading-comprehension engine over documents you control.

The important consequence, and the one this repository is built around: if the
model is only allowed to use supplied text, then _"I don't know"_ becomes a
checkable property rather than a hope. You can verify that every sentence
traces back to a passage you handed over.

### RAG vs. the alternatives

- **Fine-tuning** teaches _behavior and form_ — tone, output schema, a
  domain's idiom. It is a poor way to teach _facts_: expensive to update,
  impossible to attribute, and it still confabulates. Rule of thumb: fine-tune
  how it says things, retrieve what it says.
- **Long context** (dumping everything into the prompt) works, and is
  underrated for small corpora. It falls over on cost — you pay for every token
  on every call — on latency, and on the "lost in the middle" effect where
  facts buried mid-prompt get overlooked. Our corpus is ~370k characters,
  roughly 90k tokens. That fits in a modern context window, but you would pay
  for all of it on every question, and retrieval gives you citations for free.
- **Tool / function calling** against a real API beats RAG whenever the answer
  lives in a database rather than in prose. Don't embed your orders table.

---

## 2. The two halves of every RAG system

Every RAG system, however elaborate, is these two pipelines:

```
INDEXING  (offline, runs when documents change)
  documents ──▶ chunks ──▶ embeddings ──▶ vector store
  corpus.py     chunk.py   embed.py       store.py

QUERYING  (online, runs per question)
  question ──▶ embed ──▶ search ──▶ rank ──▶ prompt ──▶ answer + citations
               embed.py  retrieve.py        generate.py
```

`pipeline.py` is the whole system in about sixty lines: `ingest()` is the top
row, `Rag.ask()` is the bottom row. Read that file first; everything else is a
detail of one of its steps.

The asymmetry matters. Indexing is slow, batched, and paid once. Querying is
latency-critical and paid per question. Work you can push into indexing —
better chunking, richer metadata, precomputed summaries — is nearly free at
query time. That trade is the single most useful lever in RAG design.

---

## 3. Chunking

> `rag/loaders.py`, `rag/chunk.py`

### Before you can chunk: parsing

Chunking gets the attention, but the step before it decides how much there is
to chunk. A document arrives as a PDF, a web page, a Markdown file or a URL,
and how you turn that into text sets the ceiling on everything downstream.

The rule that matters: **fetch the source, not the rendering.** A rendered page
is an application — navigation, related-post rails, cookie banners, a layout
engine. The source is what the author wrote. Two measured examples from this
repository:

| Source              | Scraping the rendering       | Fetching the source                          |
| ------------------- | ---------------------------- | -------------------------------------------- |
| A GitHub README     | 12,469 chars, **0 sections** | 17,009 chars, **17 sections**                |
| A Wikipedia article | page chrome + article        | MediaWiki API plaintext with `== ==` markers |

The section counts matter more than the character counts. GitHub renders `##`
as `<h2>` inside its own layout, and boilerplate removal cannot tell the file's
headings from the page's chrome — so it strips them all, and the document
arrives as one undifferentiated blob with no breadcrumbs to give its chunks.

So `loaders.py` special-cases what is worth special-casing:

| Input                          | Handled by                      |
| ------------------------------ | ------------------------------- |
| `en.wikipedia.org/wiki/X`      | MediaWiki API                   |
| `github.com/O/R/blob/REF/path` | `raw.githubusercontent.com`     |
| `github.com/O/R`               | its README on `main`/`master`   |
| `.pdf`                         | pypdf, **one section per page** |
| any other URL, `.html`         | trafilatura article extraction  |
| `.md .txt .rst .csv`, folders  | read directly                   |

**Synthesise structure when the format has none.** A PDF has no headings, so the
PDF loader emits `## Page 12` markers. That is not cosmetic: it means a citation
can say `report.pdf > Page 12` instead of just `report.pdf`, which is the
difference between a checkable citation and a gesture at a document.

**Let a generated document declare its own identity.** Pages built by a script
know where their content really came from, but the loader would label them with
the local file path — so a citation reads `file:///C:/.../amazon-s3.md` rather
than something a reader can open. A `source_url:` line in Markdown front matter
overrides it. (Deliberately not YAML: flat `key: value` only, so nothing in a
document can execute.)

### Now, chunking

You cannot embed a 60,000-character article as one vector. A single vector has
a fixed budget of meaning; averaging an entire article into it produces
something weakly about everything and strongly about nothing. So documents are
split into chunks, and the chunk becomes the unit of retrieval.

Chunking is where most RAG systems are quietly lost. The two failure modes:

- **Chunks too large** — the embedding is diluted, retrieval gets vague, and
  you spend context tokens on irrelevant surrounding text.
- **Chunks too small** — facts get severed from the context that makes them
  interpretable. "It grows at about 1 mm per day" is useless standing alone;
  you cannot tell what _it_ is.

### The strategies, worst to best

| Strategy         | How                                                                                        | When it's right                                      |
| ---------------- | ------------------------------------------------------------------------------------------ | ---------------------------------------------------- |
| Fixed characters | Cut every N characters                                                                     | Never, really. Splits mid-word.                      |
| Fixed + overlap  | Same, with N characters repeated                                                           | Acceptable baseline.                                 |
| Recursive        | Split on paragraphs, then sentences, then words, until under budget                        | Good general default (LangChain's default).          |
| **Structural**   | Split on the document's own boundaries — headings, sections, function definitions          | **Best when structure exists.** Used here.           |
| Semantic         | Embed each sentence, cut where consecutive similarity drops                                | Expensive; helps on unstructured prose.              |
| Late chunking    | Embed the _whole_ document with a long-context embedder, then pool token vectors per chunk | Newest approach; keeps chunk vectors document-aware. |

### What this repository does

Three decisions, in `chunk.py`:

1. **Never cross a section boundary.** `_sections()` reads both heading
   dialects that reach it — Markdown `##` and MediaWiki `== ==` — so no loader
   has to rewrite its text just to be chunkable. A chunk about "Clinical
   significance" never bleeds into "Evolution".

   Nesting is tracked with a **stack**, not by indexing the path by heading
   level, and that distinction is not academic. Levels are not dense: a
   document may jump from `#` to `###`, or use `##` for every section with no
   `#` at all — which is exactly what the PDF loader's page markers do.
   Index-based nesting silently makes `## Page 2` a _child_ of `## Page 1`,
   and then of `Page 1 > Page 2 > Page 3`, so every citation after the first
   page is wrong. Headings inside fenced code blocks are ignored too: `#` in a
   Python sample is a comment, not a section.

2. **Pack whole sentences, with overlap.** `_pack()` fills a chunk to
   `CHUNK_CHARS` (1400) using complete sentences, then _backs up_ to carry the
   last `CHUNK_OVERLAP` (200) characters into the next chunk. The overlap is
   insurance: a fact stated across a sentence boundary survives intact in at
   least one chunk.

3. **Prepend a breadcrumb.** This is the highest-value line in the file:

   ```python
   "embed_text": f"{breadcrumb}\n\n{text}"   # "Myelin > Function\n\n..."
   ```

   The chunk that says "It grows at about 1 mm per day" is embedded as
   _"Schwann cell > Regeneration\n\nIt grows at about 1 mm per day"_. The
   pronoun now has an antecedent, and the vector lands near questions about
   Schwann cells instead of in the void. This trick — variously called
   contextual retrieval, context enrichment, or document-aware chunking — is
   the cheapest large retrieval win available. Note that `text` and
   `embed_text` are stored separately: the breadcrumb helps the _embedding_,
   but the reader sees it as a citation header rather than as body prose.

**Sizing.** 1400 characters is roughly 350 tokens. The rule of thumb is that a
chunk should be about the size of the answer's natural evidence — one or two
paragraphs. Both embedding models here accept far more (2,048 and 8,192
tokens), so the model's input limit is not the constraint. Retrieval precision
is.

---

## 4. Embeddings

> `rag/embed.py`

An embedding maps text to a vector such that texts with similar meaning land
near each other. "How fast do nerve impulses travel?" and "conduction velocity
of myelinated axons is 120 m/s" share almost no words but sit close in
embedding space. That is the entire magic, and its entire limitation.

Similarity is measured by **cosine similarity** — the angle between two
vectors, ignoring their length. If you normalize every vector to unit length
first, cosine similarity _is_ the dot product, so a whole-corpus search becomes
one matrix multiply. That is why `normalize()` exists, and why `retrieve.py`
can get away with `self.vectors @ qv`.

### Which Gemini embedding model?

Verified against Google's live documentation (August 2026):

| Model                      | Status                                  | Max input    | Dimensions     | `task_type`              | Modalities                     |
| -------------------------- | --------------------------------------- | ------------ | -------------- | ------------------------ | ------------------------------ |
| **`gemini-embedding-001`** | GA, shutdown **14 May 2028**            | 2,048 tokens | 128–3072 (MRL) | **Yes**                  | Text                           |
| **`gemini-embedding-2`**   | GA (22 Apr 2026), no shutdown announced | 8,192 tokens | 128–3072 (MRL) | No — use prompt prefixes | Text, image, video, audio, PDF |

Retired — do not use. All now point to `gemini-embedding-2`:
`text-embedding-004` (shut down 14 Jan 2026), `embedding-2-preview`
(10 Aug 2026), `embedding-001`, `embedding-gecko-001`, `gemini-embedding-exp`,
`gemini-embedding-exp-03-07` (all 30 Oct 2025).

Both supported models work here; switch with `RAG_EMBED_MODEL` in `.env`. The
default is `gemini-embedding-001`, for the reason immediately below.

### Task types — asymmetric embeddings

A question and its answer are not the same _kind_ of text. "Who discovered
NGF?" and "NGF was first isolated by Rita Levi-Montalcini and Stanley Cohen in
1954" are related as question-to-answer, not as paraphrases. A model trained
only for symmetric similarity will happily rank _other questions_ above the
answer.

`gemini-embedding-001` addresses this with `task_type`, which places queries
and documents into a space optimised for exactly that relationship:

```python
task_type="RETRIEVAL_DOCUMENT"   # when indexing corpus chunks
task_type="RETRIEVAL_QUERY"      # when embedding the user's question
```

Getting this backwards — or omitting it — is one of the most common silent RAG
bugs. Nothing crashes; retrieval just gets mediocre. The full set for
`gemini-embedding-001` is `RETRIEVAL_DOCUMENT`, `RETRIEVAL_QUERY`,
`SEMANTIC_SIMILARITY`, `CLASSIFICATION`, `CLUSTERING`, `CODE_RETRIEVAL_QUERY`,
`QUESTION_ANSWERING` and `FACT_VERIFICATION`.

`gemini-embedding-2` has no such parameter; the documented substitute is an
instruction prefix on the text, which `_PREFIX` in `embed.py` supplies. It is a
real trade: you gain 4× the input length and multimodality, and give up an
explicit asymmetric-retrieval mode.

### Matryoshka dimensions (MRL)

Both models are trained so that the _first_ N dimensions of a vector are
themselves a usable embedding — a Matryoshka doll. That is why you can request
768 or 1536 instead of the native 3072 and lose very little quality while
halving or quartering storage and search cost.

One sharp edge: **`gemini-embedding-001` returns truncated vectors
un-normalized**, so you must L2-normalize them yourself or cosine similarity
will be wrong. `gemini-embedding-2` renormalizes automatically. `normalize()`
is applied unconditionally in `embed.py`, which is correct for both —
normalizing an already-unit vector is a no-op.

### The two traps in the SDK

```python
# gemini-embedding-2: a list of STRINGS returns ONE aggregated vector for the
# whole list. To get one vector per input, wrap each input separately:
contents=[types.Content(parts=[types.Part.from_text(text=t)]) for t in texts]
```

This is the nastiest bug available in this API, because it fails _silently in
shape_: you asked for 32 embeddings and got 1, and if you don't check counts
you will index garbage. `_call()` asserts `len(got) == len(texts)` for exactly
this reason.

The second trap is rate limits. Embedding a corpus means thousands of requests;
429s are normal, not exceptional. `backoff.py` retries with exponential delay
and — importantly — retries _only_ transient errors, so a genuine bug fails
fast instead of taking five doublings to surface.

### The cache that makes re-indexing nearly free

Indexing is the expensive half (§2), and the naive implementation re-embeds
the entire corpus on every run. That is not merely slow — under a quota it is
fatal. The free tier allows **1,000 embedding calls per day**, so a pipeline
that re-embeds everything can be run once or twice a day and then stops
working entirely.

The fix is one design decision: make the cache **content-addressed rather than
positional**. Each text is keyed on a hash of `(model, dimension, its own
content)`:

```python
def _key(text): return sha1(f"{EMBED_MODEL}|{EMBED_DIM}|{text}")
```

The model and dimension belong in the key because vectors from different
models are not comparable and must never collide. Content belongs in the key
because that is what makes the cache survive everything else changing:

| Action                      | Embedding calls                          |
| --------------------------- | ---------------------------------------- |
| Re-run with nothing changed | **0** (~1 second)                        |
| Add one document to fifty   | only that document's chunks              |
| Edit one paragraph          | only the chunks that actually changed    |
| Change the chunk size       | all of them — every chunk's text changed |
| Switch embedding model      | all of them — correctly, the key changed |

That last pair is the point: the cache invalidates itself on exactly the
changes that _should_ invalidate it, without any explicit versioning. A
positional checkpoint (row 137 of 559) cannot do this — it only helps within a
single run, and is worthless the moment the corpus changes.

Deduplication comes free: identical text embeds once however often it appears.

### Other embedding options

Not used here, but worth knowing so you can judge the trade:

- **OpenAI** `text-embedding-3-small` / `-large` — also MRL-truncatable.
- **Voyage AI** (`voyage-3`) and **Cohere** (`embed-v4`) — consistently strong
  on retrieval benchmarks; Cohere supports int8 and binary quantized output.
- **Open weights, run locally**: `BAAI/bge-m3`, `intfloat/e5-large-v2`,
  `nomic-embed-text-v1.5`, `Alibaba-NLP/gte`. Free, private, no rate limits,
  and good enough that the gap to hosted models is usually smaller than the gap
  caused by bad chunking. Check the **MTEB leaderboard** for current standings
  — but weight the _retrieval_ subset for RAG, not the overall average.

Whatever you pick: **embedding models are not interchangeable after the fact.**
Vectors from two models are not comparable, so changing the model means
re-indexing everything. That is why `store.py` writes each `(model, dimension)`
pair to its own directory and refuses to load a mismatched index.

---

## 5. Storing vectors

> `rag/store.py`

The store keeps two artifacts side by side:

- `chunks.db` — SQLite: text, title, section, URL, and `idx`, the row number.
- `vectors.npy` — a float32 `(n_chunks, dim)` matrix. Row `idx` belongs to the
  chunk with that `idx`. That correspondence is the whole design.

Search is then a brute-force matrix multiply and a sort. With 777 chunks at
1536 dimensions that is a 4.5 MB matrix, and search takes well under a
millisecond. **No vector database is required here, and using one would be
slower and less accurate.**

### When to graduate to a real vector database

Exact search stays the right answer for longer than people expect — roughly up
to a million vectors on a machine with enough RAM. Move when you hit one of:

| Signal                               | What you need                  | Options                                        |
| ------------------------------------ | ------------------------------ | ---------------------------------------------- |
| Matrix no longer fits in RAM         | On-disk ANN index              | FAISS, DiskANN                                 |
| Search latency too high              | Approximate index (HNSW / IVF) | FAISS, Qdrant, Milvus                          |
| Concurrent writers, live updates     | A server                       | Qdrant, Weaviate, Milvus                       |
| Filtered search alongside SQL        | Postgres extension             | pgvector, pgvectorscale                        |
| You don't want to run infrastructure | Hosted                         | Pinecone, Turbopuffer, Vertex AI Vector Search |

Understand what you give up: ANN indexes are _approximate_. They trade recall
for speed, tuned by parameters like HNSW's `ef_search`. A retrieval regression
after "just swapping in a vector DB" is usually an under-tuned index, not a
model problem.

---

## 6. Retrieval

> `rag/retrieve.py`

This is where answer quality is mostly won or lost. The generator cannot use a
passage it never received.

### Dense retrieval and its blind spot

Embedding search understands paraphrase and synonymy. It is systematically weak
on rare literal tokens — identifiers, gene symbols, error codes, part numbers,
proper nouns. `MPZ`, `Nav1.6` and `NRG1` are near-noise to an embedding model,
but they are exactly the terms a specialist searches with.

### Sparse retrieval: BM25

BM25 scores documents by exact term overlap, weighting each term by how rare it
is across the corpus (**IDF**) and damping the effect of repetition and
document length. It has no idea what words _mean_, which is precisely why it
complements dense search: it matches `NRG1` perfectly and `neuregulin` not at
all.

The implementation here is ~40 lines with no dependency (`class BM25`). The
scoring formula, term by term:

```
score(q, d) = Σ  IDF(t) · f(t,d)·(k1+1) / ( f(t,d) + k1·(1 − b + b·|d|/avgdl) )
             t∈q
```

`k1` (1.5) controls how fast repeated terms stop helping; `b` (0.75) controls
how much long documents are penalised. The postings-list layout means scoring
only touches documents that actually contain a query term.

### Fusing them: Reciprocal Rank Fusion

Cosine similarity lives in ~0.5–0.9. BM25 scores are unbounded and
corpus-dependent. You cannot add them. You _can_ combine their **ranks**:

```
RRF(d) = Σ  1 / (k + rank_r(d))          k = 60
        r∈retrievers
```

Because it uses only ordinal position, RRF needs no calibration, no tuning and
no score normalization — and it beats most weighted-sum schemes people hand-
tune. `k=60` is the value from the original paper and is a fine default; it
damps the influence of the very top rank so that one retriever cannot dominate.

### MMR: stop retrieving the same paragraph six times

Because chunks overlap and articles repeat themselves, the top 6 hits are often
six views of one passage. That wastes context and hides the _second_ fact a
question needs. **Maximal Marginal Relevance** picks each next chunk by

```
argmax [ λ · sim(chunk, query) − (1 − λ) · max sim(chunk, already_selected) ]
```

with `λ = 0.7` here — mostly relevance, with a penalty for redundancy. Note in
`search()` that MMR decides _membership_ of the final set, after which fused-
score order is restored for presentation; the generator reads top-down, so the
best evidence should still come first.

### Deduplication: a step MMR doesn't cover

MMR discourages picking two similar chunks into the *same* top-k, but it still
lets both **compete** for a slot — a near-duplicate can win purely on
relevance and only get penalized against a genuinely different second choice.
A true near-duplicate carries no second fact, so the right fix is to remove it
before ranking even starts, not to out-argue it during selection.

This is not theoretical on the AWS corpus. "Amazon EC2" and "Amazon EC2 Image
Builder" share a near-identical metadata header at **cosine 0.957** — same
category, same launch date, same boilerplate summary line — and 42
cross-document chunk pairs on this corpus exceed 0.95. `_dedup()` in
`retrieve.py` collapses any pair above `DEDUP_COSINE` (0.95, keeping the
higher-fused-score one) out of the candidate pool *before* MMR runs:

```python
pool = _dedup(self.vectors, pool, fused)
chosen = _mmr(qv, self.vectors, pool, top_k, MMR_LAMBDA)
```

Worth noting how the threshold was chosen: an initial guess of 0.97 measured
clean in testing but missed the real EC2 pair sitting at 0.957 — a reminder
that a dedup threshold needs to be checked against actual near-duplicates in
your corpus, not picked by intuition and left unverified.

### What we deliberately left out: reranking

The strongest single upgrade to this pipeline would be a **cross-encoder
reranker**. Bi-encoders (what we use) embed query and document _separately_, so
the vectors can be precomputed — fast, but the model never sees the pair
together. A cross-encoder reads query and document _jointly_ and scores the
pair. Far more accurate, far too slow to run over 777 chunks — so you use it as
a second stage: retrieve 30 candidates cheaply, rerank them expensively, keep 6.

Options: Cohere Rerank, Voyage rerank, Jina reranker, `BAAI/bge-reranker-v2-m3`
(open weights), or an LLM prompted to score relevance. The typical gain is
larger than anything else on this list. See Exercise 4.

---

## 7. Grounded generation

> `rag/generate.py`

"Only answer from the context" is not a prompt. It is four independent layers,
and the prompt is the weakest of them.

**Layer 1 — the retrieval gate.** `Rag.ask()` compares the best cosine against
`MIN_COSINE` (0.55). If nothing clears it, we refuse _without calling the
generator at all_. A model that is never invoked cannot hallucinate. This
catches out-of-domain questions cleanly, and costs nothing.

**Layer 2 — the prompt contract.** `SYSTEM` in `generate.py` forbids outside
knowledge, mandates an `[n]` citation on every factual sentence, and specifies
an exact sentinel — `INSUFFICIENT_CONTEXT` — to emit when the context falls
short. Two details do a lot of work here. Giving refusal an _exact string_
makes it unambiguous to detect, rather than something you regex for ("I'm
sorry, I don't…"). And explicitly blessing _partial_ answers stops the model
having to choose between over-claiming and refusing outright.

**Layer 3 — decoding.** `temperature=0.0`. Sampling temperature is, quite
literally, the knob that trades faithfulness for variety. For grounded
extraction you want none of it.

**Layer 4 — post-hoc citation verification.** This is the layer that holds when
the others fail. After generation, every `[n]` is parsed, and:

- an answer citing **nothing** is rejected and converted into a refusal — an
  uncited claim is unverifiable, and unverifiable is indistinguishable from
  invented;
- citations pointing at blocks that were never supplied are stripped and
  reported.

That is a mechanical check, not a request. It is why this pipeline can make a
real claim about grounding rather than a hopeful one.

### What this still does not guarantee

Being honest about the limits: citation _presence_ is verified, citation
_correctness_ is not. The model could cite `[3]` for a sentence that block 3
does not support. Closing that gap requires an entailment check — a second
model call asking "does block 3 entail this sentence?" — which is what
faithfulness-scoring frameworks (RAGAS, TruLens, DeepEval) automate. That is
the natural next layer, and Exercise 5.

---

## 8. The taxonomy: kinds of RAG system

Everything below is a variation on the same two pipelines, roughly in order of
how much machinery it adds.

### 8.1 Naive / vanilla RAG

Chunk → embed → top-k cosine → stuff into the prompt. The baseline every
tutorial shows. Fails on acronyms and rare tokens, on multi-part questions, and
on anything requiring more than one document.

### 8.2 Hybrid RAG — **what this repository is**

Dense + sparse, fused. The best effort-to-benefit upgrade over naive RAG. Add
MMR for diversity. Add metadata filtering (date, author, permissions) — which
in real systems is usually more important than any ranking subtlety, because
"only search documents this user is allowed to read" is a hard requirement, not
a quality tweak.

### 8.3 Two-stage retrieval (reranking)

Retrieve broadly and cheaply, rerank narrowly and expensively. See §6. If you
add exactly one thing to this codebase, add this.

### 8.4 Query transformation

The user's question is often a poor search query. Rewrite it before retrieving:

- **Multi-query** — generate 3–5 paraphrases, retrieve for each, fuse with RRF.
  Cheap and robust; covers vocabulary you did not anticipate.
- **HyDE** (Hypothetical Document Embeddings) — ask the model to _invent_ an
  answer, then embed the invention and search with it. A fake answer looks more
  like a real answer than a question does, which sidesteps the asymmetry
  problem from §4. (It can also invent misleading terms — measure it.)
- **Query decomposition** — split "How do Schwann cells and oligodendrocytes
  differ in how many axons they myelinate?" into two sub-questions, retrieve
  separately, answer jointly. Necessary for genuine multi-hop questions.
- **Step-back prompting** — ask a more general question first to retrieve
  background, then the specific one.
- **Conversational rewriting** — in a chat, "what about in the CNS?" is
  meaningless standing alone. Rewrite it against the history _before_
  retrieving. Skipping this is the most common bug in RAG chatbots.

### 8.4a Query routing — **also in this repo**

Not every question is a retrieval question. "Summarise this", "what topics do
you cover", "how many documents are there" are about the _collection_, and no
chunk contains that answer, so similarity search returns noise (measurably:
cosine 0.57–0.59, BM25 0.00) and a well-built system refuses. Correctly — but
uselessly.

A router classifies the question first and picks the context accordingly.
`rag/overview.py` does the cheap deterministic version: narrow regex patterns
send corpus-level questions to a manifest built from stored metadata plus each
document's lead section, under its own system prompt (`SYSTEM_OVERVIEW`) that
treats summarising the context as answerable — which is safe, because the
context _is_ the material being asked about.

Two design notes worth stealing. First, the router is deliberately narrow: a
router that fires too eagerly sends factual questions down a path that cannot
answer them, which is worse than not firing at all — so "summarise the role of
Schwann cells in remyelination" stays on the retrieval route. Second, the two
routes get _separate_ prompts rather than one permissive prompt, so the strict
factual contract is never loosened to accommodate summarisation.

Bigger versions of this idea classify across several indexes (docs vs. code vs.
tickets), or route between RAG, SQL and plain chat.

### 8.5 Small-to-big / sentence-window / parent-document

Decouple the unit you _search_ from the unit you _read_. Embed small precise
chunks (a sentence) for retrieval accuracy, but pass the surrounding window or
the whole parent section to the generator for comprehension. This resolves the
chunk-size dilemma of §3 instead of compromising on it.

### 8.6 Contextual retrieval

Before embedding, prepend a short LLM-generated description of how each chunk
fits into its document. Anthropic reported large reductions in retrieval
failures from this combined with BM25. Expensive at index time — one LLM call
per chunk — and free at query time: the indexing/querying trade from §2 in its
purest form. The breadcrumb in `chunk.py` is the cheap, deterministic version
of this idea.

### 8.7 Hierarchical / RAPTOR

Cluster chunks, summarise each cluster, embed the summaries, recurse — building
a tree from detail up to abstraction. Retrieval can then draw from any level,
so "what is this whole corpus about?" and "which chromosome carries the MBP
gene?" are both answerable. The standard fix for questions that no single chunk
answers.

`rag/overview.py` is the free, deterministic degenerate case of this: a
two-level tree whose leaves are chunks and whose single summary node is built
from metadata and human-written lead sections, at zero LLM cost. RAPTOR is what
you build when your documents have no such summaries to borrow.

### 8.8 Graph RAG

Extract entities and relations into a knowledge graph, then retrieve by
traversing it. Strong for multi-hop questions about relationships ("which
diseases affect cells derived from the neural crest?") and for global
summarisation. Costly to build and to maintain; justify it before adopting it.

### 8.9 Agentic RAG / Self-RAG / Corrective RAG (CRAG)

Give the model control over retrieval instead of hard-wiring a single round:

- **Self-RAG** — the model decides _whether_ to retrieve, then critiques
  whether the retrieved passages actually support its draft.
- **CRAG** — grade the retrieved documents; if they are poor, reformulate and
  retrieve again, or fall back to web search.
- **Agentic / multi-hop** — retrieval as a tool the model may call repeatedly,
  reading its way to an answer over several turns.

More capable, and much harder to make fast, cheap and predictable. The
grounding guarantees get harder to state, too.

### 8.10 Multimodal RAG

Index images, tables, audio and video alongside text. `gemini-embedding-2` maps
all of these into one shared space, so a text query can retrieve an image
directly. The common alternative is to caption non-text content with a vision
model and embed the captions.

### 8.11 Structured / SQL RAG (text-to-query)

When the answer lives in a database, translate the question into SQL or an API
call and run it. Not really retrieval-augmented _generation_, but it is the
right answer far more often than people admit.

### 8.12 Cache-augmented generation (long context)

Skip retrieval; put the whole corpus in the prompt and rely on prompt caching
to make the repetition cheap. Genuinely competitive below a few hundred
thousand tokens if the corpus is static. You lose citations and pay in latency.

### Choosing

| Situation                             | Start with                       |
| ------------------------------------- | -------------------------------- |
| Corpus under ~100 documents, static   | Long context, or naive RAG       |
| Technical docs, identifiers, acronyms | **Hybrid (this repo)**           |
| Quality plateaued, budget available   | Add a reranker                   |
| Questions span multiple documents     | Query decomposition, then RAPTOR |
| Questions are about relationships     | Graph RAG                        |
| No chunk size satisfies everyone      | Small-to-big                     |
| Users chat rather than ask one-shot   | Conversational query rewriting   |
| Retrieval quality is erratic          | Contextual retrieval + reranking |

---

## 9. Evaluation

> `rag/evaluate.py`, `eval/questions.json`

**A RAG system you have not measured does not work — you just haven't found out
yet.** Vibes-based tuning fails because the failures are quiet: retrieval
returning plausible-but-wrong chunks looks exactly like retrieval working.

Measure the stages separately, because they fail independently.

### Retrieval metrics

- **Recall@k** — is the right document in the top _k_? This is the ceiling on
  everything downstream. Measure it first; if it is low, no prompt will help.
  Binary and coarse: rank 1 and rank 6 score identically.
- **MRR** (mean reciprocal rank) — how high did it rank? `1/rank`, averaged.
  Distinguishes "always first" from "usually barely scrapes into the top-k",
  which recall alone cannot.
- **nDCG@k** — rank quality, with a logarithmic rank discount. Normally used
  with *graded* relevance (this chunk is a 3/5 match, that one a 1/5); this
  harness computes the **binary-relevance case** instead — one correct
  document per question, present or absent — which needs no extra grading
  effort beyond what recall already has and still rewards rank over mere
  presence. It is not a lesser nDCG, just the same formula with `rel ∈ {0,1}`.
- **Context precision** — what fraction of retrieved chunks were actually used?
  Low precision wastes tokens and distracts the generator.

### Generation metrics

- **Faithfulness / groundedness** — is every claim supported by the context?
  The metric that matters most for our purpose.
- **Answer relevance** — does it address the question actually asked?
- **Correctness** — does it match a known-good answer?

### What this harness does

`rag/evaluate.py` reports over 31 questions:

- **retrieval recall@6** — did the expected document reach the context?
- **MRR / nDCG@6** — *where* in the top-6 it landed, not just whether it did.
  Computed from a live `retriever.search()` call rather than the cached
  generation answer, because rank needs the ordered hit list and re-running
  retrieval only costs an embedding call (plentiful) rather than a generation
  call (the scarce, quota-limited resource). On this corpus both currently
  read **1.000** — every doc-grounded question's correct document lands at
  rank 1, not merely somewhere in the top 6.
- **answer accuracy** — checked by requiring specific literal strings (`1971`,
  `Levi-Montalcini`, `8.8`) rather than fuzzy similarity. Exact-match checks on
  numbers and names are unglamorous, and they catch what embedding-similarity
  scoring hides.
- **refusal rate on unanswerable questions**, split into two kinds:
  - _out-of-domain_ ("capital of Mongolia") — should be stopped by the gate.
  - _adjacent-absent_ ("in what year was Theodor Schwann born?") — the topic
    **is** in the corpus, so the gate passes and retrieval returns confident,
    on-topic, useless chunks. The model knows the answer from pretraining, and
    only the prompt contract and the citation check stand between it and a
    leak. This is the honest test of grounding, and the one most eval sets
    omit.

Also tracked: **false refusals**. Grounding strictness always trades against
helpfulness — a system that refuses everything scores perfectly on leakage. You
must watch both numbers, or you will tune yourself into uselessness.

### An eval set is coupled to its corpus

This is the failure mode nobody warns you about. A question set is written
against one specific collection of documents. Swap the corpus and the
questions do not become _wrong_ — they become **meaningless**, and the
distinction matters because the output looks identical either way.

This repository's first eval set asked about myelin and Guillain-Barré
syndrome. Point the same harness at a corpus of AWS service pages and
retrieval recall collapses to near zero, answer accuracy follows, and the
report reads exactly like a broken retrieval pipeline. Every number is
"correct"; every number is about nothing.

Two consequences worth internalising:

- **A score is only meaningful next to the corpus it was measured on.** "100%
  accuracy" with no corpus, model and date attached is not a claim, it is a
  decoration. Record the configuration with the number.
- **Say so in the tool.** `rag/evaluate.py` prints an explicit warning when
  recall or accuracy falls below 70%, telling the reader to check that the
  question set still describes the indexed documents _before_ debugging the
  pipeline. A harness that reports a low score without that hint sends people
  hunting for a bug in code that is working.

The corollary: when you change corpus, rewrite the questions. It is an hour of
work, and skipping it costs you the ability to tell whether anything works.

### Going further

Frameworks worth knowing: **RAGAS** (faithfulness, context precision/recall),
**TruLens**, **DeepEval**, **promptfoo**. Benchmark datasets: BEIR (retrieval),
MTEB (embeddings), Natural Questions and HotpotQA (multi-hop). And build a
golden set from _your users' real questions_ — 50 hand-labelled real questions
beat 5,000 synthetic ones.

---

## 10. Debugging a RAG system

Always answer this question first: **is it retrieval, or generation?**

```bash
uv run inspect "your question"          # retrieval only, no generation
uv run ask "your question" --chunks     # the answer AND what it was given
```

`inspect` prints the retrieved chunks with their cosine, BM25 and RRF scores
and never calls the generator. `--chunks` prints the same passages _alongside_
the answer, marking each **CITED** or **unused**.

The **unused** ones are the informative half. They show what retrieval found
and the generator then ignored, which is the only way to distinguish two
failures that look identical from the answer alone:

- a refusal with six on-topic passages → the model had the evidence and did
  not use it (a generation or prompt problem);
- a refusal with six irrelevant passages → retrieval never found it.

That distinction is the whole first question of debugging, and it is invisible
without seeing the context. Then:

**The right chunk isn't in the list** → a retrieval problem. Check, in order:

- Is the fact even in the corpus? (`grep` for it. Half of all reported
  "hallucinations" are actually missing documents.)
- Did chunking sever it from its context? Read the chunk that _should_ have
  matched.
- Rare token? Look at the BM25 column — if BM25 missed it too, the term may be
  tokenized oddly.
- Is the query phrased very differently from the source? Try multi-query or
  HyDE.
- Are you using the right `task_type`? Swapping query and document is silent.

**The right chunk is there but ranked below the cut** → a ranking problem.
Raise `TOP_K`, or add a reranker (§6).

**The right chunk is in the context but the answer is wrong** → a generation
problem. Look for contradicting chunks, tighten the prompt, verify
`temperature=0`, and check whether the fact is buried in the middle of a long
context.

**It refuses when it shouldn't** → `MIN_COSINE` is too high, or the answer
genuinely isn't in the corpus. Check `best_cosine` in the output.

**It answers when it shouldn't** → the interesting failure. Work out which
layer let it through: was the gate passed because the question was
adjacent-absent? Did the model cite a block that doesn't support the claim?
That is the entailment gap from §7.

A note on cosine values: they are **not** probabilities, and their useful range
is model-specific. 0.55 is a reasonable gate for `gemini-embedding-001` on this
corpus; it is not a universal constant. Calibrate it by measuring scores for
known-good and known-bad questions and putting the threshold in the gap.

Here is that calibration, measured on this corpus by `uv run eval`:

| Question kind                                  | Best cosine   | What caught it          |
| ---------------------------------------------- | ------------- | ----------------------- |
| Out-of-domain (capital of Mongolia)            | 0.465 – 0.523 | the gate                |
| Adjacent-absent (Theodor Schwann's birth year) | 0.621 – 0.735 | prompt + citation check |
| Answerable                                     | 0.72 – 0.78   | —                       |

Two lessons fall straight out of that table. First, 0.55 is not arbitrary: it
sits in the empty band between out-of-domain and everything else. Second, and
more important — **the adjacent-absent range overlaps the answerable range.**
No threshold can separate them, because retrieval genuinely _did_ find the
right document; the document simply does not contain that particular fact.
This is precisely why the gate cannot be the only defence, and why layer 4
exists.

---

## 11. Cost, latency and scale

Per question, this pipeline costs:

| Step            | Cost                            | Latency     |
| --------------- | ------------------------------- | ----------- |
| Embed the query | 1 embedding call, ~20 tokens    | ~100–200 ms |
| Vector search   | free (local numpy)              | < 1 ms      |
| BM25            | free (local)                    | ~1 ms       |
| Generation      | ~2–4k input tokens, ~200 output | ~1–3 s      |

Generation dominates both columns.

### The free tier is tighter than you expect

Worth knowing before you plan a run, because both limits are enforced per
model and were hit while building this:

- **Embeddings** are metered in _individual texts_, not HTTP requests: a batch
  of 32 spends 32 units against a **100-per-minute** quota. Indexing 777 chunks
  therefore takes ~8 minutes no matter how you batch it. `RateLimiter` in
  `backoff.py` reserves capacity before sending rather than discovering the
  ceiling by rejection — which matters, because rejected requests still count.
- **Generation** has a **per-day** cap as well as a per-minute one — 20/day for
  `gemini-3.5-flash` at time of writing, which a 26-question eval exhausts.
  A daily quota cannot be waited out, yet the API still replies "please retry
  in 38s". `with_retry` detects the `PerDay` quota id and raises immediately
  instead of sleeping through five useless doublings.

Two consequences shaped this codebase: ingest checkpoints after every batch,
and the eval caches each answer to disk the moment it arrives. Both exist so
that hitting a quota costs you time, not work already paid for.

Practical levers:

- **Cache query embeddings.** Repeated questions are common in production.
- **Prompt caching** on the system instruction, if your provider supports it.
- **Smaller `TOP_K`** cuts input tokens linearly. Measure recall before cutting.
- **Truncate embeddings** (MRL, §4) — 768 instead of 3072 quarters your storage
  and search cost for a small quality loss.
- **Quantize.** float32 → int8 cuts memory 4× with minor recall loss; binary
  quantization cuts it 32× and is often used as a fast first-stage filter.
- **Batch at index time.** `EMBED_BATCH = 32` here; indexing is where you
  spend, so make it batched and resumable.

Indexing this corpus: 777 chunks, 25 API calls, a few seconds, and about 4.5 MB
of vectors at 1536 dimensions.

---

## 12. Case study: turning a 162-page PDF into a corpus

> `tools/aws_build.py`

Most RAG tutorials start from documents that are already clean. Real ones start
from a PDF someone emailed you. This is the whole path for one, and every
problem in it is typical rather than exotic.

**The task.** Take *Overview of Amazon Web Services* — 162 pages describing
every AWS service — and build a corpus with one summary page per service.

### The conversion loses everything you needed

`markitdown` converts the PDF to Markdown and produces **zero headings**. Not
"some headings" — none. The output is a flat stream of text with page headers
and footers mixed into the prose, and the chunker described in §3 has nothing
to split on. Every service would land in one enormous undifferentiated blob.

This is the normal outcome. PDF is a layout format; it stores where glyphs go,
not what they mean. A heading is "18pt bold with space above" and any converter
has to guess. Expect to reconstruct structure rather than extract it.

### Use the document's own schema

The whitepaper has a table of contents — 272 entries listing every category and
every service **in reading order**. That is precisely the segmentation the body
text lacks. So: parse the TOC to get the schema, then walk the body and start a
new section whenever a line matches a known name.

The general lesson: when a document loses its structure in conversion, look for
somewhere the structure is *stated* rather than *formatted*. A table of
contents, an index, a manifest, a sitemap.

### Then find out where the schema lies

Two failures that a quick eyeball would not catch:

**Sub-headings that masquerade as categories.** The TOC reads `Compute`, then
`Compare AWS compute services`, then `Amazon EC2`. Taking the most recent
heading as the category files EC2 — and all 14 compute services — under
"Compare AWS compute services", and the `Compute` category silently ends up
empty. The fix is a small ignore-list, but the lesson is to *check category
counts*, because nothing errors.

**The TOC is incomplete.** `Blockchain`, `Game tech` and `Serverless` have
services in the body and no TOC entries at all. A TOC-only extractor drops them
without a word — the worst kind of failure, because the output looks complete.

The recovery exploits a habit of the prose: the whitepaper opens nearly every
section by restating the service name.

```
Amazon MQ
Amazon MQ is a managed message broker service for Apache ActiveMQ...
```

A standalone line whose text the *next* line begins with is almost certainly a
section heading. That single heuristic recovered 25 missing services. Document
conventions are structure too, when the formatting is gone.

### Enrich from the source of truth, and prefer its API

The brief also asked for more than the PDF held. The obvious approach — scrape
350 product pages — is slow, fragile and rude. But AWS renders its own product
directory from a JSON endpoint, so **one request** returns all 358 products with
canonical URL, category, pricing page, launch date and free-tier status.

Always look for the API behind the page before writing a scraper. It is faster,
more stable, kinder to the host, and gives you fields the HTML never showed.

Matching the two sources is fuzzy — the whitepaper says "Amazon EC2" where the
catalogue says "Amazon Elastic Compute Cloud" — so normalise, then fall back to
containment. That reached 224 of 244; the rest keep their whitepaper text and
simply lack the extra metadata. **Partial enrichment is fine.** Do not let an
unmatched 8% block the other 92%.

### Result

244 services, 23 categories, 559 chunks, and citations that link to
aws.amazon.com. The eval in `eval/questions.json` was rewritten against this
corpus — per §9, keeping the old questions would have made every number
meaningless.

### The transferable checklist

1. Convert, then **look at what survived**. Count the headings.
2. If structure is gone, find where the document *states* it.
3. Verify the schema against the body; assume it is incomplete.
4. Check counts per group. Silent misfiling does not raise.
5. Look for the host's API before scraping its HTML.
6. Accept partial enrichment.
7. Rewrite the eval set for the new corpus.

---

## 13. Exercises

Roughly in order of value gained per hour spent.

1. **Measure a change.** Set `RAG_EMBED_DIM=768`, re-ingest into its own index
   directory, run `uv run eval`. How much accuracy did you lose for 2× less storage?
   Now try 3072. Was it worth it?

2. **Swap the embedding model.** Set `RAG_EMBED_MODEL=gemini-embedding-2`,
   re-ingest, re-evaluate. You are trading `task_type` for longer input and
   multimodality — does it show on these questions?

3. **Ablate hybrid search.** Comment out the BM25 half of the fusion in
   `retrieve.py` and re-run. Which questions break? (Look at the ones with gene
   names and eponyms.) Then ablate the dense half instead.

4. **Add a reranker.** Retrieve 30, ask a Gemini Flash model to score each
   candidate 0–10 for relevance, keep the top 6. Compare recall and accuracy.
   This is the biggest available win.

5. **Close the entailment gap.** After generating, make a second call per
   sentence: "Does block [n] support this claim? yes/no." Reject unsupported
   sentences. Measure how many were wrong — the number is usually not zero.

6. **Break the gate deliberately.** Set `MIN_COSINE=0.0` and re-run `uv run eval`.
   Watch which unanswerable questions now leak, and see how much work layers
   2–4 were doing on their own.

7. **Implement small-to-big.** Embed individual sentences, but pass the parent
   chunk to the generator. Does precision improve?

8. **Add multi-query.** Generate three paraphrases, retrieve for each, fuse
   with the existing RRF. Measure recall on the questions that currently miss.

9. **Change the corpus.** Point `TITLES` in `corpus.py` at a different domain,
   write a new eval set, and find out which of your tuning decisions were about
   RAG and which were about neuroscience.
