"""Central configuration. Everything tunable lives here, and everything in
this file is overridable from .env -- nothing is a hardcoded constant a
change requires editing code for.
"""
import os
import re
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CORPUS_DIR = DATA / "corpus"
INDEX_DIR = DATA / "index"

# --- Models -----------------------------------------------------------------
# Embedding model. Two supported, with different capabilities:
#   gemini-embedding-001 : text-only, 2048 input tokens, supports `task_type`
#                          (asymmetric query/document embeddings). GA.
#   gemini-embedding-2   : multimodal, 8192 input tokens, NO task_type,
#                          auto-normalizes truncated dimensions. GA, newest.
EMBED_MODEL = os.getenv("RAG_EMBED_MODEL", "gemini-embedding-001")

# Matryoshka (MRL) truncation. Native is 3072; 768/1536/3072 are recommended.
EMBED_DIM = int(os.getenv("RAG_EMBED_DIM", "1536"))

# gemini-2.5-flash is listed by the API but closed to new keys.
#
# flash-lite was the default until it was caught over-refusing: asked "what is
# the best X", with the subject plainly in the context at cosine 0.79, it
# emitted the refusal sentinel instead of answering what the documents do say.
# It obeys rules 1-3 of the system prompt but not the subtler 4-6. Over-refusal
# is the worst failure mode here, because the system looks correctly grounded
# while being useless, and nothing in the output signals that it went wrong.
# 3.6-flash and 3.5-flash both handle it, so correctness wins over quota.
#
# Set RAG_GEN_MODEL=gemini-3.5-flash-lite only for high-volume runs where the
# tighter daily quota on flash actually bites, and expect more refusals.
GEN_MODEL = os.getenv("RAG_GEN_MODEL", "gemini-3.6-flash")

# Fallback generation models, tried in order, each with every configured key
# exhausted first (see rag/keys.py) before moving to the next. Off by
# default -- deliberately not auto-defaulted to flash-lite, because that
# model's over-refusal problem above is a real quality trade a fallback
# should never make silently. Opt in explicitly, e.g.:
#   RAG_GEN_MODEL_FALLBACK=gemini-3.5-flash,gemini-3.5-flash-lite
# Falls back on two conditions only, deliberately narrow (see generate.py):
# every key exhausted for this model today, or the model itself being
# unavailable (a 404, e.g. a model closed to new keys -- this project hit
# exactly that with gemini-2.5-flash). Never on a generic error, which would
# just mask a real bug behind N models' worth of retries.
def _parse_model_list(env_var: str, primary: str) -> list[str]:
    raw = os.getenv(env_var, "")
    fallbacks = [m.strip() for m in re.split(r"[,\n]", raw) if m.strip()]
    seen, ordered = set(), []
    for m in [primary] + fallbacks:
        if m not in seen:
            seen.add(m)
            ordered.append(m)
    return ordered


GEN_MODELS = _parse_model_list("RAG_GEN_MODEL_FALLBACK", GEN_MODEL)

# One key, or several to round-robin across. The daily generation quota is
# as low as ~20 requests/day *per key, per model* on the free tier -- a
# single key runs out fast. Both GEMINI_API_KEYS and GEMINI_API_KEY accept
# a comma- or newline-separated list -- a real key never contains a comma,
# so splitting on one is unambiguous either way, and it means putting
# several keys directly into GEMINI_API_KEY (rather than renaming it to
# GEMINI_API_KEYS) also just works. See rag/keys.py for the rotation itself.
def _parse_keys() -> list[str]:
    raw = os.getenv("GEMINI_API_KEYS") or os.getenv("GEMINI_API_KEY") or ""
    return [k.strip() for k in re.split(r"[,\n]", raw) if k.strip()]


API_KEYS = _parse_keys()
API_KEY = API_KEYS[0] if API_KEYS else None   # back-compat for direct readers

# --- Chunking ---------------------------------------------------------------
CHUNK_CHARS = int(os.getenv("RAG_CHUNK_CHARS", "1400"))       # ~350 tokens
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "200"))    # so facts aren't cut in half
MIN_CHUNK_CHARS = int(os.getenv("RAG_MIN_CHUNK_CHARS", "120"))  # drop fragments smaller than this

# "flat" (default): the chunks in chunk.py, nothing more.
# "hierarchical": flat chunks (the leaves) PLUS one synthesized summary chunk
# per cluster of related leaves (see hierarchy.py) -- a small RAPTOR-style
# tree, two levels deep. Each mode gets its own index directory (see
# store.index_dir), so switching between them is instant once both have been
# built: change this and reload, no re-ingest. Build both with
# `RAG_CHUNK_MODE=hierarchical uv run ingest` in addition to the default.
CHUNK_MODE = os.getenv("RAG_CHUNK_MODE", "flat")

# Aim for clusters of about this many leaf chunks when building the
# hierarchical index. Smaller -> more, narrower clusters; larger -> fewer,
# broader ones. Tune per corpus size, not per query.
CLUSTER_TARGET_SIZE = int(os.getenv("RAG_CLUSTER_SIZE", "20"))

# --- Embedding requests -----------------------------------------------------
EMBED_BATCH = int(os.getenv("RAG_EMBED_BATCH", "32"))         # texts per API call
EMBED_MAX_RETRIES = int(os.getenv("RAG_EMBED_MAX_RETRIES", "6"))

# The free tier counts *individual texts*, not HTTP calls: a batch of 32
# spends 32 units against a 100-per-minute quota. Stay just under it rather
# than discovering the ceiling by being rejected.
EMBED_ITEMS_PER_MIN = int(os.getenv("RAG_EMBED_RPM", "95"))

# Generation is limited per *minute* and, on the free tier, per *day* as well
# (20/day/model at time of writing). The per-minute cap is what this throttles;
# the daily one is surfaced as an error, because it cannot be waited out.
GEN_PER_MIN = int(os.getenv("RAG_GEN_RPM", "10"))

# --- Retrieval --------------------------------------------------------------
CANDIDATES = int(os.getenv("RAG_CANDIDATES", "30"))   # each retriever proposes this many before fusion
TOP_K = int(os.getenv("RAG_TOP_K", "6"))               # chunks actually shown to the generator
RRF_K = int(os.getenv("RAG_RRF_K", "60"))              # reciprocal-rank-fusion smoothing constant
MMR_LAMBDA = float(os.getenv("RAG_MMR_LAMBDA", "0.7"))  # 1.0 = pure relevance, 0.0 = pure diversity

# Above this cosine, two candidate chunks are the same information twice, not
# two chunks that happen to agree. Distinct from MIN_COSINE (query relevance)
# and from MMR_LAMBDA (which discourages redundancy but still lets both
# compete for a slot) -- this removes one of the pair before ranking even
# starts. Measured on the AWS corpus: 42 cross-document chunk pairs exceed
# 0.95, e.g. "Amazon EC2" and "Amazon EC2 Image Builder" share a near-
# identical metadata header at 0.957 -- which is why the threshold sits at
# 0.95 and not higher: 0.97 measured clean but missed the real case.
DEDUP_COSINE = float(os.getenv("RAG_DEDUP_COSINE", "0.95"))

# Below this fused-cosine score we assume the corpus has nothing relevant
# and refuse *before* spending a generation call.
MIN_COSINE = float(os.getenv("RAG_MIN_COSINE", "0.55"))
